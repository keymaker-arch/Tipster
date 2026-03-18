"""Asyncio-based crawler scheduler — Phase 1+2+3.

Polls the DB for URLs due for crawling and dispatches crawl coroutines.
Each slice:
  1. Reset budget
  2. Score pending links (from previous slice budget exhaustion)
  3. Extract pending content items
  4. Crawl due URLs (triage + link discovery, budget-gated)
Celery Beat will replace this in Phase 5 (Production Hardening).
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timedelta, timezone
from typing import Optional

from tipster.budget import BudgetGate
from tipster.config import TipsterConfig
from tipster.crawler import CrawlResult, fetch
from tipster.db.repositories.content_items import ContentItemRepo
from tipster.db.repositories.reports import ReportRepo
from tipster.db.repositories.url_registry import UrlRegistryRepo
from tipster.db.session import get_db
from tipster.events import Event, EventBus, EventKind
from tipster.triage import triage_async

log = logging.getLogger("tipster.scheduler")

# Sentinel used for one_time URLs so they are never re-queued by list_due
_FAR_FUTURE = datetime(9999, 1, 1, tzinfo=timezone.utc)


def _next_check_at(now: datetime, interval_secs: int, recrawl_type: str) -> datetime:
    """Return the next scheduled crawl time.

    For one_time URLs this is always _FAR_FUTURE, effectively retiring the URL
    after its first fetch regardless of content outcome.
    """
    if recrawl_type == "one_time":
        return _FAR_FUTURE
    return now + timedelta(seconds=interval_secs)


class CrawlStats:
    """Mutable stats shared between scheduler and TUI."""

    def __init__(self) -> None:
        self.last_crawl_at: Optional[datetime] = None
        self.next_crawl_at: Optional[datetime] = None
        self.crawled_total: int = 0
        self.relevant_total: int = 0
        self.tokens_used: int = 0
        self.cost_usd: float = 0.0
        self.running: bool = False


async def _process_url(
    url_id: int,
    url: str,
    cfg: TipsterConfig,
    topic_id: int,
    bus: EventBus,
    stats: CrawlStats,
    budget: BudgetGate,
    llm_sem: asyncio.Semaphore,
) -> None:
    """Fetch one URL, triage it, and persist results."""
    log.debug("FETCH  url_id=%d  %s", url_id, url)
    await bus.emit(Event(kind=EventKind.CRAWL_START, url=url, message="fetching…"))

    result: CrawlResult = await fetch(url, default_delay=cfg.crawl.default_delay_seconds)
    log.debug(
        "FETCH DONE  url=%s  http=%s  ok=%s  text_len=%d  hash=%s",
        url,
        result.status_code,
        result.ok,
        len(result.text or ""),
        result.content_hash[:16] if result.content_hash else "—",
    )

    now = datetime.now(timezone.utc)
    relevant = False
    link_data: list[tuple[str, str]] = []

    db = get_db()
    try:
        url_repo = UrlRegistryRepo(db)
        entry = url_repo.get_by_id(url_id)
        if entry is None:
            return

        if result.inaccessible:
            new_interval = entry.check_interval * 2
            url_repo.update_after_crawl(
                url_id=url_id,
                last_checked=now,
                next_check_at=_next_check_at(now, new_interval, entry.recrawl_type),
                check_interval=new_interval,
                status="inaccessible",
            )
            await bus.emit(
                Event(
                    kind=EventKind.CRAWL_ERROR,
                    url=url,
                    message=f"HTTP {result.status_code} — marked inaccessible",
                )
            )
            return

        if not result.ok:
            # Apply backoff so transient failures don't cause tight retry loops
            new_interval = min(entry.check_interval * 2, 7 * 24 * 3600)
            url_repo.update_after_crawl(
                url_id=url_id,
                last_checked=now,
                next_check_at=_next_check_at(now, new_interval, entry.recrawl_type),
                check_interval=new_interval,
                status=entry.status,  # keep current status
            )
            log.debug("CRAWL ERROR  url=%s  error=%s", url, result.error or result.status_code)
            await bus.emit(
                Event(
                    kind=EventKind.CRAWL_ERROR,
                    url=url,
                    message=result.error or f"HTTP {result.status_code}",
                )
            )
            return

        stats.crawled_total += 1

        # --- Empty content check ---
        if not result.text:
            new_interval = min(entry.check_interval * 2, 7 * 24 * 3600)
            url_repo.update_after_crawl(
                url_id=url_id,
                last_checked=now,
                next_check_at=_next_check_at(now, new_interval, entry.recrawl_type),
                check_interval=new_interval,
                status=entry.status,
            )
            log.debug("EMPTY CONTENT  url=%s  html_len=%d", url, len(result.raw_html))
            await bus.emit(
                Event(
                    kind=EventKind.CRAWL_ERROR,
                    url=url,
                    message="empty content — no text extracted from page",
                )
            )
            return

        # --- Novelty check (Phase 2: cross-URL deduplication) ---
        content_repo = ContentItemRepo(db)
        existing = content_repo.get_by_hash(result.content_hash)
        log.debug(
            "NOVELTY  url=%s  hash=%s  existing_item=%s",
            url,
            result.content_hash[:16],
            existing.item_id if existing else "none",
        )

        if existing is not None:
            if existing.url_id == url_id:
                # Same URL, same content → stale, double the interval
                new_interval = min(entry.check_interval * 2, 7 * 24 * 3600)
                url_repo.update_after_crawl(
                    url_id=url_id,
                    last_checked=now,
                    next_check_at=_next_check_at(now, new_interval, entry.recrawl_type),
                    check_interval=new_interval,
                    status="active",
                )
                await bus.emit(
                    Event(
                        kind=EventKind.CRAWL_SKIP,
                        url=url,
                        message=f"unchanged (interval doubled to {new_interval}s)",
                    )
                )
            else:
                # Different URL, same content → cross-URL duplicate
                content_repo.add(
                    topic_id=topic_id,
                    url_id=url_id,
                    content_hash=result.content_hash,
                    raw_text=None,
                    topic_score=existing.topic_score,
                    is_new_source=False,
                )
                # Mark just-created item as duplicate
                dup_item = content_repo.get_by_hash(result.content_hash)
                # get the newest item for this url_id
                from tipster.db.models import ContentItem
                dup = (
                    db.query(ContentItem)
                    .filter_by(url_id=url_id, content_hash=result.content_hash)
                    .order_by(ContentItem.item_id.desc())
                    .first()
                )
                if dup:
                    content_repo.mark_duplicate(dup.item_id, existing.item_id)
                new_interval = max(int(entry.check_interval * 0.75), 60)
                url_repo.update_after_crawl(
                    url_id=url_id,
                    last_checked=now,
                    next_check_at=_next_check_at(now, new_interval, entry.recrawl_type),
                    check_interval=new_interval,
                    status="active",
                )
                await bus.emit(
                    Event(
                        kind=EventKind.CRAWL_DUPLICATE,
                        url=url,
                        message=f"duplicate of item_id={existing.item_id}",
                    )
                )
            return

        # --- Relevance triage (budget-gated) ---
        log.debug(
            "TRIAGE START  url=%s  budget=[tokens=%d cost=$%.4f]",
            url, budget.tokens_used, budget.cost_usd,
        )
        async with llm_sem:
            relevant, score, reason = await triage_async(result.text, cfg, budget)
        # Sync budget stats to CrawlStats for TUI display
        stats.tokens_used = budget.tokens_used
        stats.cost_usd = budget.cost_usd
        log.debug(
            "TRIAGE RESULT  url=%s  relevant=%s  score=%.4f  reason=%s  budget=[tokens=%d cost=$%.4f]",
            url, relevant, score, reason, budget.tokens_used, budget.cost_usd,
        )

        if "budget exhausted" in reason:
            await bus.emit(
                Event(
                    kind=EventKind.EXTRACT_DEFERRED,
                    url=url,
                    message="triage deferred — budget exhausted",
                )
            )
            # Don't update last_checked so it stays due next slice
            return

        if not relevant:
            new_interval = max(int(entry.check_interval * 0.75), 60)
            url_repo.update_after_crawl(
                url_id=url_id,
                last_checked=now,
                next_check_at=_next_check_at(now, new_interval, entry.recrawl_type),
                check_interval=new_interval,
                status="active",
            )
            await bus.emit(
                Event(
                    kind=EventKind.TRIAGE_IRRELEVANT,
                    url=url,
                    score=score,
                    message=f"score={score:.2f} — {reason[:80]}",
                )
            )
            return

        # --- Relevant — save to content_items ---
        stats.relevant_total += 1

        from sqlalchemy import func
        from tipster.db.models import ContentItem, UrlRegistry as UrlReg
        prior = (
            db.query(func.count(ContentItem.item_id))
            .join(UrlReg, ContentItem.url_id == UrlReg.url_id)
            .filter(
                ContentItem.topic_id == topic_id,
                UrlReg.domain == entry.domain,
            )
            .scalar()
        )
        is_new_source = (prior == 0)

        saved_item = content_repo.add(
            topic_id=topic_id,
            url_id=url_id,
            content_hash=result.content_hash,
            raw_text=result.text,
            topic_score=score,
            is_new_source=is_new_source,
        )
        log.debug(
            "SAVED  item_id=%d  url=%s  score=%.4f  is_new_source=%s  text_len=%d",
            saved_item.item_id, url, score, is_new_source, len(result.text or ""),
        )

        new_interval = max(int(entry.check_interval * 0.75), 60)
        url_repo.update_after_crawl(
            url_id=url_id,
            last_checked=now,
            next_check_at=_next_check_at(now, new_interval, entry.recrawl_type),
            check_interval=new_interval,
            status="active",
            is_new_source=is_new_source,
        )

        await bus.emit(
            Event(
                kind=EventKind.TRIAGE_RELEVANT,
                url=url,
                score=score,
                message=f"score={score:.2f} — saved{'  [new source]' if is_new_source else ''}",
            )
        )

        # Capture link_data before closing DB session
        link_data = result.link_data

    finally:
        db.close()

    stats.last_crawl_at = now

    # --- Link discovery (Phase 3) — outside the DB session ---
    if result.ok and relevant and link_data:
        log.debug("LINK DISCOVERY  source=%s  candidates=%d", url, len(link_data))
        from tipster.link_scorer import discover_links
        async with llm_sem:
            await discover_links(
                text=result.text,
                link_data=link_data,
                source_url=url,
                topic_id=topic_id,
                cfg=cfg,
                budget=budget,
                bus=bus,
            )
        stats.tokens_used = budget.tokens_used
        stats.cost_usd = budget.cost_usd
        log.debug(
            "LINK DISCOVERY DONE  source=%s  budget=[tokens=%d cost=$%.4f]",
            url, budget.tokens_used, budget.cost_usd,
        )


class CrawlScheduler:
    """Polls DB for due URLs and dispatches crawl tasks.

    Each slice:
      1. Reset budget gate
      2. Run extract_pending (process items saved in the previous slice)
      3. Crawl all due URLs (triage budget-gated)
    """

    POLL_INTERVAL = 30  # seconds between scheduler ticks

    def __init__(
        self,
        cfg: TipsterConfig,
        topic_id: int,
        bus: EventBus,
        stats: CrawlStats,
        db_path: str,
    ) -> None:
        self._cfg = cfg
        self._topic_id = topic_id
        self._bus = bus
        self._stats = stats
        self._db_path = db_path
        self._active_tasks: set[asyncio.Task] = set()
        self._running = False
        # Semaphores are created lazily in run() once the event loop is active.
        self._crawl_sem: Optional[asyncio.Semaphore] = None
        self._llm_sem: Optional[asyncio.Semaphore] = None
        self._budget = BudgetGate(
            max_tokens=cfg.budget.max_tokens_per_slice,
            max_cost_usd=cfg.budget.max_cost_per_slice_usd,
        )
        self._last_report_checked: Optional[datetime] = None

    async def run(self) -> None:
        self._running = True
        self._stats.running = True
        self._crawl_sem = asyncio.Semaphore(self._cfg.crawl.max_crawl_workers)
        self._llm_sem = asyncio.Semaphore(self._cfg.crawl.max_llm_workers)
        log.info(
            "Scheduler started (max_crawl_workers=%d, max_llm_workers=%d)",
            self._cfg.crawl.max_crawl_workers,
            self._cfg.crawl.max_llm_workers,
        )

        # Immediate first cycle
        await self._crawl_due_urls()

        while self._running:
            await asyncio.sleep(self.POLL_INTERVAL)
            await self._crawl_due_urls()

    async def _crawl_due_urls(self) -> None:
        from tipster.extractor import extract_pending
        from tipster.link_scorer import score_pending_links
        from tipster.directives_consumer import apply_directives

        now = datetime.now(timezone.utc)
        self._stats.next_crawl_at = now + timedelta(seconds=self.POLL_INTERVAL)

        # Reset budget at the start of each slice
        self._budget.reset()
        log.debug("SLICE START  budget reset  max_tokens=%d  max_cost=$%.2f",
                  self._cfg.budget.max_tokens_per_slice,
                  self._cfg.budget.max_cost_per_slice_usd)

        # Step 0: apply any pending directives from the previous feedback cycle
        await apply_directives(self._topic_id, self._cfg, self._bus)

        # Step 1: score any links deferred from the previous slice
        await score_pending_links(self._topic_id, self._cfg, self._budget, self._bus)

        # Step 2: extract any items left pending from a previous interrupted run
        await extract_pending(self._cfg, self._topic_id, self._budget, self._bus)
        self._stats.tokens_used = self._budget.tokens_used
        self._stats.cost_usd = self._budget.cost_usd

        # Step 3: crawl due URLs
        db = get_db()
        try:
            url_repo = UrlRegistryRepo(db)
            due = url_repo.list_due(self._topic_id, now)
        finally:
            db.close()

        if not due:
            await self._bus.emit(
                Event(kind=EventKind.SCHEDULER_TICK, message="scheduler tick — no URLs due")
            )
            # Still check for a report in case extraction in step 2 produced new items
            await self._maybe_generate_report()
            return

        await self._bus.emit(
            Event(
                kind=EventKind.SCHEDULER_TICK,
                message=f"scheduler tick — {len(due)} URL(s) due [{self._budget.summary}]",
            )
        )

        assert self._crawl_sem is not None
        assert self._llm_sem is not None

        crawl_tasks: list[asyncio.Task] = []
        for entry in due:
            # Capture primitive values before the session might expire
            url_id = entry.url_id
            url = entry.url

            async def _guarded(uid: int = url_id, u: str = url) -> None:
                async with self._crawl_sem:  # type: ignore[union-attr]
                    await _process_url(
                        url_id=uid,
                        url=u,
                        cfg=self._cfg,
                        topic_id=self._topic_id,
                        bus=self._bus,
                        stats=self._stats,
                        budget=self._budget,
                        llm_sem=self._llm_sem,  # type: ignore[arg-type]
                    )

            task = asyncio.create_task(_guarded())
            self._active_tasks.add(task)
            task.add_done_callback(self._active_tasks.discard)
            crawl_tasks.append(task)

        # Wait for all crawl tasks to finish before extracting — this ensures
        # content items saved by this batch are extracted immediately rather
        # than waiting for the next scheduler tick.
        log.debug("CRAWL BATCH  waiting for %d task(s) to complete", len(crawl_tasks))
        await asyncio.gather(*crawl_tasks, return_exceptions=True)
        log.debug("CRAWL BATCH DONE  budget=[tokens=%d cost=$%.4f]",
                  self._budget.tokens_used, self._budget.cost_usd)

        # Step 4: extract items that were just saved by this crawl batch
        await extract_pending(self._cfg, self._topic_id, self._budget, self._bus)
        self._stats.tokens_used = self._budget.tokens_used
        self._stats.cost_usd = self._budget.cost_usd
        log.debug("SLICE END  budget=[tokens=%d cost=$%.4f]",
                  self._budget.tokens_used, self._budget.cost_usd)

        # Step 5: check if a report should be generated now that new items are extracted
        await self._maybe_generate_report()

    async def _maybe_generate_report(self) -> None:
        """Generate a report if the configured schedule requires it."""
        from tipster.reporter import generate_report

        now = datetime.now(timezone.utc)

        # Check if there are any unreported extracted items
        db = get_db()
        try:
            item_repo = ContentItemRepo(db)
            unreported = item_repo.list_unreported(self._topic_id)
            if not unreported:
                return

            # Check last report time from DB
            last_report = ReportRepo(db).get_last(self._topic_id)
            last_report_at = last_report.generated_at if last_report else None
        finally:
            db.close()

        # Determine if enough time has passed for the next report
        interval_secs = self._report_interval_seconds()
        if last_report_at is None:
            # No report ever generated — generate one now if we have items
            should_report = True
        else:
            # Make last_report_at timezone-aware if needed
            if last_report_at.tzinfo is None:
                from datetime import timezone as _tz
                last_report_at = last_report_at.replace(tzinfo=_tz.utc)
            elapsed = (now - last_report_at).total_seconds()
            should_report = elapsed >= interval_secs

        if should_report:
            log.info("Report schedule triggered (%d unreported items)", len(unreported))
            await generate_report(self._topic_id, self._cfg, self._bus)

    def _report_interval_seconds(self) -> float:
        """Parse schedule.report_interval into seconds."""
        ri = self._cfg.schedule.report_interval.strip().lower()
        if ri == "daily":
            return 86400.0
        if ri == "weekly":
            return 7 * 86400.0
        if ri == "hourly":
            return 3600.0
        # Fallback: treat as minutes if numeric, else daily
        try:
            return float(ri) * 60
        except ValueError:
            return 86400.0

    def stop(self) -> None:
        self._running = False
        self._stats.running = False
