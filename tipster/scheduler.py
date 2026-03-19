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
        self.running: bool = False
        # Crawler worker pool
        self.active_workers: int = 0   # workers currently processing a URL
        self.queue_depth: int = 0      # URLs waiting in the queue
        # Extractor (single housekeeper call for now; 0 or 1)
        # TODO: update to a true worker count once extractor is refactored into
        # a persistent worker pool (see extractor.py: extract_pending TODO).
        self.active_extractor: int = 0
        # Session-level crawl progress (never reset)
        self.crawled_total: int = 0
        self.relevant_total: int = 0
        # Cumulative session cost/tokens (never reset; accumulated in housekeeper)
        self.session_cost_usd: float = 0.0
        self.session_tokens: int = 0


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
        log.debug(
            "LINK DISCOVERY DONE  source=%s  budget=[tokens=%d cost=$%.4f]",
            url, budget.tokens_used, budget.cost_usd,
        )


class CrawlScheduler:
    """Crawls URLs using a persistent worker pool fed by a queue.

    Architecture:
      - N persistent worker coroutines pull (url_id, url) from an asyncio.Queue
      - A DB poller periodically queries for due URLs and enqueues new ones
      - A housekeeper runs directives, link scoring, extraction, and reports
    """

    POLL_INTERVAL = 30  # seconds between DB polls for newly due URLs

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
        self._running = False
        # Queue and dedup set — populated in run() once the event loop is active.
        self._crawl_queue: asyncio.Queue[tuple[int, str]] = asyncio.Queue()
        self._queued_url_ids: set[int] = set()
        # LLM semaphore — created lazily in run().
        self._llm_sem: Optional[asyncio.Semaphore] = None
        self._budget = BudgetGate(
            max_tokens=cfg.budget.max_tokens_per_slice,
            max_cost_usd=cfg.budget.max_cost_per_slice_usd,
        )
        self._last_report_checked: Optional[datetime] = None

    async def run(self) -> None:
        self._running = True
        self._stats.running = True
        self._llm_sem = asyncio.Semaphore(self._cfg.crawl.max_llm_workers)
        log.info(
            "Scheduler started (max_crawl_workers=%d, max_llm_workers=%d)",
            self._cfg.crawl.max_crawl_workers,
            self._cfg.crawl.max_llm_workers,
        )

        workers = [
            asyncio.create_task(self._worker(i))
            for i in range(self._cfg.crawl.max_crawl_workers)
        ]
        poller = asyncio.create_task(self._db_poller())
        housekeeper = asyncio.create_task(self._housekeeper())

        await asyncio.gather(poller, housekeeper, *workers, return_exceptions=True)

    async def _worker(self, worker_id: int) -> None:
        """Persistent worker: pulls URLs from the queue and processes them one at a time."""
        log.debug("Worker %d started", worker_id)
        while self._running:
            try:
                url_id, url = await asyncio.wait_for(self._crawl_queue.get(), timeout=1.0)
            except asyncio.TimeoutError:
                continue

            # Remove from dedup set immediately so the next DB poll can re-enqueue
            # this URL if it stays due (e.g. budget was exhausted during triage).
            self._queued_url_ids.discard(url_id)
            self._stats.active_workers += 1
            self._stats.queue_depth = self._crawl_queue.qsize()

            try:
                await _process_url(
                    url_id=url_id,
                    url=url,
                    cfg=self._cfg,
                    topic_id=self._topic_id,
                    bus=self._bus,
                    stats=self._stats,
                    budget=self._budget,
                    llm_sem=self._llm_sem,  # type: ignore[arg-type]
                )
            except Exception:
                log.exception("Worker %d: unhandled error for url_id=%d %s", worker_id, url_id, url)
            finally:
                self._stats.active_workers -= 1
                self._stats.queue_depth = self._crawl_queue.qsize()
                self._crawl_queue.task_done()

        log.debug("Worker %d stopped", worker_id)

    async def _db_poller(self) -> None:
        """Periodically scans the DB for due URLs and pushes new ones onto the queue."""
        # Immediate first poll so crawling starts without waiting POLL_INTERVAL
        await self._enqueue_due_urls()

        while self._running:
            await asyncio.sleep(self.POLL_INTERVAL)
            await self._enqueue_due_urls()

    async def _enqueue_due_urls(self) -> None:
        """Query DB for due URLs and enqueue any that are not already in the queue."""
        now = datetime.now(timezone.utc)

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
            return

        newly_queued = 0
        for entry in due:
            if entry.url_id not in self._queued_url_ids:
                self._queued_url_ids.add(entry.url_id)
                await self._crawl_queue.put((entry.url_id, entry.url))
                newly_queued += 1
        self._stats.queue_depth = self._crawl_queue.qsize()

        if newly_queued:
            msg = (
                f"scheduler tick — queued {newly_queued} new URL(s) "
                f"(queue depth: {self._crawl_queue.qsize()}) [{self._budget.summary}]"
            )
        else:
            msg = f"scheduler tick — {len(due)} URL(s) due, all already queued"
        await self._bus.emit(Event(kind=EventKind.SCHEDULER_TICK, message=msg))

    async def _housekeeper(self) -> None:
        """Periodically runs directives, link scoring, extraction, and report generation."""
        from tipster.extractor import extract_pending
        from tipster.link_scorer import score_pending_links
        from tipster.directives_consumer import apply_directives

        while self._running:
            # Accumulate this cycle's LLM usage into session totals before reset
            self._stats.session_cost_usd += self._budget.cost_usd
            self._stats.session_tokens += self._budget.tokens_used

            # Reset budget for the new cycle
            self._budget.reset()
            log.debug("HOUSEKEEPING START  budget reset  max_tokens=%d  max_cost=$%.2f",
                      self._cfg.budget.max_tokens_per_slice,
                      self._cfg.budget.max_cost_per_slice_usd)

            await apply_directives(self._topic_id, self._cfg, self._bus)
            await score_pending_links(self._topic_id, self._cfg, self._budget, self._bus)

            self._stats.active_extractor = 1
            await extract_pending(self._cfg, self._topic_id, self._budget, self._bus)
            self._stats.active_extractor = 0

            await self._maybe_generate_report()

            await asyncio.sleep(self.POLL_INTERVAL)

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
