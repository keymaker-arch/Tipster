"""Fact Extractor — Phase 2.

Runs extraction_model LLM on pending content items, producing:
- extracted_json: structured facts as a JSON blob
- article_sum_md: a concise Markdown summary of the article

The LLM first classifies the page type (article | list | other) and then
applies the appropriate extraction strategy:
  - article: 5–10 sentence Markdown summary + key facts + entities
  - list:    structured item-by-item extraction (trending repos, news feeds, etc.)
  - other:   brief 2–3 sentence description

Updates content_items.status from pending_extraction → extracted (or failed).
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from dataclasses import dataclass
from functools import partial
from typing import TYPE_CHECKING, Any

from tipster import llm as llm_module
from tipster.budget import BudgetGate
from tipster.config import TipsterConfig
from tipster.db.repositories.content_items import ContentItemRepo
from tipster.db.session import get_db
from tipster.events import Event, EventBus, EventKind

if TYPE_CHECKING:
    pass

log = logging.getLogger("tipster.extractor")


_EXTRACT_SYSTEM = """\
You are a web content extractor for a web intelligence crawler.

STEP 1 — Classify the page type:
- "article": a blog post, essay, news article, documentation page, or research paper
  (primarily prose focused on a single topic or story)
- "list": a ranking or index page containing multiple discrete items — trending
  repositories, news feeds, search results, product listings, release notes, changelogs, etc.
- "other": homepages, profile pages, login pages, error pages, or anything that
  doesn't fit the above

STEP 2 — Extract based on the page type:

For "article":
  Write a 5-10 sentence Markdown summary capturing the main thesis, key claims, and
  conclusions. Include notable facts and named entities.

For "list":
  Extract every distinct item on the page as a structured object. Do NOT truncate —
  capture all items. Include all available metadata per item (name, description, URL,
  stars, author, language, date, score, rank, etc.).

For "other":
  Write a 2-3 sentence description of the page's purpose or content.

If an extraction focus hint is provided, prioritise that aspect in your extraction.

Return ONLY valid JSON (no markdown fences, no prose).

Article format:
{
  "page_type": "article",
  "title": "<title or inferred heading>",
  "summary": "<5-10 sentences in Markdown>",
  "key_facts": ["<fact>", ...],
  "entities": ["<person/org/project>", ...]
}

List format:
{
  "page_type": "list",
  "title": "<list title>",
  "summary": "<1-2 sentences describing what this list is>",
  "items": [
    {"name": "...", "description": "...", "url": "...", <any other available metadata>},
    ...
  ]
}

Other format:
{
  "page_type": "other",
  "title": "<title>",
  "summary": "<2-3 sentences>"
}
"""


def _build_extract_prompt(cfg: TipsterConfig, text: str, prompt_snippet: str = "") -> str:
    hint = f"\nExtraction focus: {prompt_snippet}" if prompt_snippet else ""
    return (
        f"Topic: {cfg.topic.name}\n"
        f"Description: {cfg.topic.description}{hint}\n\n"
        f"--- Page content ---\n{text or '(empty)'}"
    )


def extract_one(
    item_id: int,
    raw_text: str,
    cfg: TipsterConfig,
    prompt_snippet: str = "",
) -> tuple[bool, str, int, float]:
    """Extract facts from one content item.

    Returns (success, message_or_payload, tokens_used, cost_usd).
    Budget recording and DB updates are the caller's responsibility.
    """
    prompt = _build_extract_prompt(cfg, raw_text or "", prompt_snippet)

    try:
        raw, tokens, cost = llm_module.complete_with_usage(
            model=cfg.llm.extraction_model,
            messages=[
                {"role": "system", "content": _EXTRACT_SYSTEM},
                {"role": "user", "content": prompt},
            ],
            max_tokens=4096,
            temperature=0.2,
            api_base=cfg.llm.api_base,
        )
    except Exception as exc:
        return False, f"LLM error: {exc}", 0, 0.0

    # Strip fences
    raw = raw.strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```[a-z]*\n?", "", raw)
        raw = re.sub(r"\n?```$", "", raw)

    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return False, f"JSON parse error: {raw[:80]}", tokens, cost

    extracted_json = json.dumps(parsed)
    # summary is present for all page types; for lists it describes the list itself
    article_sum_md = parsed.get("summary", "")
    return True, json.dumps({"extracted_json": extracted_json, "article_sum_md": article_sum_md}), tokens, cost


# ---------------------------------------------------------------------------
# Extraction worker pool
# ---------------------------------------------------------------------------

@dataclass
class ExtractTask:
    """A unit of extraction work enqueued from a crawler worker or the catch-up poller."""
    item_id: int
    url_id: int
    raw_text: str
    url: str
    domain: str
    topic_score: float
    is_new_source: bool
    prompt_snippet: str


class ExtractionWorkerPool:
    """Persistent worker pool that drains extraction tasks as soon as they arrive.

    Workers are started alongside the crawler workers and run for the lifetime of
    the scheduler.  The crawler enqueues a task immediately after saving a new
    content item; a catch-up DB poller handles items that existed before startup or
    were missed during budget exhaustion.

    Budget interaction
    ------------------
    Workers check ``budget.can_proceed()`` before each LLM call.  When the budget is
    exhausted the worker holds its item and polls every 5 s until the housekeeper
    resets the budget.  ``budget.record()`` is always called from the asyncio event
    loop (after ``run_in_executor`` returns), so no threading lock is needed — asyncio
    is single-threaded between await points.
    """

    def __init__(
        self,
        cfg: TipsterConfig,
        topic_id: int,
        budget: BudgetGate,
        bus: EventBus,
        stats: Any,          # CrawlStats — typed as Any to avoid circular import
        max_workers: int,
    ) -> None:
        self._cfg = cfg
        self._topic_id = topic_id
        self._budget = budget
        self._bus = bus
        self._stats = stats
        self._max_workers = max_workers
        self._queue: asyncio.Queue[ExtractTask] = asyncio.Queue()
        self._queued_ids: set[int] = set()
        self._running = False

    def enqueue(self, task: ExtractTask) -> None:
        """Enqueue an extraction task.  Silently deduplicates by item_id."""
        if task.item_id not in self._queued_ids:
            self._queued_ids.add(task.item_id)
            self._queue.put_nowait(task)

    async def _worker(self, worker_id: int) -> None:
        log.debug("Extractor worker %d started", worker_id)
        loop = asyncio.get_running_loop()

        while self._running:
            # Wait for a task, checking _running every second so we can exit cleanly.
            try:
                task = await asyncio.wait_for(self._queue.get(), timeout=1.0)
            except asyncio.TimeoutError:
                continue

            self._queued_ids.discard(task.item_id)

            # If the budget is exhausted, hold the item and wait for the housekeeper
            # to reset it.  The item remains pending_extraction in the DB so a future
            # restart will recover it via the catch-up poller.
            while not self._budget.can_proceed():
                if not self._running:
                    self._queue.task_done()
                    return
                log.debug(
                    "Extractor worker %d: budget exhausted, waiting for reset (item_id=%d)",
                    worker_id, task.item_id,
                )
                await asyncio.sleep(5)

            self._stats.active_extractor += 1
            log.debug(
                "EXTRACT START  worker=%d  item_id=%d  url=%s  text_len=%d  hint=%r",
                worker_id, task.item_id, task.url, len(task.raw_text or ""), task.prompt_snippet,
            )

            try:
                success, payload, tokens, cost = await loop.run_in_executor(
                    None,
                    partial(extract_one, task.item_id, task.raw_text or "", self._cfg, task.prompt_snippet),
                )
                # Record in the event loop — no lock needed; asyncio is cooperative.
                if tokens or cost:
                    self._budget.record(tokens, cost)
            except Exception as exc:
                success, payload, tokens, cost = False, f"unhandled: {exc}", 0, 0.0
            finally:
                self._stats.active_extractor -= 1
                self._queue.task_done()

            await self._persist_result(task, success, payload)

        log.debug("Extractor worker %d stopped", worker_id)

    async def _persist_result(self, task: ExtractTask, success: bool, payload: str) -> None:
        """Write extraction result to DB and emit an event."""
        db = get_db()
        try:
            content_repo = ContentItemRepo(db)
            if success:
                data = json.loads(payload)
                parsed = json.loads(data["extracted_json"])
                content_repo.mark_extracted(
                    item_id=task.item_id,
                    extracted_json=data["extracted_json"],
                    article_sum_md=data["article_sum_md"],
                )
                log.debug(
                    "EXTRACT OK  item_id=%d  url=%s  page_type=%s  title=%r  "
                    "facts=%d  entities=%d  summary_len=%d",
                    task.item_id, task.url,
                    parsed.get("page_type", "?"),
                    parsed.get("title", "")[:80],
                    len(parsed.get("key_facts", [])),
                    len(parsed.get("entities", [])),
                    len(data.get("article_sum_md", "")),
                )
                await self._bus.emit(
                    Event(
                        kind=EventKind.EXTRACT_OK,
                        url=task.url,
                        message=f"extracted item_id={task.item_id}",
                        data={
                            "item_id": task.item_id,
                            "url_id": task.url_id,
                            "url": task.url,
                            "domain": task.domain,
                            "score": task.topic_score,
                            "is_new_source": task.is_new_source,
                            "page_type": parsed.get("page_type", "article"),
                            "title": parsed.get("title", ""),
                            "summary": parsed.get("summary", ""),
                            "key_facts": parsed.get("key_facts", []),
                            "entities": parsed.get("entities", []),
                            "items": parsed.get("items", []),
                        },
                    )
                )
            else:
                log.debug(
                    "EXTRACT FAILED  item_id=%d  url=%s  reason=%s",
                    task.item_id, task.url, payload[:120],
                )
                from tipster.db.models import ContentItem as _CI
                db.query(_CI).filter_by(item_id=task.item_id).update({"status": "failed"})
                db.commit()
                await self._bus.emit(
                    Event(
                        kind=EventKind.EXTRACT_ERROR,
                        url=task.url,
                        message=payload[:80],
                    )
                )
        finally:
            db.close()

    async def _scan_pending(self) -> None:
        from tipster.db.models import ContentItem as _CI, UrlRegistry as _UR
        db = get_db()
        try:
            rows = (
                db.query(
                    _CI.item_id, _CI.url_id, _CI.raw_text, _CI.topic_score,
                    _CI.is_new_source, _UR.url, _UR.domain, _UR.prompt_snippet,
                )
                .join(_UR, _CI.url_id == _UR.url_id)
                .filter(_CI.topic_id == self._topic_id, _CI.status == "pending_extraction")
                .all()
            )
        finally:
            db.close()

        for r in rows:
            self.enqueue(ExtractTask(
                item_id=r.item_id,
                url_id=r.url_id,
                raw_text=r.raw_text or "",
                url=r.url,
                domain=r.domain or "",
                topic_score=r.topic_score or 0.0,
                is_new_source=bool(r.is_new_source),
                prompt_snippet=r.prompt_snippet or "",
            ))

    async def run(self) -> None:
        """Start all worker coroutines.  Runs until stop() is called.

        A one-time DB scan at startup recovers any items left in pending_extraction
        from before this process started (e.g. after a crash or restart).  During
        normal operation the crawler enqueues tasks directly via enqueue(), so no
        recurring DB poll is needed.
        """
        self._running = True
        log.info(
            "Extraction worker pool started (max_extractor_workers=%d)",
            self._max_workers,
        )
        await self._scan_pending()
        workers = [
            asyncio.create_task(self._worker(i))
            for i in range(self._max_workers)
        ]
        await asyncio.gather(*workers, return_exceptions=True)

    def stop(self) -> None:
        self._running = False
