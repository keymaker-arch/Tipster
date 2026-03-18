"""Internal event bus — asyncio.Queue-based, in-process for Phase 1.

Workers emit Event objects; the TUI Activity Log consumes them.
Every emitted event is also forwarded to the Python logging system at DEBUG
level (logger ``tipster.events``), so attaching a file handler to the
``tipster`` logger is sufficient to get a complete, detailed activity log.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Optional

_log = logging.getLogger("tipster.events")


def _fmt(event: "Event") -> str:
    """Format an event as a single human-readable debug line."""
    parts: list[str] = [f"[{event.kind.value}]"]
    if event.url:
        parts.append(event.url)
    if event.score is not None:
        parts.append(f"score={event.score:.4f}")
    if event.message:
        parts.append(event.message)
    return "  ".join(parts)


class EventKind(str, Enum):
    CRAWL_START = "crawl_start"
    CRAWL_OK = "crawl_ok"
    CRAWL_SKIP = "crawl_skip"          # unchanged content
    CRAWL_DUPLICATE = "crawl_duplicate"  # cross-URL duplicate
    CRAWL_ERROR = "crawl_error"
    TRIAGE_RELEVANT = "triage_relevant"
    TRIAGE_IRRELEVANT = "triage_irrelevant"
    EXTRACT_START = "extract_start"
    EXTRACT_OK = "extract_ok"
    EXTRACT_DEFERRED = "extract_deferred"
    EXTRACT_ERROR = "extract_error"
    LINK_DISCOVERED = "link_discovered"
    LINK_DEFERRED = "link_deferred"
    SCHEDULER_TICK = "scheduler_tick"
    STATS_UPDATE = "stats_update"
    REPORT_READY = "report_ready"        # a new report has been generated
    DIRECTIVE_APPLIED = "directive_applied"  # a directive was applied


@dataclass
class Event:
    kind: EventKind
    url: str = ""
    message: str = ""
    score: Optional[float] = None
    data: Optional[dict] = None          # arbitrary payload (e.g. report content)
    ts: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class EventBus:
    """Simple wrapper around asyncio.Queue with broadcast support."""

    def __init__(self, maxsize: int = 500) -> None:
        self._q: asyncio.Queue[Event] = asyncio.Queue(maxsize=maxsize)

    async def emit(self, event: Event) -> None:
        _log.debug("%s", _fmt(event))
        try:
            self._q.put_nowait(event)
        except asyncio.QueueFull:
            # Drop oldest to make room
            try:
                self._q.get_nowait()
            except asyncio.QueueEmpty:
                pass
            self._q.put_nowait(event)

    async def receive(self) -> Event:
        return await self._q.get()

    def emit_nowait(self, event: Event) -> None:
        _log.debug("%s", _fmt(event))
        try:
            self._q.put_nowait(event)
        except asyncio.QueueFull:
            try:
                self._q.get_nowait()
            except asyncio.QueueEmpty:
                pass
            try:
                self._q.put_nowait(event)
            except asyncio.QueueFull:
                pass
