"""URL Registry repository."""

from datetime import datetime, timezone
from typing import Optional
from urllib.parse import urlparse

from sqlalchemy import func
from sqlalchemy.orm import Session

from tipster.db.models import UrlRegistry


class UrlRegistryRepo:
    def __init__(self, db: Session) -> None:
        self._db = db

    def get_by_url(self, url: str) -> Optional[UrlRegistry]:
        return self._db.query(UrlRegistry).filter(UrlRegistry.url == url).first()

    def get_by_id(self, url_id: int) -> Optional[UrlRegistry]:
        return self._db.query(UrlRegistry).filter(UrlRegistry.url_id == url_id).first()

    def list_by_topic(self, topic_id: int) -> list[UrlRegistry]:
        return self._db.query(UrlRegistry).filter(UrlRegistry.topic_id == topic_id).all()

    def list_due(self, topic_id: int, now: Optional[datetime] = None) -> list[UrlRegistry]:
        """Return URLs that are due for crawling."""
        if now is None:
            now = datetime.now(timezone.utc)
        return (
            self._db.query(UrlRegistry)
            .filter(
                UrlRegistry.topic_id == topic_id,
                UrlRegistry.status.in_(["pending", "active"]),
                (UrlRegistry.next_check_at <= now) | (UrlRegistry.next_check_at.is_(None)),
            )
            .all()
        )

    def list_pending_score(self, topic_id: int) -> list[UrlRegistry]:
        """Return URLs waiting to be link-scored (budget was exhausted last slice)."""
        return (
            self._db.query(UrlRegistry)
            .filter(
                UrlRegistry.topic_id == topic_id,
                UrlRegistry.status == "pending_score",
            )
            .all()
        )

    def add(
        self,
        topic_id: int,
        url: str,
        added_by: str = "manual",
        relevance_score: float = 0.5,
        source_weight: float = 0.5,
        status: str = "pending",
        check_interval: int = 3600,
        recrawl_type: str = "periodic",
        prompt_snippet: str = "",
    ) -> UrlRegistry:
        domain = urlparse(url).netloc
        existing = self.get_by_url(url)
        if existing:
            return existing
        entry = UrlRegistry(
            topic_id=topic_id,
            url=url,
            domain=domain,
            relevance_score=relevance_score,
            source_weight=source_weight,
            added_by=added_by,
            status=status,
            check_interval=check_interval,
            recrawl_type=recrawl_type,
            prompt_snippet=prompt_snippet or None,
        )
        self._db.add(entry)
        self._db.commit()
        self._db.refresh(entry)
        return entry

    def update_after_crawl(
        self,
        url_id: int,
        last_checked: datetime,
        next_check_at: datetime,
        check_interval: int,
        status: str,
        is_new_source: bool = False,
    ) -> None:
        entry = self.get_by_id(url_id)
        if entry:
            entry.last_checked = last_checked
            entry.next_check_at = next_check_at
            entry.check_interval = check_interval
            entry.status = status
            if is_new_source:
                entry.is_new_source = True
            self._db.commit()

    def count_by_topic(self, topic_id: int) -> int:
        return self._db.query(UrlRegistry).filter(UrlRegistry.topic_id == topic_id).count()

    def seconds_until_next_due(
        self,
        topic_id: int,
        already_queued: frozenset[int] = frozenset(),
    ) -> Optional[float]:
        """Return seconds until the next unqueued URL becomes due for crawling.

        Returns 0.0  if any immediately-due URL (next_check_at IS NULL or past)
                     is not yet in already_queued — the poller should enqueue now.
        Returns N    if the next due event is N seconds in the future.
        Returns None if no crawlable URLs exist at all (caller should sleep for
                     a sensible cap and try again).

        already_queued must be the set of url_ids already in the crawl queue so
        that newly discovered URLs (next_check_at=NULL) are not confused with ones
        that have already been dispatched, which would otherwise cause a spin loop.
        """
        now = datetime.now(timezone.utc)

        # Check for unqueued URLs that are already due (includes next_check_at=NULL,
        # which is how newly discovered URLs arrive in the registry).
        immediate_q = (
            self._db.query(UrlRegistry.url_id)
            .filter(
                UrlRegistry.topic_id == topic_id,
                UrlRegistry.status.in_(["pending", "active"]),
                (UrlRegistry.next_check_at.is_(None)) | (UrlRegistry.next_check_at <= now),
            )
        )
        if already_queued:
            immediate_q = immediate_q.filter(UrlRegistry.url_id.notin_(already_queued))
        if immediate_q.first() is not None:
            return 0.0

        # No immediately-due URLs — find when the next scheduled one arrives.
        next_at = (
            self._db.query(func.min(UrlRegistry.next_check_at))
            .filter(
                UrlRegistry.topic_id == topic_id,
                UrlRegistry.status.in_(["pending", "active"]),
                UrlRegistry.next_check_at > now,
            )
            .scalar()
        )
        if next_at is None:
            return None
        if next_at.tzinfo is None:
            next_at = next_at.replace(tzinfo=timezone.utc)
        return max(0.0, (next_at - now).total_seconds())
