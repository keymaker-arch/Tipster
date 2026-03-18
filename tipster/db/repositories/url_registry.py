"""URL Registry repository."""

from datetime import datetime, timezone
from typing import Optional
from urllib.parse import urlparse

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
