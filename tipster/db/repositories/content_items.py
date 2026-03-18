"""Content Items repository."""

from typing import Optional

from sqlalchemy.orm import Session

from tipster.db.models import ContentItem


class ContentItemRepo:
    def __init__(self, db: Session) -> None:
        self._db = db

    def get_by_id(self, item_id: int) -> Optional[ContentItem]:
        return self._db.query(ContentItem).filter(ContentItem.item_id == item_id).first()

    def get_by_hash(self, content_hash: str) -> Optional[ContentItem]:
        return self._db.query(ContentItem).filter(ContentItem.content_hash == content_hash).first()

    def list_pending_extraction(self, topic_id: int) -> list[ContentItem]:
        return (
            self._db.query(ContentItem)
            .filter(
                ContentItem.topic_id == topic_id,
                ContentItem.status == "pending_extraction",
            )
            .all()
        )

    def list_unreported(self, topic_id: int) -> list[ContentItem]:
        return (
            self._db.query(ContentItem)
            .filter(
                ContentItem.topic_id == topic_id,
                ContentItem.status == "extracted",
                ContentItem.reported == False,
            )
            .all()
        )

    def add(
        self,
        topic_id: int,
        url_id: int,
        content_hash: str,
        raw_text: str,
        topic_score: Optional[float] = None,
        is_new_source: bool = False,
    ) -> ContentItem:
        item = ContentItem(
            topic_id=topic_id,
            url_id=url_id,
            content_hash=content_hash,
            raw_text=raw_text,
            topic_score=topic_score,
            status="pending_extraction",
            reported=False,
            is_new_source=is_new_source,
        )
        self._db.add(item)
        self._db.commit()
        self._db.refresh(item)
        return item

    def mark_extracted(
        self,
        item_id: int,
        extracted_json: str,
        article_sum_md: str,
    ) -> None:
        item = self.get_by_id(item_id)
        if item:
            item.extracted_json = extracted_json
            item.article_sum_md = article_sum_md
            item.status = "extracted"
            self._db.commit()

    def mark_duplicate(self, item_id: int, duplicate_of: int) -> None:
        item = self.get_by_id(item_id)
        if item:
            item.status = "failed"
            item.duplicate_of = duplicate_of
            self._db.commit()

    def mark_reported(self, item_ids: list[int]) -> None:
        self._db.query(ContentItem).filter(ContentItem.item_id.in_(item_ids)).update(
            {"reported": True}, synchronize_session="fetch"
        )
        self._db.commit()

    def count_by_topic(self, topic_id: int) -> int:
        return self._db.query(ContentItem).filter(ContentItem.topic_id == topic_id).count()

    def count_pending(self, topic_id: int) -> int:
        return (
            self._db.query(ContentItem)
            .filter(ContentItem.topic_id == topic_id, ContentItem.status == "pending_extraction")
            .count()
        )
