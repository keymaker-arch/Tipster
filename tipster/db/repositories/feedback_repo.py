"""Feedback repository."""

from typing import Optional

from sqlalchemy.orm import Session

from tipster.db.models import Feedback


class FeedbackRepo:
    def __init__(self, db: Session) -> None:
        self._db = db

    def add(
        self,
        topic_id: int,
        item_id: Optional[int] = None,
        url_id: Optional[int] = None,
        judgement: Optional[int] = None,   # +1 / -1
        comment: Optional[str] = None,
        weight_delta: Optional[float] = None,
    ) -> Feedback:
        fb = Feedback(
            topic_id=topic_id,
            item_id=item_id,
            url_id=url_id,
            judgement=judgement,
            comment=comment,
            weight_delta=weight_delta,
        )
        self._db.add(fb)
        self._db.commit()
        self._db.refresh(fb)
        return fb
