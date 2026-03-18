"""Directive Store repository."""

from datetime import datetime, timezone
from typing import Optional

from sqlalchemy.orm import Session

from tipster.db.models import Directive


class DirectiveRepo:
    def __init__(self, db: Session) -> None:
        self._db = db

    def list_active(self, topic_id: int) -> list[Directive]:
        now = datetime.now(timezone.utc)
        return (
            self._db.query(Directive)
            .filter(
                Directive.topic_id == topic_id,
                Directive.applied == False,
                (Directive.expires_at.is_(None)) | (Directive.expires_at > now),
            )
            .all()
        )

    def add(
        self,
        topic_id: int,
        directive_type: str,
        target: Optional[str] = None,
        params_json: Optional[str] = None,
        expires_at: Optional[datetime] = None,
    ) -> Directive:
        d = Directive(
            topic_id=topic_id,
            directive_type=directive_type,
            target=target,
            params_json=params_json,
            expires_at=expires_at,
        )
        self._db.add(d)
        self._db.commit()
        self._db.refresh(d)
        return d

    def count_active(self, topic_id: int) -> int:
        return len(self.list_active(topic_id))

    def mark_applied(self, directive_id: int) -> None:
        d = self._db.query(Directive).filter(Directive.directive_id == directive_id).first()
        if d:
            d.applied = True
            self._db.commit()
