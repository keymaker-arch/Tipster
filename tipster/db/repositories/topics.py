"""Topics repository."""

from typing import Optional

from sqlalchemy.orm import Session

from tipster.db.models import Topic


class TopicRepo:
    def __init__(self, db: Session) -> None:
        self._db = db

    def get_active(self) -> Optional[Topic]:
        """Return the first active topic (MVP: single-topic)."""
        return self._db.query(Topic).filter(Topic.is_active == True).first()

    def get_by_id(self, topic_id: int) -> Optional[Topic]:
        return self._db.query(Topic).filter(Topic.topic_id == topic_id).first()

    def get_by_name(self, name: str) -> Optional[Topic]:
        return self._db.query(Topic).filter(Topic.name == name).first()

    def create(self, name: str, description: Optional[str] = None) -> Topic:
        topic = Topic(name=name, description=description, is_active=True)
        self._db.add(topic)
        self._db.commit()
        self._db.refresh(topic)
        return topic

    def list_all(self) -> list[Topic]:
        return self._db.query(Topic).all()
