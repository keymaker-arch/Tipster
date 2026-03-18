"""Reports repository."""

from typing import Optional

from sqlalchemy.orm import Session

from tipster.db.models import Report


class ReportRepo:
    def __init__(self, db: Session) -> None:
        self._db = db

    def add(self, topic_id: int, narrative_md: str, report_json: str) -> Report:
        report = Report(
            topic_id=topic_id,
            narrative_md=narrative_md,
            report_json=report_json,
        )
        self._db.add(report)
        self._db.commit()
        self._db.refresh(report)
        return report

    def get_last(self, topic_id: int) -> Optional[Report]:
        return (
            self._db.query(Report)
            .filter(Report.topic_id == topic_id)
            .order_by(Report.generated_at.desc())
            .first()
        )

    def count_by_topic(self, topic_id: int) -> int:
        return self._db.query(Report).filter(Report.topic_id == topic_id).count()
