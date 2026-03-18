"""Prompt Examples repository (few-shot examples for LLM prompts)."""

from typing import Optional

from sqlalchemy.orm import Session

from tipster.db.models import PromptExample

_MAX_EXAMPLES = 20


class PromptExampleRepo:
    def __init__(self, db: Session) -> None:
        self._db = db

    def add(
        self,
        topic_id: int,
        content_snippet: str,
        judgement: Optional[int] = None,
        label: Optional[str] = None,
        domain: Optional[str] = None,
    ) -> PromptExample:
        ex = PromptExample(
            topic_id=topic_id,
            domain=domain,
            content_snippet=content_snippet[:1000],  # cap snippet size
            judgement=judgement,
            label=label,
        )
        self._db.add(ex)
        self._db.commit()
        self._db.refresh(ex)
        return ex

    def list_for_prompt(
        self,
        topic_id: int,
        domain: Optional[str] = None,
        max_count: int = _MAX_EXAMPLES,
    ) -> list[PromptExample]:
        """Return up to max_count examples, domain-filtered and recency-first."""
        q = self._db.query(PromptExample).filter(PromptExample.topic_id == topic_id)

        # Domain-matching examples first
        if domain:
            domain_rows = (
                q.filter(PromptExample.domain == domain)
                .order_by(PromptExample.created_at.desc())
                .limit(max_count)
                .all()
            )
            remaining = max_count - len(domain_rows)
            if remaining > 0:
                seen_ids = {r.example_id for r in domain_rows}
                generic_rows = (
                    q.filter(PromptExample.example_id.notin_(seen_ids))
                    .order_by(PromptExample.created_at.desc())
                    .limit(remaining)
                    .all()
                )
                return domain_rows + generic_rows
            return domain_rows

        return (
            q.order_by(PromptExample.created_at.desc())
            .limit(max_count)
            .all()
        )
