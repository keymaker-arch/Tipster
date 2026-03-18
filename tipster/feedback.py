"""Feedback Processor & Comment Interpreter — Phase 4.

Handles per-item user feedback:
  - Judgement (+1/-1) → updates source_weight in url_registry, appends to prompt_examples
  - Free-text comment → LLM interprets → emits directives to directive_store
"""

from __future__ import annotations

import json
import logging
from typing import Optional

log = logging.getLogger("tipster.feedback")

_WEIGHT_DELTA = 0.05  # source_weight nudge per judgement

_COMMENT_SYSTEM = """\
You are a directive extractor for a web intelligence crawler called Tipster.
The user has just read an intelligence report and left a free-text comment about an article.
Parse the comment and emit one or more structured directives in JSON.

Available directive types:
- BOOST_CRAWL_PRIORITY: increase crawl frequency for a domain/URL
  {"type": "BOOST_CRAWL_PRIORITY", "target": "<domain or url>", "magnitude": 0.5, "duration_days": 7}
- UPDATE_LINK_SCORE_HINT: add a hint to the link scorer for a domain
  {"type": "UPDATE_LINK_SCORE_HINT", "target": "<domain>", "hint": "<hint text>", "polarity": "positive|negative"}
- EXPAND_TOPIC: add a subtopic to track
  {"type": "EXPAND_TOPIC", "target": "<subtopic description>"}
- BLACKLIST_SOURCE: block a domain or URL from future crawls
  {"type": "BLACKLIST_SOURCE", "target": "<domain or url>"}
- SCHEDULE_DEEP_DIVE: pin a URL to a specific crawl frequency
  {"type": "SCHEDULE_DEEP_DIVE", "target": "<url>", "interval_hours": 24}

Return ONLY a valid JSON array of directive objects. If no actionable directive can be inferred,
return an empty array [].
Do NOT add markdown fences or prose.
"""


def process_judgement(
    topic_id: int,
    item_id: int,
    url_id: int,
    judgement: int,         # +1 = interesting, -1 = not interesting
    content_snippet: str,
    domain: str,
) -> None:
    """Record a judgement: update source_weight and add to prompt_examples."""
    from tipster.db.session import get_db
    from tipster.db.repositories.feedback_repo import FeedbackRepo
    from tipster.db.repositories.prompt_examples import PromptExampleRepo
    from tipster.db.models import UrlRegistry

    weight_delta = _WEIGHT_DELTA * judgement

    db = get_db()
    try:
        # Record feedback
        FeedbackRepo(db).add(
            topic_id=topic_id,
            item_id=item_id,
            url_id=url_id,
            judgement=judgement,
            weight_delta=weight_delta,
        )

        # Update source_weight in url_registry (clamp 0..1)
        url_entry = db.query(UrlRegistry).filter_by(url_id=url_id).first()
        if url_entry:
            url_entry.source_weight = max(0.0, min(1.0, url_entry.source_weight + weight_delta))
            db.commit()

        # Append to prompt_examples for future few-shot prompts
        label = "interesting" if judgement > 0 else "not_interesting"
        PromptExampleRepo(db).add(
            topic_id=topic_id,
            content_snippet=content_snippet,
            judgement=judgement,
            label=label,
            domain=domain,
        )
    finally:
        db.close()

    log.info(
        "Judgement recorded: item_id=%d url_id=%d judgement=%+d weight_delta=%+.3f",
        item_id, url_id, judgement, weight_delta,
    )


async def process_comment(
    topic_id: int,
    item_id: int,
    url_id: int,
    comment: str,
    cfg,   # TipsterConfig
    bus,   # EventBus
) -> list[str]:
    """Parse a free-text comment via LLM → emit directives.

    Returns list of directive_type strings created.
    """
    import asyncio
    from functools import partial
    from tipster import llm as llm_module
    from tipster.db.session import get_db
    from tipster.db.repositories.feedback_repo import FeedbackRepo
    from tipster.db.repositories.directives import DirectiveRepo
    from tipster.events import Event, EventKind

    # Record raw comment in feedback table
    db = get_db()
    try:
        FeedbackRepo(db).add(
            topic_id=topic_id,
            item_id=item_id,
            url_id=url_id,
            comment=comment,
        )
    finally:
        db.close()

    # LLM interpretation
    loop = asyncio.get_running_loop()
    try:
        raw = await loop.run_in_executor(
            None,
            partial(
                llm_module.complete,
                cfg.llm.comment_model,
                [
                    {"role": "system", "content": _COMMENT_SYSTEM},
                    {"role": "user", "content": f"Article URL: (url_id={url_id})\nComment: {comment}"},
                ],
                max_tokens=512,
                temperature=0.1,
                api_base=cfg.llm.api_base,
            ),
        )
    except Exception as exc:
        log.warning("Comment interpreter LLM error: %s", exc)
        return []

    raw = raw.strip()
    if raw.startswith("```"):
        import re
        raw = re.sub(r"^```[a-z]*\n?", "", raw)
        raw = re.sub(r"\n?```$", "", raw)

    try:
        directives_data = json.loads(raw)
        if not isinstance(directives_data, list):
            directives_data = []
    except (json.JSONDecodeError, ValueError):
        log.warning("Comment interpreter parse error — raw: %s", raw[:120])
        return []

    created_types: list[str] = []

    db = get_db()
    try:
        dir_repo = DirectiveRepo(db)
        for d in directives_data:
            dtype = d.get("type", "")
            if not dtype:
                continue
            target = d.get("target", "")
            params = {k: v for k, v in d.items() if k not in ("type", "target")}
            dir_repo.add(
                topic_id=topic_id,
                directive_type=dtype,
                target=target,
                params_json=json.dumps(params) if params else None,
            )
            created_types.append(dtype)
            await bus.emit(
                Event(
                    kind=EventKind.DIRECTIVE_APPLIED,
                    message=f"directive {dtype} → {target or '(general)'}",
                )
            )
    finally:
        db.close()

    if created_types:
        log.info("Comment → %d directive(s): %s", len(created_types), created_types)

    return created_types
