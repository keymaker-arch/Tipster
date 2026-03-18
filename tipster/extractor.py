"""Fact Extractor — Phase 2.

Runs extraction_model LLM on pending content items, producing:
- extracted_json: structured facts as a JSON blob
- article_sum_md: a concise Markdown summary of the article

Updates content_items.status from pending_extraction → extracted (or failed).
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from functools import partial
from typing import Optional

from tipster import llm as llm_module
from tipster.budget import BudgetGate
from tipster.config import TipsterConfig
from tipster.db.repositories.content_items import ContentItemRepo
from tipster.db.session import get_db
from tipster.events import Event, EventBus, EventKind

log = logging.getLogger("tipster.extractor")

_EXTRACT_SYSTEM = """\
You are a fact extractor for a web intelligence crawler.
Given a topic and an article text, extract the key information and summarise it.

Return ONLY valid JSON (no markdown fences, no prose):
{
  "title": "<article title or inferred heading>",
  "summary": "<2-4 sentence Markdown summary of what this article says>",
  "key_facts": ["<fact1>", "<fact2>", ...],
  "entities": ["<person/org/product name>", ...],
  "relevance_notes": "<one sentence on why this is relevant to the topic>"
}

Rules:
- summary: write in Markdown, use plain sentences, no bullet lists
- key_facts: 3-7 specific, verifiable claims from the article
- entities: named people, organisations, products, projects mentioned
- Keep all fields concise
"""


def _build_extract_prompt(cfg: TipsterConfig, text: str) -> str:
    excerpt = text[:4000] if text else "(empty)"
    return (
        f"Topic: {cfg.topic.name}\n"
        f"Description: {cfg.topic.description}\n\n"
        f"--- Article text ---\n{excerpt}"
    )


def extract_one(
    item_id: int,
    raw_text: str,
    cfg: TipsterConfig,
    budget: Optional[BudgetGate] = None,
) -> tuple[bool, str]:
    """Extract facts from one content item.

    Returns (success: bool, message: str).
    Caller is responsible for DB updates.
    """
    if budget is not None and not budget.can_proceed():
        return False, "budget exhausted — deferred to next slice"

    prompt = _build_extract_prompt(cfg, raw_text or "")

    try:
        raw, tokens, cost = llm_module.complete_with_usage(
            model=cfg.llm.extraction_model,
            messages=[
                {"role": "system", "content": _EXTRACT_SYSTEM},
                {"role": "user", "content": prompt},
            ],
            max_tokens=512,
            temperature=0.2,
            api_base=cfg.llm.api_base,
        )
        if budget is not None:
            budget.record(tokens, cost)
    except Exception as exc:
        return False, f"LLM error: {exc}"

    # Strip fences
    raw = raw.strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```[a-z]*\n?", "", raw)
        raw = re.sub(r"\n?```$", "", raw)

    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return False, f"JSON parse error: {raw[:80]}"

    extracted_json = json.dumps(parsed)
    article_sum_md = parsed.get("summary", "")
    return True, json.dumps({"extracted_json": extracted_json, "article_sum_md": article_sum_md})


async def extract_pending(
    cfg: TipsterConfig,
    topic_id: int,
    budget: BudgetGate,
    bus: EventBus,
) -> int:
    """Process all content_items with status=pending_extraction.

    Runs at the start of each crawl slice, before new crawl jobs.
    Returns the number of items successfully extracted.
    """
    db = get_db()
    try:
        from tipster.db.models import ContentItem as _CI, UrlRegistry as _UR
        rows = (
            db.query(_CI.item_id, _CI.raw_text, _UR.url)
            .join(_UR, _CI.url_id == _UR.url_id)
            .filter(_CI.topic_id == topic_id, _CI.status == "pending_extraction")
            .all()
        )
    finally:
        db.close()

    if not rows:
        return 0

    # rows is a list of (item_id, raw_text, url) named tuples — all primitives
    pending_data = [(r.item_id, r.raw_text, r.url) for r in rows]

    await bus.emit(
        Event(
            kind=EventKind.EXTRACT_START,
            message=f"extracting {len(pending_data)} pending item(s)…",
        )
    )

    extracted_count = 0
    loop = asyncio.get_running_loop()

    for item_id, raw_text, item_url in pending_data:
        if not budget.can_proceed():
            remaining = len(pending_data) - extracted_count
            await bus.emit(
                Event(
                    kind=EventKind.EXTRACT_DEFERRED,
                    message=f"budget exhausted — {remaining} item(s) deferred to next slice",
                )
            )
            break

        success, payload = await loop.run_in_executor(
            None,
            partial(extract_one, item_id, raw_text or "", cfg, budget),
        )

        db = get_db()
        try:
            content_repo = ContentItemRepo(db)
            if success:
                data = json.loads(payload)
                content_repo.mark_extracted(
                    item_id=item_id,
                    extracted_json=data["extracted_json"],
                    article_sum_md=data["article_sum_md"],
                )
                extracted_count += 1
                await bus.emit(
                    Event(
                        kind=EventKind.EXTRACT_OK,
                        url=item_url,
                        message=f"extracted item_id={item_id}",
                    )
                )
            else:
                if "budget exhausted" in payload:
                    # Leave as pending_extraction for next slice
                    pass
                else:
                    from tipster.db.models import ContentItem as _CI
                    db.query(_CI).filter_by(item_id=item_id).update({"status": "failed"})
                    db.commit()
                    await bus.emit(
                        Event(
                            kind=EventKind.EXTRACT_ERROR,
                            url=item_url,
                            message=payload[:80],
                        )
                    )
        finally:
            db.close()

    return extracted_count
