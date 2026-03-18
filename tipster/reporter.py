"""Reporter Module — Phase 4.

Reads all unreported, extracted content items and synthesises a Markdown
digest + structured JSON report.  Saves to the reports table and marks
items reported=True.
"""

from __future__ import annotations

import json
import logging
from typing import Optional

from tipster.config import TipsterConfig
from tipster.db.models import ContentItem, UrlRegistry
from tipster.db.repositories.content_items import ContentItemRepo
from tipster.db.repositories.reports import ReportRepo
from tipster.db.session import get_db
from tipster.events import Event, EventBus, EventKind

log = logging.getLogger("tipster.reporter")

_REPORT_SYSTEM = """\
You are an intelligence analyst synthesising a research digest for a human expert.
You will receive a list of recently extracted web articles on a given topic.
For each article you have: a summary, the source URL, and a relevance score.

Produce a Markdown digest with:
1. A concise **headline** (one sentence, the most important development).
2. A **Findings** section with bullet points — one per article, formatted as:
   - [Source domain] Brief finding (max 2 sentences).
   Flag articles from new sources with a ★ prefix.
3. A short **Summary** paragraph at the end tying the findings together.

Be factual, concise, and attribute every claim to its source.
Do NOT hallucinate facts not present in the provided summaries.
"""


def _build_report_prompt(cfg: TipsterConfig, items: list[dict]) -> str:
    lines = [
        f"Topic: {cfg.topic.name}",
        f"Description: {cfg.topic.description}",
        "",
        f"Articles to synthesise ({len(items)} total):",
        "",
    ]
    for i, item in enumerate(items, 1):
        star = "★ NEW SOURCE  " if item.get("is_new_source") else ""
        lines.append(
            f"[{i}] {star}Source: {item['url']}  Score: {item['score']:.2f}\n"
            f"     Summary: {item['summary'][:400]}"
        )
    return "\n".join(lines)


async def generate_report(
    topic_id: int,
    cfg: TipsterConfig,
    bus: EventBus,
) -> Optional[dict]:
    """Generate a report for unreported extracted items.

    Returns a dict with keys: report_id, narrative_md, report_json, items
    or None if there are no unreported items.
    """
    import asyncio
    from functools import partial

    db = get_db()
    try:
        item_repo = ContentItemRepo(db)
        raw_items = item_repo.list_unreported(topic_id)
        if not raw_items:
            return None

        # Pull primitive values before closing session
        item_data: list[dict] = []
        item_ids: list[int] = []
        for ci in raw_items:
            # Resolve URL via join — avoid DetachedInstanceError
            url_row = db.query(UrlRegistry.url, UrlRegistry.domain)\
                .filter(UrlRegistry.url_id == ci.url_id).first()
            url = url_row.url if url_row else f"url_id={ci.url_id}"
            domain = url_row.domain if url_row else ""
            item_data.append({
                "item_id": ci.item_id,
                "url_id": ci.url_id,
                "url": url,
                "domain": domain,
                "summary": ci.article_sum_md or ci.raw_text[:300] if ci.raw_text else "(no summary)",
                "score": ci.topic_score or 0.0,
                "is_new_source": ci.is_new_source,
            })
            item_ids.append(ci.item_id)
    finally:
        db.close()

    log.info("Generating report for topic %d: %d items", topic_id, len(item_data))

    # LLM synthesis
    prompt = _build_report_prompt(cfg, item_data)
    loop = asyncio.get_running_loop()
    from tipster import llm as llm_module
    try:
        narrative_md = await loop.run_in_executor(
            None,
            partial(
                llm_module.complete,
                cfg.llm.report_model,
                [
                    {"role": "system", "content": _REPORT_SYSTEM},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=2048,
                temperature=0.3,
                api_base=cfg.llm.api_base,
            ),
        )
    except Exception as exc:
        log.error("Report LLM error: %s", exc)
        return None

    # Build structured JSON
    report_json = json.dumps({
        "topic_id": topic_id,
        "item_count": len(item_data),
        "items": [
            {
                "item_id": d["item_id"],
                "url_id": d["url_id"],
                "url": d["url"],
                "domain": d["domain"],
                "score": d["score"],
                "is_new_source": d["is_new_source"],
                "summary": d["summary"],
            }
            for d in item_data
        ],
    }, indent=2)

    # Persist to DB and mark items reported
    db = get_db()
    try:
        report_repo = ReportRepo(db)
        report = report_repo.add(
            topic_id=topic_id,
            narrative_md=narrative_md,
            report_json=report_json,
        )
        report_id = report.report_id

        item_repo = ContentItemRepo(db)
        item_repo.mark_reported(item_ids)
    finally:
        db.close()

    log.info("Report %d generated (%d items)", report_id, len(item_ids))

    await bus.emit(
        Event(
            kind=EventKind.REPORT_READY,
            message=f"report #{report_id} — {len(item_ids)} item(s)",
            data={
                "report_id": report_id,
                "narrative_md": narrative_md,
                "items": item_data,
                "item_ids": item_ids,
            },
        )
    )

    return {
        "report_id": report_id,
        "narrative_md": narrative_md,
        "report_json": report_json,
        "items": item_data,
        "item_ids": item_ids,
    }
