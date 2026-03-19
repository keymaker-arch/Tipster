"""Fact Extractor — Phase 2.

Runs extraction_model LLM on pending content items, producing:
- extracted_json: structured facts as a JSON blob
- article_sum_md: a concise Markdown summary of the article

The LLM first classifies the page type (article | list | other) and then
applies the appropriate extraction strategy:
  - article: 5–10 sentence Markdown summary + key facts + entities
  - list:    structured item-by-item extraction (trending repos, news feeds, etc.)
  - other:   brief 2–3 sentence description

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
You are a web content extractor for a web intelligence crawler.

STEP 1 — Classify the page type:
- "article": a blog post, essay, news article, documentation page, or research paper
  (primarily prose focused on a single topic or story)
- "list": a ranking or index page containing multiple discrete items — trending
  repositories, news feeds, search results, product listings, release notes, changelogs, etc.
- "other": homepages, profile pages, login pages, error pages, or anything that
  doesn't fit the above

STEP 2 — Extract based on the page type:

For "article":
  Write a 5-10 sentence Markdown summary capturing the main thesis, key claims, and
  conclusions. Include notable facts and named entities.

For "list":
  Extract every distinct item on the page as a structured object. Do NOT truncate —
  capture all items. Include all available metadata per item (name, description, URL,
  stars, author, language, date, score, rank, etc.).

For "other":
  Write a 2-3 sentence description of the page's purpose or content.

If an extraction focus hint is provided, prioritise that aspect in your extraction.

Return ONLY valid JSON (no markdown fences, no prose).

Article format:
{
  "page_type": "article",
  "title": "<title or inferred heading>",
  "summary": "<5-10 sentences in Markdown>",
  "key_facts": ["<fact>", ...],
  "entities": ["<person/org/project>", ...]
}

List format:
{
  "page_type": "list",
  "title": "<list title>",
  "summary": "<1-2 sentences describing what this list is>",
  "items": [
    {"name": "...", "description": "...", "url": "...", <any other available metadata>},
    ...
  ]
}

Other format:
{
  "page_type": "other",
  "title": "<title>",
  "summary": "<2-3 sentences>"
}
"""


def _build_extract_prompt(cfg: TipsterConfig, text: str, prompt_snippet: str = "") -> str:
    hint = f"\nExtraction focus: {prompt_snippet}" if prompt_snippet else ""
    return (
        f"Topic: {cfg.topic.name}\n"
        f"Description: {cfg.topic.description}{hint}\n\n"
        f"--- Page content ---\n{text or '(empty)'}"
    )


def extract_one(
    item_id: int,
    raw_text: str,
    cfg: TipsterConfig,
    budget: Optional[BudgetGate] = None,
    prompt_snippet: str = "",
) -> tuple[bool, str]:
    """Extract facts from one content item.

    Returns (success: bool, message: str).
    Caller is responsible for DB updates.
    """
    if budget is not None and not budget.can_proceed():
        return False, "budget exhausted — deferred to next slice"

    prompt = _build_extract_prompt(cfg, raw_text or "", prompt_snippet)

    try:
        raw, tokens, cost = llm_module.complete_with_usage(
            model=cfg.llm.extraction_model,
            messages=[
                {"role": "system", "content": _EXTRACT_SYSTEM},
                {"role": "user", "content": prompt},
            ],
            max_tokens=4096,
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
    # summary is present for all page types; for lists it describes the list itself
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
            db.query(
                _CI.item_id, _CI.url_id, _CI.raw_text, _CI.topic_score,
                _CI.is_new_source, _UR.url, _UR.domain, _UR.prompt_snippet,
            )
            .join(_UR, _CI.url_id == _UR.url_id)
            .filter(_CI.topic_id == topic_id, _CI.status == "pending_extraction")
            .all()
        )
    finally:
        db.close()

    if not rows:
        return 0

    # rows is a list of named tuples — all primitives, safe after session close
    pending_data = [
        (r.item_id, r.url_id, r.raw_text, r.url, r.domain or "",
         r.topic_score or 0.0, bool(r.is_new_source), r.prompt_snippet or "")
        for r in rows
    ]

    log.debug("EXTRACT PENDING  %d item(s) queued", len(pending_data))
    await bus.emit(
        Event(
            kind=EventKind.EXTRACT_START,
            message=f"extracting {len(pending_data)} pending item(s)…",
        )
    )

    extracted_count = 0
    loop = asyncio.get_running_loop()

    for item_id, url_id, raw_text, item_url, domain, score, is_new_source, prompt_snippet in pending_data:
        if not budget.can_proceed():
            remaining = len(pending_data) - extracted_count
            log.debug("EXTRACT DEFERRED  budget exhausted  remaining=%d", remaining)
            await bus.emit(
                Event(
                    kind=EventKind.EXTRACT_DEFERRED,
                    message=f"budget exhausted — {remaining} item(s) deferred to next slice",
                )
            )
            break

        log.debug("EXTRACT START  item_id=%d  url=%s  text_len=%d  hint=%r",
                  item_id, item_url, len(raw_text or ""), prompt_snippet or "")
        success, payload = await loop.run_in_executor(
            None,
            partial(extract_one, item_id, raw_text or "", cfg, budget, prompt_snippet),
        )

        db = get_db()
        try:
            content_repo = ContentItemRepo(db)
            if success:
                data = json.loads(payload)
                parsed = json.loads(data["extracted_json"])
                content_repo.mark_extracted(
                    item_id=item_id,
                    extracted_json=data["extracted_json"],
                    article_sum_md=data["article_sum_md"],
                )
                extracted_count += 1
                log.debug(
                    "EXTRACT OK  item_id=%d  url=%s  page_type=%s  title=%r  "
                    "facts=%d  entities=%d  summary_len=%d",
                    item_id, item_url,
                    parsed.get("page_type", "?"),
                    parsed.get("title", "")[:80],
                    len(parsed.get("key_facts", [])),
                    len(parsed.get("entities", [])),
                    len(data.get("article_sum_md", "")),
                )
                await bus.emit(
                    Event(
                        kind=EventKind.EXTRACT_OK,
                        url=item_url,
                        message=f"extracted item_id={item_id}",
                        data={
                            "item_id": item_id,
                            "url_id": url_id,
                            "url": item_url,
                            "domain": domain,
                            "score": score,
                            "is_new_source": is_new_source,
                            "page_type": parsed.get("page_type", "article"),
                            "title": parsed.get("title", ""),
                            "summary": parsed.get("summary", ""),
                            "key_facts": parsed.get("key_facts", []),
                            "entities": parsed.get("entities", []),
                            "items": parsed.get("items", []),
                        },
                    )
                )
            else:
                if "budget exhausted" in payload:
                    # Leave as pending_extraction for next slice
                    log.debug("EXTRACT DEFERRED (budget)  item_id=%d  url=%s", item_id, item_url)
                else:
                    log.debug("EXTRACT FAILED  item_id=%d  url=%s  reason=%s",
                              item_id, item_url, payload[:120])
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

    log.debug("EXTRACT PENDING DONE  extracted=%d / %d", extracted_count, len(pending_data))
    return extracted_count
