"""Reporter Module — Phase 4.

Reads all unreported, extracted content items and assembles a Markdown digest
plus structured JSON report deterministically — no LLM required at this stage.

All the intelligence work (page-type classification, summarisation, fact
extraction, list-item parsing) has already been done by the extractor.  The
reporter's job is purely to format those pre-extracted results into a readable
digest and persist them.

Report structure
----------------
# <Topic> — Intelligence Digest
*N finding(s) · date*

## Findings   (sorted by relevance score desc, then crawl time desc)

One section per content item:
  - ★ NEW SOURCE badge if applicable
  - Title + domain + score
  - Per-item summary
  - Key facts list   (articles)
  - Items list       (lists, e.g. GitHub trending)

## Report Summary
  Counts, score range, domains covered.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Optional

from tipster.config import TipsterConfig
from tipster.db.models import ContentItem, UrlRegistry
from tipster.db.repositories.content_items import ContentItemRepo
from tipster.db.repositories.reports import ReportRepo
from tipster.db.session import get_db
from tipster.events import Event, EventBus, EventKind

log = logging.getLogger("tipster.reporter")


def _fmt_score(score: float) -> str:
    return f"{score:.2f}"


def _render_item(item: dict) -> str:
    """Render one content item as a Markdown section."""
    lines: list[str] = []

    new_badge = "★ **NEW SOURCE**  " if item.get("is_new_source") else ""
    title = item.get("title") or item["url"]
    domain = item["domain"] or item["url"]
    score = item.get("score", 0.0)

    lines.append(f"### {new_badge}{title}")
    lines.append(f"*{domain} · relevance {_fmt_score(score)}*")
    lines.append(f"[{item['url']}]({item['url']})")
    lines.append("")

    page_type = item.get("page_type", "article")
    summary = item.get("summary", "")
    if summary:
        lines.append(summary)
        lines.append("")

    if page_type == "article":
        key_facts = item.get("key_facts", [])
        if key_facts:
            lines.append("**Key facts:**")
            for fact in key_facts:
                lines.append(f"- {fact}")
            lines.append("")
        entities = item.get("entities", [])
        if entities:
            lines.append(f"**Entities mentioned:** {', '.join(entities)}")
            lines.append("")

    elif page_type == "list":
        sub_items = item.get("items", [])
        if sub_items:
            lines.append(f"**{len(sub_items)} item(s):**")
            for si in sub_items[:50]:  # cap display at 50 entries
                name = si.get("name") or si.get("title") or ""
                desc = si.get("description") or ""
                url = si.get("url") or ""
                # Collect any extra metadata fields
                meta = {k: v for k, v in si.items()
                        if k not in ("name", "title", "description", "url") and v}
                meta_str = "  ·  ".join(f"{k}: {v}" for k, v in meta.items())
                entry = f"- **{name}**"
                if desc:
                    entry += f" — {desc}"
                if url:
                    entry += f"  [{url}]({url})"
                if meta_str:
                    entry += f"  *({meta_str})*"
                lines.append(entry)
            lines.append("")

    lines.append("---")
    lines.append("")
    return "\n".join(lines)


def _build_narrative(cfg: TipsterConfig, items: list[dict], generated_at: datetime) -> str:
    """Assemble the full Markdown digest from pre-extracted item data."""
    date_str = generated_at.strftime("%Y-%m-%d %H:%M UTC")
    n = len(items)

    sections: list[str] = [
        f"# {cfg.topic.name} — Intelligence Digest",
        f"*{n} finding(s) · {date_str}*",
        "",
        "---",
        "",
        "## Findings",
        "",
    ]

    for item in items:
        sections.append(_render_item(item))

    # Summary statistics
    scores = [d["score"] for d in items if d.get("score") is not None]
    domains = sorted({d["domain"] for d in items if d.get("domain")})
    new_sources = [d for d in items if d.get("is_new_source")]

    sections += [
        "## Report Summary",
        "",
        f"- **Findings:** {n}",
        f"- **Domains covered:** {len(domains)} ({', '.join(domains[:10])}{'…' if len(domains) > 10 else ''})",
        f"- **New sources:** {len(new_sources)}",
    ]
    if scores:
        sections.append(f"- **Score range:** {min(scores):.2f} – {max(scores):.2f}")
    sections.append(f"- **Generated:** {date_str}")
    sections.append("")

    return "\n".join(sections)


async def generate_report(
    topic_id: int,
    cfg: TipsterConfig,
    bus: EventBus,
) -> Optional[dict]:
    """Generate a report for unreported extracted items.

    Assembles the Markdown digest deterministically from already-extracted data.
    Returns a dict with keys: report_id, narrative_md, report_json, items, item_ids
    or None if there are no unreported items.
    """
    db = get_db()
    try:
        item_repo = ContentItemRepo(db)
        raw_items = item_repo.list_unreported(topic_id)
        if not raw_items:
            return None

        item_data: list[dict] = []
        item_ids: list[int] = []

        for ci in raw_items:
            url_row = db.query(UrlRegistry.url, UrlRegistry.domain) \
                .filter(UrlRegistry.url_id == ci.url_id).first()
            url = url_row.url if url_row else f"url_id={ci.url_id}"
            domain = url_row.domain if url_row else ""

            # Parse the extractor's JSON output
            extracted: dict = {}
            if ci.extracted_json:
                try:
                    extracted = json.loads(ci.extracted_json)
                except (json.JSONDecodeError, TypeError):
                    pass

            item_data.append({
                "item_id": ci.item_id,
                "url_id": ci.url_id,
                "url": url,
                "domain": domain,
                "score": ci.topic_score or 0.0,
                "is_new_source": ci.is_new_source,
                # Fields from extractor
                "page_type": extracted.get("page_type", "article"),
                "title": extracted.get("title", ""),
                "summary": extracted.get("summary") or ci.article_sum_md or "",
                "key_facts": extracted.get("key_facts", []),
                "entities": extracted.get("entities", []),
                "items": extracted.get("items", []),  # for list pages
            })
            item_ids.append(ci.item_id)
    finally:
        db.close()

    # Sort: highest score first, then by item_id desc (newest)
    item_data.sort(key=lambda d: (-d["score"], -d["item_id"]))

    log.info("Generating report for topic %d: %d items", topic_id, len(item_data))

    generated_at = datetime.now(timezone.utc)
    narrative_md = _build_narrative(cfg, item_data, generated_at)

    report_json = json.dumps({
        "topic_id": topic_id,
        "generated_at": generated_at.isoformat(),
        "item_count": len(item_data),
        "items": [
            {
                "item_id": d["item_id"],
                "url_id": d["url_id"],
                "url": d["url"],
                "domain": d["domain"],
                "score": d["score"],
                "is_new_source": d["is_new_source"],
                "page_type": d["page_type"],
                "title": d["title"],
                "summary": d["summary"],
                "key_facts": d["key_facts"],
                "entities": d["entities"],
                "items": d["items"],
            }
            for d in item_data
        ],
    }, indent=2)

    # Persist and mark items reported
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
