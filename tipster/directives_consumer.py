"""Directive Consumers — Phase 4.

Applies active directives from directive_store to the crawler subsystems:
  - BOOST_CRAWL_PRIORITY  → shrink check_interval for matching domain URLs
  - UPDATE_LINK_SCORE_HINT → inject runtime hint into link scorer config
  - EXPAND_TOPIC           → add sub-topic to relevance_hints + emit TUI notice
  - BLACKLIST_SOURCE       → set url_registry status=blacklisted for domain/URL
  - SCHEDULE_DEEP_DIVE     → pin check_interval for a specific URL
"""

from __future__ import annotations

import json
import logging
from typing import Optional

log = logging.getLogger("tipster.directives_consumer")


async def apply_directives(
    topic_id: int,
    cfg,      # TipsterConfig — mutated in-place for runtime overrides
    bus,      # EventBus
) -> int:
    """Apply all active, unapplied directives.

    Returns the count of directives applied.
    """
    from tipster.db.session import get_db
    from tipster.db.repositories.directives import DirectiveRepo
    from tipster.db.models import UrlRegistry
    from tipster.events import Event, EventKind

    db = get_db()
    try:
        dir_repo = DirectiveRepo(db)
        actives = dir_repo.list_active(topic_id)
    finally:
        db.close()

    if not actives:
        return 0

    applied = 0
    for directive in actives:
        dtype = directive.directive_type
        target = directive.target or ""
        params: dict = {}
        if directive.params_json:
            try:
                params = json.loads(directive.params_json)
            except Exception:
                pass

        try:
            if dtype == "BLACKLIST_SOURCE":
                applied += await _apply_blacklist(directive.directive_id, target, topic_id, cfg, bus)

            elif dtype == "BOOST_CRAWL_PRIORITY":
                applied += await _apply_boost(directive.directive_id, target, topic_id, params, cfg, bus)

            elif dtype == "SCHEDULE_DEEP_DIVE":
                applied += await _apply_deep_dive(directive.directive_id, target, topic_id, params, cfg, bus)

            elif dtype == "UPDATE_LINK_SCORE_HINT":
                _apply_hint(cfg, target, params)
                applied += 1
                # Mark applied (hint is baked into runtime config now)
                db = get_db()
                try:
                    DirectiveRepo(db).mark_applied(directive.directive_id)
                finally:
                    db.close()
                await bus.emit(Event(
                    kind=EventKind.DIRECTIVE_APPLIED,
                    message=f"UPDATE_LINK_SCORE_HINT [{params.get('polarity','+')}] for {target}: {params.get('hint','')}",
                ))

            elif dtype == "EXPAND_TOPIC":
                if target and target not in cfg.topic.relevance_hints:
                    cfg.topic.relevance_hints.append(target)
                applied += 1
                db = get_db()
                try:
                    DirectiveRepo(db).mark_applied(directive.directive_id)
                finally:
                    db.close()
                await bus.emit(Event(
                    kind=EventKind.DIRECTIVE_APPLIED,
                    message=f"EXPAND_TOPIC: added '{target}' to relevance hints",
                ))

        except Exception as exc:
            log.warning("Error applying directive %s (%s): %s", directive.directive_id, dtype, exc)

    return applied


async def _apply_blacklist(
    directive_id: int,
    target: str,
    topic_id: int,
    cfg,
    bus,
) -> int:
    """Blacklist a domain or URL — set url_registry status=blacklisted."""
    from tipster.db.session import get_db
    from tipster.db.models import UrlRegistry
    from tipster.db.repositories.directives import DirectiveRepo
    from tipster.events import Event, EventKind

    if not target:
        return 0

    # Add to runtime blacklist in config so the link scorer skips it immediately
    if target not in cfg.sources.blacklist:
        cfg.sources.blacklist.append(target)

    db = get_db()
    try:
        # Update all matching URLs
        rows = db.query(UrlRegistry).filter(
            UrlRegistry.topic_id == topic_id,
        ).all()
        count = 0
        for row in rows:
            if target in row.url or target in row.domain:
                if row.status not in ("blacklisted",):
                    row.status = "blacklisted"
                    row.source_weight = 0.0
                    count += 1
        db.commit()
        DirectiveRepo(db).mark_applied(directive_id)
    finally:
        db.close()

    await bus.emit(Event(
        kind=EventKind.DIRECTIVE_APPLIED,
        message=f"BLACKLIST_SOURCE: {target} — {count} URL(s) blacklisted",
    ))
    log.info("Blacklisted %s (%d URLs)", target, count)
    return 1


async def _apply_boost(
    directive_id: int,
    target: str,
    topic_id: int,
    params: dict,
    cfg,
    bus,
) -> int:
    """Shrink check_interval for matching domain URLs by magnitude factor."""
    from tipster.db.session import get_db
    from tipster.db.models import UrlRegistry
    from tipster.db.repositories.directives import DirectiveRepo
    from tipster.events import Event, EventKind

    if not target:
        return 0

    magnitude = float(params.get("magnitude", 0.5))
    magnitude = max(0.1, min(0.9, magnitude))  # clamp

    db = get_db()
    try:
        rows = db.query(UrlRegistry).filter(
            UrlRegistry.topic_id == topic_id,
            UrlRegistry.status.in_(["pending", "active"]),
        ).all()
        count = 0
        for row in rows:
            if target in row.domain or target in row.url:
                row.check_interval = max(60, int(row.check_interval * magnitude))
                count += 1
        db.commit()
        DirectiveRepo(db).mark_applied(directive_id)
    finally:
        db.close()

    await bus.emit(Event(
        kind=EventKind.DIRECTIVE_APPLIED,
        message=f"BOOST_CRAWL_PRIORITY: {target} — {count} URL(s) interval ×{magnitude:.2f}",
    ))
    log.info("Boosted crawl priority for %s (%d URLs)", target, count)
    return 1


async def _apply_deep_dive(
    directive_id: int,
    target: str,
    topic_id: int,
    params: dict,
    cfg,
    bus,
) -> int:
    """Override check_interval for a specific URL."""
    from tipster.db.session import get_db
    from tipster.db.models import UrlRegistry
    from tipster.db.repositories.directives import DirectiveRepo
    from tipster.events import Event, EventKind

    if not target:
        return 0

    interval_hours = float(params.get("interval_hours", 24))
    interval_seconds = max(60, int(interval_hours * 3600))

    db = get_db()
    try:
        row = db.query(UrlRegistry).filter(
            UrlRegistry.topic_id == topic_id,
            UrlRegistry.url == target,
        ).first()
        if row:
            row.check_interval = interval_seconds
        db.commit()
        DirectiveRepo(db).mark_applied(directive_id)
    finally:
        db.close()

    await bus.emit(Event(
        kind=EventKind.DIRECTIVE_APPLIED,
        message=f"SCHEDULE_DEEP_DIVE: {target} — interval={interval_hours}h",
    ))
    return 1


def _apply_hint(cfg, domain: str, params: dict) -> None:
    """Inject a link score hint into the runtime config."""
    polarity = params.get("polarity", "positive")
    hint = params.get("hint", "")
    if not hint:
        return
    hints = cfg.topic.link_score_hints
    if polarity == "negative":
        neg = hints.get("negative", [])
        if hint not in neg:
            neg.append(hint)
        hints["negative"] = neg
    else:
        pos = hints.get("positive", [])
        if hint not in pos:
            pos.append(hint)
        hints["positive"] = pos
