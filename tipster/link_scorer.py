"""Link Discovery & Scoring — Phase 3.

For each relevant page crawled:
1. Pre-filter outbound links (strip known-bad, already-known, negative-hint matches)
2. Batch-score remaining candidates with one LLM call
3. Insert links ≥ link_score_threshold into url_registry as status='pending'
4. When budget is exhausted mid-batch, save unscored links as status='pending_score'
   so they are scored in the next slice instead of being silently dropped.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Optional
from urllib.parse import urlparse, urlunparse, urlencode, parse_qsl

from tipster import llm as llm_module
from tipster.budget import BudgetGate
from tipster.config import TipsterConfig

log = logging.getLogger("tipster.link_scorer")

# Tracking params to strip from URLs
_TRACKING_PARAMS = frozenset({
    "utm_source", "utm_medium", "utm_campaign", "utm_term", "utm_content",
    "fbclid", "gclid", "msclkid", "ref", "source", "mc_cid", "mc_eid",
})

# Max candidates to send to LLM per page (pre-filter reduces this further)
_MAX_CANDIDATES = 40

_SCORE_SYSTEM = """\
You are a link relevance scorer for a web intelligence crawler.
Given a topic and a list of links (URL + anchor text), score each link 0.0–1.0
for how likely it is to lead to relevant content about the topic.

Return ONLY a valid JSON array (no markdown, no prose), one object per link:
[{"url": "<url>", "score": <0.0-1.0>}, ...]

Scoring rules:
- 0.8-1.0: very likely to contain relevant content (topic keywords in anchor/URL)
- 0.5-0.8: possibly relevant
- 0.0-0.5: off-topic, navigation, boilerplate, ads
- Use link_score_hints to guide scoring: 'positive' patterns → boost, 'negative' → suppress
- If anchor text matches a negative pattern, score must be < 0.3
"""


def normalise_url(url: str) -> str:
    """Normalise a URL for deduplication."""
    try:
        p = urlparse(url)
        scheme = p.scheme.lower()
        netloc = p.netloc.lower()
        # Remove default ports
        if netloc.endswith(":80") and scheme == "http":
            netloc = netloc[:-3]
        elif netloc.endswith(":443") and scheme == "https":
            netloc = netloc[:-4]
        # Strip tracking params, sort remaining
        qs = [(k, v) for k, v in parse_qsl(p.query) if k.lower() not in _TRACKING_PARAMS]
        qs.sort()
        path = p.path.rstrip("/") or "/"
        return urlunparse((scheme, netloc, path, p.params, urlencode(qs), ""))
    except Exception:
        return url


def _matches_any(text: str, patterns: list[str]) -> bool:
    text_lower = text.lower()
    return any(p.lower() in text_lower for p in patterns)


def _prefilter(
    link_data: list[tuple[str, str]],
    known_urls: set[str],
    blacklist_domains: list[str],
    negative_hints: list[str],
    topic_id_for_log: int,
) -> tuple[list[tuple[str, str]], list[str]]:
    """Split link_data into (candidates_to_score, pre_rejected_urls).

    Rejects:
    - already-known URLs
    - blacklisted domains
    - links whose anchor text matches a negative hint (hard reject, skip LLM)
    - non-http(s) or obviously irrelevant extensions
    """
    _SKIP_EXTS = {".pdf", ".zip", ".tar.gz", ".jpg", ".jpeg", ".png", ".gif",
                  ".mp4", ".mp3", ".css", ".js", ".ico", ".svg", ".woff", ".ttf"}

    candidates: list[tuple[str, str]] = []
    rejected: list[str] = []
    seen: set[str] = set()

    for url, anchor in link_data:
        norm = normalise_url(url)
        if norm in seen:
            continue
        seen.add(norm)

        parsed = urlparse(norm)
        # Skip already-known
        if norm in known_urls or url in known_urls:
            continue
        # Skip blacklisted domains
        if any(b in parsed.netloc for b in blacklist_domains if b):
            rejected.append(norm)
            continue
        # Skip bad extensions
        if any(parsed.path.lower().endswith(ext) for ext in _SKIP_EXTS):
            rejected.append(norm)
            continue
        # Hard-reject on negative hints
        combined = f"{anchor} {norm}"
        if negative_hints and _matches_any(combined, negative_hints):
            rejected.append(norm)
            continue

        candidates.append((norm, anchor))
        if len(candidates) >= _MAX_CANDIDATES:
            break

    return candidates, rejected


def _build_score_prompt(cfg: TipsterConfig, candidates: list[tuple[str, str]]) -> str:
    pos = cfg.topic.link_score_hints.get("positive", [])
    neg = cfg.topic.link_score_hints.get("negative", [])
    hints = cfg.topic.relevance_hints

    links_text = "\n".join(
        f'{i+1}. URL: {url}  Anchor: "{anchor}"'
        for i, (url, anchor) in enumerate(candidates)
    )
    return (
        f"Topic: {cfg.topic.name}\n"
        f"Description: {cfg.topic.description}\n"
        f"Relevance hints: {', '.join(hints)}\n"
        f"Positive link patterns: {', '.join(pos)}\n"
        f"Negative link patterns: {', '.join(neg)}\n\n"
        f"--- Links to score ---\n{links_text}"
    )


def score_links_batch(
    candidates: list[tuple[str, str]],
    cfg: TipsterConfig,
    budget: Optional[BudgetGate] = None,
) -> tuple[list[tuple[str, float]], bool]:
    """Score a batch of (url, anchor) candidates with one LLM call.

    Returns (scored_list, budget_ok) where scored_list is [(url, score), ...].
    budget_ok=False means the budget was exhausted before the call.
    """
    if not candidates:
        return [], True

    if budget is not None and not budget.can_proceed():
        return [], False

    prompt = _build_score_prompt(cfg, candidates)

    try:
        raw, tokens, cost = llm_module.complete_with_usage(
            model=cfg.llm.link_score_model,
            messages=[
                {"role": "system", "content": _SCORE_SYSTEM},
                {"role": "user", "content": prompt},
            ],
            max_tokens=512,
            temperature=0.1,
            api_base=cfg.llm.api_base,
        )
        if budget is not None:
            budget.record(tokens, cost)
    except Exception as exc:
        log.warning("Link scoring LLM error: %s", exc)
        return [], True  # treat as scored (don't block progress)

    raw = raw.strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```[a-z]*\n?", "", raw)
        raw = re.sub(r"\n?```$", "", raw)

    try:
        results = json.loads(raw)
        scored = [(str(r["url"]), float(r["score"])) for r in results if "url" in r and "score" in r]
        return scored, True
    except (json.JSONDecodeError, KeyError, TypeError) as exc:
        log.warning("Link score parse error: %s — raw: %s", exc, raw[:120])
        return [], True


async def discover_links(
    link_data: list[tuple[str, str]],
    source_url: str,
    topic_id: int,
    cfg: TipsterConfig,
    budget: BudgetGate,
    bus,
) -> int:
    """Score outbound links from a crawled page and register passing ones.

    Returns the number of new URLs added to the registry.
    """
    import asyncio
    from functools import partial
    from tipster.db.session import get_db
    from tipster.db.repositories.url_registry import UrlRegistryRepo
    from tipster.events import Event, EventKind

    if not link_data:
        return 0

    # Load known URLs for dedup
    db = get_db()
    try:
        repo = UrlRegistryRepo(db)
        existing_rows = db.query(__import__("tipster.db.models", fromlist=["UrlRegistry"]).UrlRegistry)\
            .filter_by(topic_id=topic_id).all()
        known_urls = {normalise_url(r.url) for r in existing_rows}
        known_urls |= {r.url for r in existing_rows}
        blacklist = cfg.sources.blacklist
        negative_hints = cfg.topic.link_score_hints.get("negative", [])
    finally:
        db.close()

    candidates, _ = _prefilter(link_data, known_urls, blacklist, negative_hints, topic_id)

    if not candidates:
        return 0

    log.debug("Link discovery: %d candidates from %s", len(candidates), source_url)

    # If budget exhausted, queue all candidates as pending_score
    if not budget.can_proceed():
        await _queue_unscored(candidates, topic_id, cfg, bus)
        return 0

    # Score in batches of _MAX_CANDIDATES (already limited by pre-filter, but safe)
    loop = asyncio.get_running_loop()
    scored, budget_ok = await loop.run_in_executor(
        None, partial(score_links_batch, candidates, cfg, budget)
    )

    threshold = cfg.discovery.link_score_threshold
    added = 0

    db = get_db()
    try:
        repo = UrlRegistryRepo(db)
        scored_urls = {normalise_url(url) for url, _ in scored}

        for url, score in scored:
            if score >= threshold:
                entry = repo.add(
                    topic_id=topic_id,
                    url=url,
                    added_by="discovery",
                    relevance_score=score,
                )
                if entry.added_by == "discovery" or entry.relevance_score != score:
                    # Update score if we have a better one
                    pass
                added += 1
                await bus.emit(
                    Event(
                        kind=EventKind.LINK_DISCOVERED,
                        url=url,
                        score=score,
                        message=f"score={score:.2f} → queued",
                    )
                )

        # Candidates not returned by LLM or budget-exhausted mid-call → queue
        if not budget_ok:
            unscored = [(u, a) for u, a in candidates if normalise_url(u) not in scored_urls]
            if unscored:
                await _queue_unscored(unscored, topic_id, cfg, bus)
    finally:
        db.close()

    return added


async def score_pending_links(
    topic_id: int,
    cfg: TipsterConfig,
    budget: BudgetGate,
    bus,
) -> int:
    """Score URLs that were queued as pending_score in a previous slice."""
    import asyncio
    from functools import partial
    from tipster.db.session import get_db
    from tipster.db.repositories.url_registry import UrlRegistryRepo
    from tipster.events import Event, EventKind

    db = get_db()
    try:
        repo = UrlRegistryRepo(db)
        pending = repo.list_pending_score(topic_id)
        if not pending:
            return 0
        # Use url as anchor (no original anchor text stored, use path as proxy)
        candidates = [(normalise_url(e.url), urlparse(e.url).path) for e in pending]
        pending_ids = {normalise_url(e.url): e.url_id for e in pending}
    finally:
        db.close()

    if not budget.can_proceed():
        return 0

    loop = asyncio.get_running_loop()
    scored, _ = await loop.run_in_executor(
        None, partial(score_links_batch, candidates, cfg, budget)
    )

    threshold = cfg.discovery.link_score_threshold
    promoted = 0

    db = get_db()
    try:
        from tipster.db.models import UrlRegistry as UR
        for url, score in scored:
            url_id = pending_ids.get(url)
            if url_id is None:
                continue
            entry = db.query(UR).filter_by(url_id=url_id).first()
            if entry is None:
                continue
            if score >= threshold:
                entry.status = "pending"
                entry.relevance_score = score
                promoted += 1
                await bus.emit(
                    Event(
                        kind=EventKind.LINK_DISCOVERED,
                        url=url,
                        score=score,
                        message=f"deferred score={score:.2f} → queued",
                    )
                )
            else:
                entry.status = "rejected"
        db.commit()
    finally:
        db.close()

    return promoted


async def _queue_unscored(
    candidates: list[tuple[str, str]],
    topic_id: int,
    cfg: TipsterConfig,
    bus,
) -> None:
    """Save unscored link candidates as status='pending_score' for next slice."""
    from tipster.db.session import get_db
    from tipster.db.repositories.url_registry import UrlRegistryRepo
    from tipster.events import Event, EventKind

    db = get_db()
    try:
        repo = UrlRegistryRepo(db)
        queued = 0
        for url, _ in candidates:
            existing = repo.get_by_url(url)
            if existing is None:
                repo.add(
                    topic_id=topic_id,
                    url=url,
                    added_by="discovery",
                    status="pending_score",
                )
                queued += 1
    finally:
        db.close()

    if queued:
        await bus.emit(
            Event(
                kind=EventKind.LINK_DEFERRED,
                message=f"{queued} link(s) queued for scoring next slice (budget exhausted)",
            )
        )
