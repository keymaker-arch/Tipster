"""Link Discovery & Scoring — Phase 3.

Two distinct link-evaluation strategies:

A) Content-aware discovery (select_links_from_content / discover_links):
   Used when we have the full page content.  The LLM sees the page text and
   the candidate links together and decides which links are worth following —
   producing far fewer but higher-quality discoveries than blind URL scoring.

B) URL-only scoring (score_links_batch / score_pending_links):
   Used for links that were deferred (budget exhausted) and stored as
   status='pending_score'.  At retry time we no longer have the source page,
   so we fall back to scoring by URL + anchor text alone.
"""

from __future__ import annotations

import json
import logging
import re
from typing import NamedTuple, Optional
from urllib.parse import urlparse, urlunparse, urlencode, parse_qsl

from tipster import llm as llm_module
from tipster.budget import BudgetGate
from tipster.config import TipsterConfig

log = logging.getLogger("tipster.link_scorer")


class ScoredLink(NamedTuple):
    """Result of scoring one candidate link."""
    url: str
    score: float
    recrawl_type: str   # "periodic" | "one_time"
    check_interval: int  # seconds; the LLM-suggested re-crawl frequency


# Tracking params to strip from URLs
_TRACKING_PARAMS = frozenset({
    "utm_source", "utm_medium", "utm_campaign", "utm_term", "utm_content",
    "fbclid", "gclid", "msclkid", "ref", "source", "mc_cid", "mc_eid",
})

# Max candidates to send to LLM per page (pre-filter reduces this further)
_MAX_CANDIDATES = 40

_SCORE_SYSTEM = """\
You are a link relevance scorer for a web intelligence crawler.
Given a topic and a list of links (URL + anchor text), evaluate each link and return:

1. "score": 0.0–1.0 — how likely the page contains relevant content about the topic
2. "recrawl": true | false — whether this URL is a periodically-updated feed/index/listing
   that should be re-crawled on a schedule (true), or a single static item that only needs
   to be fetched once (false)
3. "interval_hours": integer — estimated hours between meaningful updates (only when recrawl=true)
   - 1–6:   live feeds, trending pages, dashboards, real-time indexes
   - 24:    daily-updated blogs, news sites, changelogs, release pages
   - 168:   weekly-updated sites, digests, newsletters
   - 720:   monthly or infrequently updated resource lists

Return ONLY a valid JSON array (no markdown, no prose), one object per link:
[{"url": "<url>", "score": <float>, "recrawl": <bool>, "interval_hours": <int>}, ...]

Scoring rules:
- 0.8-1.0: very likely to contain relevant content (topic keywords in anchor/URL)
- 0.5-0.8: possibly relevant
- 0.0-0.5: off-topic, navigation, boilerplate, ads
- Use link_score_hints to guide scoring: 'positive' patterns → boost, 'negative' → suppress
- If anchor text matches a negative pattern, score must be < 0.3

Recrawl examples:
- github.com/trending              → recrawl=true,  interval_hours=24
- github.com/user/repo/issues/123  → recrawl=false
- anthropic.com/news               → recrawl=true,  interval_hours=168
- anthropic.com/news/specific-post → recrawl=false
- news.ycombinator.com             → recrawl=true,  interval_hours=6
- en.wikipedia.org/wiki/Article    → recrawl=false
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


_CONTENT_SELECT_SYSTEM = """\
You are a link discovery agent for a web intelligence crawler.

Given a page's content and its outbound links, select the links that are worth
following to find more relevant content about the topic.

For each selected link also classify:
- "recrawl": true if the target URL is a periodically-updated feed, index, or
  listing (e.g. a blog homepage, trending page, release list); false if it is a
  single static item (e.g. an individual article, commit, or issue)
- "interval_hours": estimated hours between meaningful updates (only when
  recrawl=true): 1-6 live feeds; 24 daily; 168 weekly; 720 monthly

Return ONLY a valid JSON array (no markdown, no prose):
[{"url": "...", "recrawl": <bool>, "interval_hours": <int>}, ...]

Selection rules:
- Use the page content to understand which links lead to truly relevant content
- For list/ranking pages: select links to individual items that seem on-topic
- For article pages: select links to related articles or cited sources
- Skip navigation, ads, login/signup, user-profile, and off-topic links
- Select at most 20 links; prefer quality over quantity
"""

_CONTENT_SELECT_MAX_CHARS = 8000  # page content chars sent to the LLM for context


def _build_content_select_prompt(
    cfg: TipsterConfig,
    text: str,
    candidates: list[tuple[str, str]],
) -> str:
    hints = cfg.topic.relevance_hints
    neg = cfg.topic.link_score_hints.get("negative", [])
    links_text = "\n".join(
        f'{i+1}. URL: {url}  Anchor: "{anchor}"'
        for i, (url, anchor) in enumerate(candidates)
    )
    return (
        f"Topic: {cfg.topic.name}\n"
        f"Description: {cfg.topic.description}\n"
        f"Relevance hints: {', '.join(hints)}\n"
        f"Negative patterns (skip): {', '.join(neg)}\n\n"
        f"--- Page content (excerpt) ---\n{text[:_CONTENT_SELECT_MAX_CHARS]}\n\n"
        f"--- Candidate links ---\n{links_text}"
    )


def select_links_from_content(
    text: str,
    candidates: list[tuple[str, str]],
    cfg: TipsterConfig,
    budget: Optional[BudgetGate] = None,
) -> tuple[list[ScoredLink], bool]:
    """Select links to follow using full page content as context.

    The LLM sees the actual page text and decides which links are relevant —
    far more accurate than scoring by URL/anchor patterns alone.

    Returns (selected_links, budget_ok).  Selected links have score=1.0
    (the LLM already filtered them; downstream threshold checks will pass).
    budget_ok=False means budget was exhausted before the call.
    """
    if not candidates:
        return [], True
    if budget is not None and not budget.can_proceed():
        return [], False

    prompt = _build_content_select_prompt(cfg, text, candidates)

    try:
        raw, tokens, cost = llm_module.complete_with_usage(
            model=cfg.llm.link_score_model,
            messages=[
                {"role": "system", "content": _CONTENT_SELECT_SYSTEM},
                {"role": "user", "content": prompt},
            ],
            max_tokens=1024,
            temperature=0.1,
            api_base=cfg.llm.api_base,
        )
        if budget is not None:
            budget.record(tokens, cost)
    except Exception as exc:
        log.warning("Content-aware link selection LLM error: %s", exc)
        return [], True  # don't block progress on LLM failure

    raw = raw.strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```[a-z]*\n?", "", raw)
        raw = re.sub(r"\n?```$", "", raw)

    try:
        results = json.loads(raw)
        selected: list[ScoredLink] = []
        candidate_urls = {normalise_url(u) for u, _ in candidates}
        for r in results:
            url = str(r.get("url", ""))
            if not url or normalise_url(url) not in candidate_urls:
                continue  # skip hallucinated URLs
            recrawl = bool(r.get("recrawl", True))
            recrawl_type = "periodic" if recrawl else "one_time"
            check_interval = _parse_check_interval(r.get("interval_hours"), recrawl)
            selected.append(ScoredLink(url, 1.0, recrawl_type, check_interval))
        return selected, True
    except (json.JSONDecodeError, KeyError, TypeError) as exc:
        log.warning("Content-aware link selection parse error: %s — raw: %s", exc, raw[:120])
        return [], True


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


_MAX_INTERVAL_SECS = 30 * 24 * 3600  # 30 days ceiling
_MIN_INTERVAL_SECS = 3600             # 1 hour floor


def _parse_check_interval(raw_hours: object, recrawl: bool) -> int:
    """Convert LLM-returned interval_hours to clamped seconds."""
    if not recrawl:
        return _MAX_INTERVAL_SECS  # won't be used, but set a safe value
    try:
        hours = int(raw_hours) if raw_hours is not None else 24
    except (TypeError, ValueError):
        hours = 24
    return max(_MIN_INTERVAL_SECS, min(hours * 3600, _MAX_INTERVAL_SECS))


def score_links_batch(
    candidates: list[tuple[str, str]],
    cfg: TipsterConfig,
    budget: Optional[BudgetGate] = None,
) -> tuple[list[ScoredLink], bool]:
    """Score a batch of (url, anchor) candidates with one LLM call.

    Returns (scored_list, budget_ok).  Each ScoredLink carries relevance score,
    recrawl classification, and the suggested check interval.
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
            max_tokens=1024,
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
        scored: list[ScoredLink] = []
        for r in results:
            if "url" not in r or "score" not in r:
                continue
            recrawl = bool(r.get("recrawl", True))  # default periodic if field missing
            recrawl_type = "periodic" if recrawl else "one_time"
            check_interval = _parse_check_interval(r.get("interval_hours"), recrawl)
            scored.append(ScoredLink(str(r["url"]), float(r["score"]), recrawl_type, check_interval))
        return scored, True
    except (json.JSONDecodeError, KeyError, TypeError) as exc:
        log.warning("Link score parse error: %s — raw: %s", exc, raw[:120])
        return [], True


async def discover_links(
    text: str,
    link_data: list[tuple[str, str]],
    source_url: str,
    topic_id: int,
    cfg: TipsterConfig,
    budget: BudgetGate,
    bus,
) -> int:
    """Select outbound links to follow using the page content as context.

    The LLM sees the actual page text alongside the candidate links and decides
    which are worth crawling — far more accurate than pure URL/anchor scoring.

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

    # If budget exhausted, queue all candidates for deferred URL-only scoring
    if not budget.can_proceed():
        await _queue_unscored(candidates, topic_id, cfg, bus)
        return 0

    # Content-aware selection: LLM sees the page text + candidates and chooses
    loop = asyncio.get_running_loop()
    selected, budget_ok = await loop.run_in_executor(
        None, partial(select_links_from_content, text, candidates, cfg, budget)
    )

    added = 0
    selected_urls = {normalise_url(link.url) for link in selected}

    db = get_db()
    try:
        repo = UrlRegistryRepo(db)
        for link in selected:
            repo.add(
                topic_id=topic_id,
                url=link.url,
                added_by="discovery",
                relevance_score=link.score,
                check_interval=link.check_interval,
                recrawl_type=link.recrawl_type,
            )
            added += 1
            await bus.emit(
                Event(
                    kind=EventKind.LINK_DISCOVERED,
                    url=link.url,
                    score=link.score,
                    message=(
                        f"recrawl={link.recrawl_type}"
                        + (f" every {link.check_interval // 3600}h" if link.recrawl_type == "periodic" else "")
                        + " → queued"
                    ),
                )
            )

        # Candidates the LLM skipped but budget ran out mid-call → defer
        if not budget_ok:
            unscored = [(u, a) for u, a in candidates if normalise_url(u) not in selected_urls]
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
        for link in scored:
            url_id = pending_ids.get(link.url)
            if url_id is None:
                continue
            entry = db.query(UR).filter_by(url_id=url_id).first()
            if entry is None:
                continue
            if link.score >= threshold:
                entry.status = "pending"
                entry.relevance_score = link.score
                entry.recrawl_type = link.recrawl_type
                entry.check_interval = link.check_interval
                promoted += 1
                await bus.emit(
                    Event(
                        kind=EventKind.LINK_DISCOVERED,
                        url=link.url,
                        score=link.score,
                        message=(
                            f"deferred score={link.score:.2f} "
                            f"recrawl={link.recrawl_type}"
                            + (f" every {link.check_interval // 3600}h" if link.recrawl_type == "periodic" else "")
                            + " → queued"
                        ),
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
