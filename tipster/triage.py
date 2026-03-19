"""Relevance Triage — LLM-based scoring of crawled content.

Uses triage_model from config. Returns a relevance score 0.0-1.0 and a boolean.
"""

from __future__ import annotations

import json
import re
from tipster import llm as llm_module
from tipster.config import TipsterConfig

_TRIAGE_SYSTEM = """\
You are a relevance filter for a web intelligence crawler.
Given a topic description, relevance hints, and a text excerpt, decide whether the content
is relevant to the topic.

Return ONLY valid JSON (no markdown, no prose):
{"relevant": true/false, "score": <0.0-1.0>, "reason": "<one sentence>"}

Rules:
- score 0.8-1.0: directly on-topic
- score 0.5-0.8: tangentially related
- score 0.0-0.5: off-topic
- relevant=true if score >= 0.5
"""


def _build_triage_prompt(cfg: TipsterConfig, text: str) -> str:
    hints = ", ".join(cfg.topic.relevance_hints) if cfg.topic.relevance_hints else "none"
    excerpt = text[:3000] if text else "(empty page)"
    return (
        f"Topic: {cfg.topic.name}\n"
        f"Description: {cfg.topic.description}\n"
        f"Relevance hints: {hints}\n\n"
        f"--- Content excerpt ---\n{excerpt}"
    )


def triage(
    text: str,
    cfg: TipsterConfig,
) -> tuple[bool, float, str, int, float]:
    """Synchronous triage call.

    Returns (relevant, score, reason, tokens_used, cost_usd).
    Budget checking and recording are the caller's responsibility.
    """
    if not text or not text.strip():
        return False, 0.0, "empty page", 0, 0.0

    prompt = _build_triage_prompt(cfg, text)

    try:
        raw, tokens, cost = llm_module.complete_with_usage(
            model=cfg.llm.triage_model,
            messages=[
                {"role": "system", "content": _TRIAGE_SYSTEM},
                {"role": "user", "content": prompt},
            ],
            max_tokens=128,
            temperature=0.1,
            api_base=cfg.llm.api_base,
        )
    except Exception as exc:
        return False, 0.0, f"LLM error: {exc}", 0, 0.0

    raw = raw.strip()
    # Strip markdown fences
    if raw.startswith("```"):
        raw = re.sub(r"^```[a-z]*\n?", "", raw)
        raw = re.sub(r"\n?```$", "", raw)

    try:
        result = json.loads(raw)
        relevant = bool(result.get("relevant", False))
        score = float(result.get("score", 0.0))
        reason = str(result.get("reason", ""))
        return relevant, score, reason, tokens, cost
    except (json.JSONDecodeError, KeyError, ValueError):
        # Fallback: try to parse keywords
        lower = raw.lower()
        relevant = '"relevant": true' in lower or "'relevant': true" in lower
        return relevant, 0.5 if relevant else 0.0, raw[:100], tokens, cost


async def triage_async(
    text: str,
    cfg: TipsterConfig,
) -> tuple[bool, float, str, int, float]:
    """Async wrapper — runs triage in thread pool to avoid blocking the event loop.

    Budget checking must happen before calling this; budget recording must happen
    after it returns, in the event loop (not inside the executor thread).
    """
    import asyncio
    from functools import partial

    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, partial(triage, text, cfg))
