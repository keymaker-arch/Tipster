"""Provider-agnostic LLM interface via LiteLLM.

All LLM calls go through this module.  The API at api-vip.codex-for.me
requires stream=True, so we always stream and accumulate the response.
"""

from __future__ import annotations

import os
from typing import Optional

import litellm


# Suppress LiteLLM's verbose logging by default
litellm.suppress_debug_info = True
# Allow LiteLLM to silently drop params not supported by a given model (e.g. gpt-5 + temperature)
litellm.drop_params = True


def _apply_env_overrides(model: str, api_base: Optional[str] = None) -> dict:
    """Build extra kwargs for litellm.completion based on env / config."""
    kwargs: dict = {}
    base = api_base or os.environ.get("OPENAI_API_BASE") or os.environ.get("TIPSTER_API_BASE")
    if base:
        kwargs["api_base"] = base
    api_key = (
        os.environ.get("OPENAI_API_KEY")
        or os.environ.get("TIPSTER_LLM_API_KEY")
    )
    if api_key:
        kwargs["api_key"] = api_key
    return kwargs


def complete(
    model: str,
    messages: list[dict],
    *,
    max_tokens: int = 2048,
    temperature: float = 0.2,
    api_base: Optional[str] = None,
) -> str:
    """Run a chat completion and return the assistant's full text."""
    text, _, _ = complete_with_usage(
        model, messages, max_tokens=max_tokens, temperature=temperature, api_base=api_base
    )
    return text


def complete_with_usage(
    model: str,
    messages: list[dict],
    *,
    max_tokens: int = 2048,
    temperature: float = 0.2,
    api_base: Optional[str] = None,
) -> tuple[str, int, float]:
    """Run a chat completion and return (text, total_tokens, cost_usd).

    Always uses streaming internally (required by the test API endpoint).
    Token count is read from the final stream chunk's usage field when available,
    otherwise estimated from text length.  Cost uses litellm.completion_cost()
    with a safe fallback of $0.000002 / token if the model is unknown.
    """
    kwargs = _apply_env_overrides(model, api_base)
    response = litellm.completion(
        model=model,
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
        stream=True,
        stream_options={"include_usage": True},
        **kwargs,
    )
    result_parts: list[str] = []
    total_tokens: int = 0
    for chunk in response:
        delta = chunk.choices[0].delta.content
        if delta:
            result_parts.append(delta)
        # Capture usage from final chunk if the API provides it
        if hasattr(chunk, "usage") and chunk.usage:
            try:
                total_tokens = int(chunk.usage.total_tokens or 0)
            except (AttributeError, TypeError, ValueError):
                pass

    text = "".join(result_parts)

    # Fallback token estimate: ~4 chars per token for English
    if total_tokens == 0:
        input_chars = sum(len(m.get("content", "")) for m in messages)
        output_chars = len(text)
        total_tokens = max(1, (input_chars + output_chars) // 4)

    # Cost estimate
    cost_usd = 0.0
    try:
        cost_usd = litellm.completion_cost(
            model=model,
            prompt=str(messages),
            completion=text,
        )
    except Exception:
        # Unknown model — use a conservative default rate
        cost_usd = total_tokens * 0.000002

    return text, total_tokens, cost_usd


def verify(model: str, api_base: Optional[str] = None, api_key: Optional[str] = None) -> bool:
    """Send a minimal verification call.  Returns True if successful."""
    import os as _os

    env_backup: dict[str, Optional[str]] = {}
    if api_key:
        env_backup["OPENAI_API_KEY"] = _os.environ.get("OPENAI_API_KEY")
        _os.environ["OPENAI_API_KEY"] = api_key
    if api_base:
        env_backup["OPENAI_API_BASE"] = _os.environ.get("OPENAI_API_BASE")
        _os.environ["OPENAI_API_BASE"] = api_base

    try:
        result = complete(
            model=model,
            messages=[{"role": "user", "content": "Reply with OK"}],
            max_tokens=10,
            api_base=api_base,
        )
        return bool(result.strip())
    except Exception:
        return False
    finally:
        for k, v in env_backup.items():
            if v is None:
                _os.environ.pop(k, None)
            else:
                _os.environ[k] = v
