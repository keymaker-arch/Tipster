"""Onboarding logic for `tipster init`.

Step 0: Collect LLM provider credentials, write to .env, verify via LiteLLM.
Step 1: User describes topic in free text → LLM extracts structure → tipster.yaml.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Optional

import click
from dotenv import dotenv_values, set_key
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Confirm, Prompt
from rich.syntax import Syntax

from tipster import llm as llm_module

console = Console()

# ---------------------------------------------------------------------------
# Step 0 — LLM provider setup
# ---------------------------------------------------------------------------

DEFAULT_API_BASE = "https://api-vip.codex-for.me/v1"
DEFAULT_MODEL = "openai/gpt-5"


def _env_has_valid_key(env_path: Path) -> bool:
    """Return True if .env already has a non-empty API key."""
    if not env_path.exists():
        return False
    vals = dotenv_values(env_path)
    return bool(vals.get("OPENAI_API_KEY") or vals.get("TIPSTER_LLM_API_KEY"))


def step0_provider_setup(env_path: Path, force: bool = False) -> dict:
    """Interactive Step 0: collect LLM credentials and verify them.

    Returns a dict with keys: api_base, api_key, model.
    """
    console.print(Panel("[bold cyan]Step 0 — LLM Provider Setup[/bold cyan]", expand=False))

    # If .env already exists with a key, skip unless forced
    if _env_has_valid_key(env_path) and not force:
        console.print(
            f"[green]✓[/green] Found existing credentials in [bold]{env_path}[/bold]. "
            "Skipping provider setup."
        )
        vals = dotenv_values(env_path)
        api_key = vals.get("OPENAI_API_KEY") or vals.get("TIPSTER_LLM_API_KEY", "")
        api_base = vals.get("OPENAI_API_BASE") or vals.get("TIPSTER_API_BASE") or DEFAULT_API_BASE
        model = vals.get("TIPSTER_MODEL") or DEFAULT_MODEL
        return {"api_base": api_base, "api_key": api_key, "model": model}

    console.print(
        "\nTipster needs an OpenAI-compatible LLM endpoint.\n"
        f"Default: [cyan]{DEFAULT_API_BASE}[/cyan]\n"
    )

    api_base = Prompt.ask(
        "API base URL",
        default=DEFAULT_API_BASE,
    ).strip().rstrip("/")

    api_key = Prompt.ask(
        "API key",
        password=True,
        default="",
    ).strip()

    model = Prompt.ask(
        "Default model name (LiteLLM format, e.g. openai/gpt-5)",
        default=DEFAULT_MODEL,
    ).strip()

    # Verification loop
    while True:
        console.print("\n[dim]Verifying LLM connection…[/dim]")
        ok = llm_module.verify(model=model, api_base=api_base, api_key=api_key)
        if ok:
            console.print("[green]✓ LLM connection verified.[/green]")
            break
        console.print(
            "[red]✗ Verification failed.[/red] Possible causes: wrong API key, "
            "unreachable endpoint, or invalid model name."
        )
        if not Confirm.ask("Try again with different credentials?", default=True):
            raise click.Abort()
        api_base = Prompt.ask("API base URL", default=api_base).strip().rstrip("/")
        api_key = Prompt.ask("API key", password=True, default="").strip()
        model = Prompt.ask("Model name", default=model).strip()

    # Write to .env
    env_path.touch(mode=0o600, exist_ok=True)
    if api_key:
        set_key(str(env_path), "OPENAI_API_KEY", api_key)
    if api_base and api_base != DEFAULT_API_BASE:
        set_key(str(env_path), "OPENAI_API_BASE", api_base)
    else:
        set_key(str(env_path), "OPENAI_API_BASE", api_base)
    set_key(str(env_path), "TIPSTER_MODEL", model)

    # Ensure .env is in .gitignore
    _ensure_gitignore(env_path.parent)

    console.print(f"[green]✓ Credentials written to [bold]{env_path}[/bold][/green]")
    return {"api_base": api_base, "api_key": api_key, "model": model}


def _ensure_gitignore(directory: Path) -> None:
    gi = directory / ".gitignore"
    entry = ".env\n"
    if gi.exists():
        contents = gi.read_text()
        if ".env" not in contents:
            gi.write_text(contents + entry)
    else:
        gi.write_text(entry)


# ---------------------------------------------------------------------------
# Step 1 — Topic description → tipster.yaml
# ---------------------------------------------------------------------------

_ONBOARD_SYSTEM_PROMPT = """\
You are a configuration assistant for Tipster, a web intelligence crawler.
The user will describe what they want to monitor. Your job is to extract structured
configuration in JSON, then it will be formatted into tipster.yaml.

Return ONLY valid JSON (no markdown fences, no prose) with this exact schema:
{
  "topic_name": "<short title>",
  "description": "<2-4 sentence description for relevance prompts>",
  "relevance_hints": ["<term1>", "<term2>", ...],
  "link_score_hints": {
    "positive": ["<anchor pattern1>", ...],
    "negative": ["<anchor pattern1>", ...]
  },
  "seed_urls": ["<url1>", "<url2>", ...],
  "domain_weights": {"<domain>": <0.0-1.0>, ...},
  "report_interval": "daily",
  "report_time": "08:00",
  "slice_duration_minutes": 60,
  "max_tokens_per_slice": 500000,
  "max_cost_per_slice_usd": 0.50
}

Rules:
- seed_urls: ONLY include URLs that the user explicitly mentioned in their message.
  Do NOT invent or suggest URLs. If the user mentioned no URLs, return an empty list.
- relevance_hints: 4-8 concise keyword phrases.
- link_score_hints: 3-6 positive patterns (words suggesting relevant links), 3-6 negative.
- domain_weights: assign 0.7-0.95 to authoritative domains the user explicitly named; omit others.
- Keep topic_name under 50 characters.
"""


def _verify_urls(urls: list[str]) -> tuple[list[str], list[str]]:
    """Check each URL with a real HTTP HEAD request.

    Returns (reachable, unreachable) lists.
    """
    import httpx

    reachable: list[str] = []
    unreachable: list[str] = []
    headers = {"User-Agent": "Tipster/0.1 (+https://github.com/tipster)"}

    for url in urls:
        try:
            with httpx.Client(follow_redirects=True, timeout=10.0) as client:
                resp = client.head(url, headers=headers)
                if resp.status_code < 500:
                    reachable.append(url)
                else:
                    unreachable.append(url)
        except Exception:
            unreachable.append(url)

    return reachable, unreachable


def step1_generate_yaml(
    api_base: str,
    api_key: str,
    model: str,
    yaml_path: Path,
    db_path: Path,
) -> Optional[dict]:
    """Interactive Step 1: collect topic description and generate tipster.yaml.

    Returns the parsed config dict (so the caller can seed the DB), or None if aborted.
    """
    console.print(Panel("[bold cyan]Step 1 — Topic Setup[/bold cyan]", expand=False))
    console.print(
        "\nDescribe what you want to monitor. You can include starting URLs.\n"
        "[dim]Example: 'Monitor AI safety research from LessWrong and arXiv. "
        "Focus on alignment, interpretability, and RLHF.'[/dim]\n"
    )

    description = Prompt.ask("What do you want to monitor?").strip()
    if not description:
        raise click.UsageError("Topic description cannot be empty.")

    console.print("\n[dim]Extracting configuration with LLM…[/dim]")

    # Set env vars for this call
    _set_env(api_base, api_key)

    raw_json = llm_module.complete(
        model=model,
        messages=[
            {"role": "system", "content": _ONBOARD_SYSTEM_PROMPT},
            {"role": "user", "content": description},
        ],
        max_tokens=1024,
        temperature=0.2,
        api_base=api_base,
    )

    # Parse JSON (strip any accidental markdown fences)
    raw_json = _strip_fences(raw_json)
    try:
        cfg = json.loads(raw_json)
    except json.JSONDecodeError as exc:
        console.print(f"[red]Failed to parse LLM response as JSON: {exc}[/red]")
        console.print(f"Raw response:\n{raw_json}")
        raise click.Abort()

    # Verify seed URLs against the real web
    candidate_urls: list[str] = cfg.get("seed_urls", [])
    if candidate_urls:
        console.print(f"\n[dim]Verifying {len(candidate_urls)} seed URL(s) against the web…[/dim]")
        reachable, unreachable = _verify_urls(candidate_urls)
        if unreachable:
            console.print(
                f"[yellow]⚠ {len(unreachable)} URL(s) unreachable and removed from seed list:[/yellow]"
            )
            for u in unreachable:
                console.print(f"  [dim red]✗ {u}[/dim red]")
        if reachable:
            console.print(f"[green]✓ {len(reachable)} reachable seed URL(s) kept.[/green]")
        cfg["seed_urls"] = reachable
    else:
        console.print(
            "\n[dim]No seed URLs mentioned — add them later with "
            "[bold]tipster add-url[/bold].[/dim]"
        )

    hint_count = len(cfg.get("relevance_hints", []))
    link_hints = cfg.get("link_score_hints", {})
    pos_count = len(link_hints.get("positive", []))
    neg_count = len(link_hints.get("negative", []))
    console.print(
        f"[green]✓ Config ready:[/green] {len(cfg.get('seed_urls', []))} seed URL(s), "
        f"{hint_count} relevance hints, "
        f"{pos_count} positive / {neg_count} negative link-score hints."
    )

    yaml_text = _build_yaml(cfg, model)

    if yaml_path.exists():
        console.print(f"\n[yellow]⚠ {yaml_path} already exists.[/yellow]")
        if not Confirm.ask(f"Overwrite {yaml_path}?", default=False):
            console.print("Aborted — existing tipster.yaml kept.")
            return None

    yaml_path.write_text(yaml_text)
    console.print(f"\n[green]✓ tipster.yaml written to [bold]{yaml_path}[/bold][/green]")
    console.print(
        f"\n[dim]Edit [bold]{yaml_path}[/bold] to fine-tune, then run:[/dim]\n"
        "  [bold cyan]tipster start[/bold cyan]\n"
        "\n[dim]To add seed URLs manually:[/dim]\n"
        "  [bold cyan]tipster add-url https://example.com[/bold cyan]"
    )
    return cfg


def _strip_fences(text: str) -> str:
    """Remove markdown code fences if present."""
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-z]*\n?", "", text)
        text = re.sub(r"\n?```$", "", text)
    return text.strip()


def _set_env(api_base: str, api_key: str) -> None:
    if api_key:
        os.environ["OPENAI_API_KEY"] = api_key
    if api_base:
        os.environ["OPENAI_API_BASE"] = api_base


def _build_yaml(cfg: dict, model: str) -> str:
    """Build a tipster.yaml string from the LLM-extracted config dict."""
    topic_name = cfg.get("topic_name", "My Topic")
    description = cfg.get("description", "")
    relevance_hints = cfg.get("relevance_hints", [])
    link_hints = cfg.get("link_score_hints", {"positive": [], "negative": []})
    seed_urls = cfg.get("seed_urls", [])
    domain_weights = cfg.get("domain_weights", {})
    report_interval = cfg.get("report_interval", "daily")
    report_time = cfg.get("report_time", "08:00")
    slice_mins = cfg.get("slice_duration_minutes", 60)
    max_tokens = cfg.get("max_tokens_per_slice", 500_000)
    max_cost = cfg.get("max_cost_per_slice_usd", 0.50)

    def _indent_list(items: list[str], indent: int = 4) -> str:
        pad = " " * indent
        return "\n".join(f"{pad}- \"{item}\"" for item in items)

    def _indent_dict(d: dict, indent: int = 4) -> str:
        pad = " " * indent
        return "\n".join(f"{pad}{k}: {v}" for k, v in d.items())

    seed_block = _indent_list(seed_urls, 2)
    rel_hints_block = _indent_list(relevance_hints)
    pos_block = _indent_list(link_hints.get("positive", []), 6)
    neg_block = _indent_list(link_hints.get("negative", []), 6)
    dw_block = "\n".join(f"    {k}: {v}" for k, v in domain_weights.items())

    lines = [
        "# tipster.yaml — generated by `tipster init`",
        "# Edit any field to fine-tune crawler behaviour. Re-run `tipster init` to regenerate.",
        "# Credentials are in .env — never put API keys in this file.",
        "",
        "topic:",
        f'  name: "{topic_name}"',
        "  description: |",
    ]
    for line in description.splitlines():
        lines.append(f"    {line}")
    lines += [
        "",
        "  # Terms injected into relevance triage and link scorer prompts.",
        "  relevance_hints:",
        rel_hints_block,
        "",
        "  # Anchor text patterns used by the link scorer.",
        "  link_score_hints:",
        "    positive:",
        pos_block,
        "    negative:",
        neg_block,
    ]

    lines += [
        "",
        "seed_urls:",
        seed_block if seed_block else "  []",
        "",
        "discovery:",
        "  # Links scoring below this threshold are not fetched (0.0–1.0).",
        "  link_score_threshold: 0.6",
        "",
        "sources:",
        "  # Initial domain weights (0.0–1.0). Adjusted automatically by feedback.",
        "  domain_weights:",
        dw_block if dw_block else "    {}",
        "  blacklist: []",
        "",
        "schedule:",
        f"  slice_duration_minutes: {slice_mins}",
        f'  report_interval: "{report_interval}"',
        f'  report_time: "{report_time}"',
        "",
        "budget:",
        f"  max_tokens_per_slice: {max_tokens}",
        f"  max_cost_per_slice_usd: {max_cost}",
        "",
        "llm:",
        f'  onboard_model: "{model}"',
        f'  triage_model: "{model}"',
        f'  extraction_model: "{model}"',
        f'  link_score_model: "{model}"',
        f'  report_model: "{model}"',
        f'  comment_model: "{model}"',
        "  # API keys are read from .env automatically via python-dotenv.",
        "",
        "crawl:",
        "  default_delay_seconds: 1",
        "  # Advanced: limit concurrent workers to reduce memory / API rate-limit pressure.",
        "  # max_crawl_workers: 10  # max URLs fetched and triaged simultaneously",
        "  # max_llm_workers: 5     # max concurrent LLM calls across all crawl tasks",
    ]

    return "\n".join(lines) + "\n"
