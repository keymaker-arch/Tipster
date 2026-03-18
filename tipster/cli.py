"""Tipster CLI — entry point for all commands."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse

import click
from dotenv import load_dotenv
from rich.console import Console
from rich.table import Table

console = Console()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_env_and_db(
    work_dir: Path,
    db_path: Path,
) -> tuple:
    """Load .env, initialise DB, return (config, db_session) or abort."""
    from tipster.db.session import init_db, get_db

    env_file = work_dir / ".env"
    if env_file.exists():
        load_dotenv(env_file)

    init_db(str(db_path))
    db = get_db()
    return db


def _resolve_paths(work_dir_str: str) -> tuple[Path, Path, Path]:
    """Return (work_dir, yaml_path, db_path)."""
    work_dir = Path(work_dir_str).resolve()
    yaml_path = work_dir / "tipster.yaml"
    db_path = work_dir / "tipster.db"
    return work_dir, yaml_path, db_path


# ---------------------------------------------------------------------------
# CLI group
# ---------------------------------------------------------------------------

@click.group()
@click.version_option(package_name="tipster")
def cli() -> None:
    """Tipster — LLM-powered autonomous web intelligence crawler."""


# ---------------------------------------------------------------------------
# tipster init
# ---------------------------------------------------------------------------

@cli.command("init")
@click.option(
    "--dir", "work_dir", default=".", show_default=True,
    help="Working directory where tipster.yaml and .env will be written.",
)
@click.option(
    "--force-provider", is_flag=True, default=False,
    help="Re-run provider setup even if .env already exists.",
)
def cmd_init(work_dir: str, force_provider: bool) -> None:
    """Initialise Tipster: set up LLM provider and generate tipster.yaml."""
    from tipster.onboarding import step0_provider_setup, step1_generate_yaml
    from tipster.db.session import init_db, get_db
    from tipster.db.repositories.directives import DirectiveRepo

    wdir, yaml_path, db_path = _resolve_paths(work_dir)
    env_path = wdir / ".env"

    console.print(
        f"\n[bold magenta]Tipster Init[/bold magenta] — working directory: [cyan]{wdir}[/cyan]\n"
    )

    # Check for existing directives (warn before overwriting yaml)
    try:
        init_db(str(db_path))
        db = get_db()
        from tipster.db.repositories.topics import TopicRepo
        topic = TopicRepo(db).get_active()
        if topic:
            n = DirectiveRepo(db).count_active(topic.topic_id)
            if n > 0:
                console.print(
                    f"[yellow]⚠ You have {n} active directive(s) in the database.[/yellow]\n"
                    "  Run [bold]tipster export[/bold] to merge them into the new config,\n"
                    "  or they will remain active independently."
                )
        db.close()
    except Exception:
        pass  # DB might not exist yet — fine

    # Step 0 — LLM provider
    creds = step0_provider_setup(env_path, force=force_provider)

    # Step 1 — Topic → tipster.yaml
    cfg = step1_generate_yaml(
        api_base=creds["api_base"],
        api_key=creds["api_key"],
        model=creds["model"],
        yaml_path=yaml_path,
        db_path=db_path,
    )

    if cfg is None:
        return  # user aborted the yaml overwrite

    # Persist topic + seed URLs to the DB so `tipster status` works immediately
    from tipster.db.session import init_db, get_db
    from tipster.db.repositories.topics import TopicRepo
    from tipster.db.repositories.url_registry import UrlRegistryRepo

    init_db(str(db_path))
    db = get_db()
    topic_repo = TopicRepo(db)

    # Deactivate any previous active topic (re-init replaces it)
    existing = topic_repo.get_active()
    if existing:
        existing.is_active = False
        db.commit()

    topic = topic_repo.create(
        name=cfg.get("topic_name", "My Topic"),
        description=cfg.get("description", ""),
    )

    url_repo = UrlRegistryRepo(db)
    seeded = 0
    for url in cfg.get("seed_urls", []):
        url_repo.add(topic.topic_id, url, added_by="seed")
        seeded += 1

    db.close()

    if seeded:
        console.print(f"[green]✓ {seeded} seed URL(s) added to the registry.[/green]")

    console.print("\n[bold green]Onboarding complete![/bold green]")


# ---------------------------------------------------------------------------
# tipster add-url
# ---------------------------------------------------------------------------

@cli.command("add-url")
@click.argument("url")
@click.option(
    "--dir", "work_dir", default=".", show_default=True,
    help="Working directory containing tipster.yaml and tipster.db.",
)
def cmd_add_url(url: str, work_dir: str) -> None:
    """Add a URL to the crawl registry manually."""
    from tipster.db.session import init_db, get_db
    from tipster.db.repositories.url_registry import UrlRegistryRepo
    from tipster.db.repositories.topics import TopicRepo

    wdir, yaml_path, db_path = _resolve_paths(work_dir)

    # Basic URL validation
    parsed = urlparse(url)
    if not parsed.scheme or not parsed.netloc:
        console.print(f"[red]Invalid URL:[/red] {url}")
        sys.exit(1)

    # Load .env
    env_file = wdir / ".env"
    if env_file.exists():
        load_dotenv(env_file)

    # Initialise DB (creates tables if not present)
    init_db(str(db_path))
    db = get_db()

    # Ensure a topic exists (use yaml if available, else a placeholder)
    topic_repo = TopicRepo(db)
    topic = topic_repo.get_active()

    if topic is None:
        # Try to load name from yaml, else use placeholder
        name = "Default"
        if yaml_path.exists():
            try:
                from tipster.config import TipsterConfig
                cfg = TipsterConfig.from_yaml(yaml_path)
                name = cfg.topic.name
            except Exception:
                pass
        topic = topic_repo.create(name=name)
        console.print(f"[dim]Created topic '{topic.name}' (topic_id={topic.topic_id})[/dim]")

    url_repo = UrlRegistryRepo(db)
    existing = url_repo.get_by_url(url)
    if existing:
        console.print(
            f"[yellow]URL already in registry:[/yellow] {url} "
            f"(url_id={existing.url_id}, status={existing.status})"
        )
        db.close()
        return

    entry = url_repo.add(
        topic_id=topic.topic_id,
        url=url,
        added_by="manual",
    )
    console.print(
        f"[green]✓ Added:[/green] {url} "
        f"[dim](url_id={entry.url_id}, domain={entry.domain})[/dim]"
    )
    db.close()


# ---------------------------------------------------------------------------
# tipster status  (convenience command — shows DB summary)
# ---------------------------------------------------------------------------

@cli.command("status")
@click.option("--dir", "work_dir", default=".", show_default=True)
def cmd_status(work_dir: str) -> None:
    """Show a summary of the current Tipster state."""
    from tipster.db.session import init_db, get_db
    from tipster.db.repositories.topics import TopicRepo
    from tipster.db.repositories.url_registry import UrlRegistryRepo
    from tipster.db.repositories.content_items import ContentItemRepo
    from tipster.db.repositories.directives import DirectiveRepo

    wdir, yaml_path, db_path = _resolve_paths(work_dir)

    if not db_path.exists():
        console.print("[yellow]No tipster.db found. Run `tipster init` first.[/yellow]")
        return

    env_file = wdir / ".env"
    if env_file.exists():
        load_dotenv(env_file)

    init_db(str(db_path))
    db = get_db()

    topic = TopicRepo(db).get_active()
    if topic is None:
        console.print("[yellow]No active topic. Run `tipster init` first.[/yellow]")
        db.close()
        return

    url_count = UrlRegistryRepo(db).count_by_topic(topic.topic_id)
    item_count = ContentItemRepo(db).count_by_topic(topic.topic_id)
    pending_count = ContentItemRepo(db).count_pending(topic.topic_id)
    directive_count = DirectiveRepo(db).count_active(topic.topic_id)

    table = Table(title=f"Tipster Status — {topic.name}", show_header=False, box=None)
    table.add_column("Key", style="dim")
    table.add_column("Value", style="bold")
    table.add_row("Topic", topic.name)
    table.add_row("Topic ID", str(topic.topic_id))
    table.add_row("DB", str(db_path))
    table.add_row("URLs known", str(url_count))
    table.add_row("Content items", str(item_count))
    table.add_row("Pending extraction", str(pending_count))
    table.add_row("Active directives", str(directive_count))

    console.print(table)
    db.close()


# ---------------------------------------------------------------------------
# tipster start
# ---------------------------------------------------------------------------

@cli.command("start")
@click.option("--dir", "work_dir", default=".", show_default=True)
def cmd_start(work_dir: str) -> None:
    """Start the Tipster crawler service with TUI dashboard."""
    from tipster.config import load_config
    from tipster.db.session import init_db, get_db
    from tipster.db.repositories.topics import TopicRepo
    from tipster.db.repositories.url_registry import UrlRegistryRepo
    from tipster.events import EventBus
    from tipster.scheduler import CrawlScheduler, CrawlStats
    from tipster.tui import TipsterApp

    wdir, yaml_path, db_path = _resolve_paths(work_dir)

    if not yaml_path.exists():
        console.print("[red]tipster.yaml not found. Run `tipster init` first.[/red]")
        sys.exit(1)

    env_file = wdir / ".env"
    if env_file.exists():
        load_dotenv(env_file)

    cfg = load_config(yaml_path, env_file)
    init_db(str(db_path))
    db = get_db()

    # Resolve or create topic
    topic_repo = TopicRepo(db)
    topic = topic_repo.get_active()
    if topic is None:
        topic = topic_repo.create(cfg.topic.name, cfg.topic.description)
        console.print(f"[dim]Created topic '{topic.name}'[/dim]")

    # Fresh-start vs resume: seed URL Registry from yaml if no URLs exist yet
    url_repo = UrlRegistryRepo(db)
    existing_count = url_repo.count_by_topic(topic.topic_id)
    if existing_count == 0 and cfg.seed_urls:
        for seed in cfg.seed_urls:
            url_repo.add(topic.topic_id, seed.url, added_by="seed", prompt_snippet=seed.prompt)
        console.print(f"[green]✓ Seeded {len(cfg.seed_urls)} URL(s) from tipster.yaml[/green]")
    elif existing_count > 0:
        console.print(
            f"[dim]Resuming — {existing_count} URL(s) already in registry[/dim]"
        )

    # Capture primitive values before closing session
    topic_id = topic.topic_id
    topic_name = topic.name
    db.close()

    bus = EventBus()
    stats = CrawlStats()
    scheduler = CrawlScheduler(
        cfg=cfg,
        topic_id=topic_id,
        bus=bus,
        stats=stats,
        db_path=str(db_path),
    )

    app = TipsterApp(
        cfg=cfg,
        topic_id=topic_id,
        topic_name=topic_name,
        bus=bus,
        stats=stats,
        scheduler=scheduler,
    )
    app.run()


# ---------------------------------------------------------------------------
# tipster export
# ---------------------------------------------------------------------------

@cli.command("export")
@click.option("--dir", "work_dir", default=".", show_default=True)
@click.option(
    "--output", "-o", default=None,
    help="Output YAML path (default: overwrites tipster.yaml).",
)
def cmd_export(work_dir: str, output: Optional[str]) -> None:
    """Merge active directives from directive_store into tipster.yaml."""
    import json as _json
    from tipster.config import load_config
    from tipster.db.session import init_db, get_db
    from tipster.db.repositories.topics import TopicRepo
    from tipster.db.repositories.directives import DirectiveRepo
    import yaml as _yaml

    wdir, yaml_path, db_path = _resolve_paths(work_dir)

    if not yaml_path.exists():
        console.print("[red]tipster.yaml not found.[/red]")
        sys.exit(1)
    if not db_path.exists():
        console.print("[red]tipster.db not found.[/red]")
        sys.exit(1)

    env_file = wdir / ".env"
    if env_file.exists():
        load_dotenv(env_file)

    init_db(str(db_path))
    db = get_db()

    topic = TopicRepo(db).get_active()
    if topic is None:
        console.print("[yellow]No active topic.[/yellow]")
        db.close()
        return

    directives = DirectiveRepo(db).list_active(topic.topic_id)
    db.close()

    # Load existing YAML as raw dict
    with open(yaml_path) as f:
        raw = _yaml.safe_load(f) or {}

    # Apply directives to raw config dict
    for d in directives:
        params: dict = {}
        if d.params_json:
            try:
                params = _json.loads(d.params_json)
            except Exception:
                pass
        target = d.target or ""

        if d.directive_type == "BLACKLIST_SOURCE" and target:
            bl = raw.setdefault("sources", {}).setdefault("blacklist", [])
            if target not in bl:
                bl.append(target)

        elif d.directive_type == "UPDATE_LINK_SCORE_HINT" and target:
            polarity = params.get("polarity", "positive")
            hint = params.get("hint", "")
            if hint:
                hints = raw.setdefault("topic", {}).setdefault("link_score_hints", {})
                lst = hints.setdefault(polarity, [])
                if hint not in lst:
                    lst.append(hint)

        elif d.directive_type == "EXPAND_TOPIC" and target:
            rh = raw.setdefault("topic", {}).setdefault("relevance_hints", [])
            if target not in rh:
                rh.append(target)

        elif d.directive_type == "BOOST_CRAWL_PRIORITY" and target:
            dw = raw.setdefault("sources", {}).setdefault("domain_weights", {})
            magnitude = float(params.get("magnitude", 0.5))
            existing = dw.get(target, 0.5)
            dw[target] = min(1.0, existing + (1.0 - existing) * (1.0 - magnitude))

    out_path = Path(output) if output else yaml_path
    with open(out_path, "w") as f:
        _yaml.dump(raw, f, allow_unicode=True, default_flow_style=False, sort_keys=False)

    console.print(
        f"[green]✓ Exported {len(directives)} directive(s) → [bold]{out_path}[/bold][/green]"
    )


# ---------------------------------------------------------------------------
# tipster report  (manual report trigger)
# ---------------------------------------------------------------------------

@cli.command("report")
@click.option("--dir", "work_dir", default=".", show_default=True)
def cmd_report(work_dir: str) -> None:
    """Manually trigger report generation and print it to the console."""
    import asyncio as _asyncio
    from tipster.config import load_config
    from tipster.db.session import init_db
    from tipster.db.repositories.topics import TopicRepo
    from tipster.events import EventBus

    wdir, yaml_path, db_path = _resolve_paths(work_dir)

    if not yaml_path.exists():
        console.print("[red]tipster.yaml not found.[/red]")
        sys.exit(1)

    env_file = wdir / ".env"
    if env_file.exists():
        load_dotenv(env_file)

    cfg = load_config(yaml_path, env_file)
    init_db(str(db_path))
    db = _load_env_and_db(wdir, db_path)
    topic = TopicRepo(db).get_active()
    if topic is None:
        console.print("[yellow]No active topic.[/yellow]")
        db.close()
        return
    topic_id = topic.topic_id
    db.close()

    bus = EventBus()

    async def _run():
        from tipster.reporter import generate_report
        result = await generate_report(topic_id, cfg, bus)
        return result

    result = _asyncio.run(_run())
    if result is None:
        console.print("[yellow]No unreported items to report.[/yellow]")
        return

    from rich.markdown import Markdown
    console.print(Markdown(result["narrative_md"]))
    console.print(
        f"\n[dim]Report #{result['report_id']} saved — {len(result['item_ids'])} item(s) marked reported.[/dim]"
    )
