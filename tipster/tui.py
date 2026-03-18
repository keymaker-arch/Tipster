"""Textual TUI dashboard — Phase 1–4.

Keyboard-only conversation interface.

Layout:
  ┌─ header ──────────────────────────────────────────────────────┐
  ├─ STATUS ───────────────────┬─ WORKING LOG ────────────────────┤
  │ Topic / URLs / Cost / etc  │ Brief live activity events       │
  │ Schedule / Directives      │ (proof the system is not stuck)  │
  ├────────────────────────────┴──────────────────────────────────┤
  │                                                               │
  │  MAIN  (reports rendered here; future: LLM chat responses)   │
  │                                                               │
  ├───────────────────────────────────────────────────────────────┤
  │ ▶  [input — always focused]                                   │
  └───────────────────────────────────────────────────────────────┘
"""

from __future__ import annotations

import asyncio
import re
from datetime import datetime, timezone
from typing import Optional

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal
from textual.widgets import Footer, Header, Input, Label, RichLog, Static

from tipster.config import TipsterConfig
from tipster.db.repositories.content_items import ContentItemRepo
from tipster.db.repositories.directives import DirectiveRepo
from tipster.db.repositories.url_registry import UrlRegistryRepo
from tipster.db.session import get_db
from tipster.events import Event, EventBus, EventKind
from tipster.scheduler import CrawlScheduler, CrawlStats

# ---------------------------------------------------------------------------
# Brief icon + style for working log
# ---------------------------------------------------------------------------

_KIND_BRIEF: dict[EventKind, tuple[str, str]] = {
    EventKind.CRAWL_START:        ("dim cyan",    "→"),
    EventKind.CRAWL_OK:           ("green",       "✓"),
    EventKind.CRAWL_SKIP:         ("dim yellow",  "="),
    EventKind.CRAWL_DUPLICATE:    ("dim yellow",  "≡"),
    EventKind.CRAWL_ERROR:        ("red",         "✗"),
    EventKind.TRIAGE_RELEVANT:    ("bold green",  "★"),
    EventKind.TRIAGE_IRRELEVANT:  ("dim",         "·"),
    EventKind.EXTRACT_START:      ("dim cyan",    "↳"),
    EventKind.EXTRACT_OK:         ("cyan",        "✓"),
    EventKind.EXTRACT_DEFERRED:   ("yellow",      "⏸"),
    EventKind.EXTRACT_ERROR:      ("red",         "✗"),
    EventKind.LINK_DISCOVERED:    ("blue",        "+"),
    EventKind.LINK_DEFERRED:      ("dim yellow",  "~"),
    EventKind.SCHEDULER_TICK:     ("dim blue",    "●"),
    EventKind.STATS_UPDATE:       ("dim",         "·"),
    EventKind.REPORT_READY:       ("bold magenta","★"),
    EventKind.DIRECTIVE_APPLIED:  ("bold yellow", "⚡"),
}


def _domain(url: str) -> str:
    """Quick domain extraction for brief display."""
    s = url.split("://", 1)[-1]
    return s.split("/")[0][:38]


def _brief_line(event: Event) -> str:
    ts = event.ts.strftime("%H:%M:%S")
    style, icon = _KIND_BRIEF.get(event.kind, ("dim", "·"))
    if event.url:
        desc = _domain(event.url)
    else:
        desc = (event.message or event.kind.value)[:45]
    return f"[dim]{ts}[/dim] [{style}]{icon}[/{style}] {desc}"


def _fmt_ago(dt: Optional[datetime]) -> str:
    if dt is None:
        return "—"
    diff = (datetime.now(timezone.utc) - dt).total_seconds()
    if diff < 60:
        return f"{int(diff)}s ago"
    if diff < 3600:
        return f"{int(diff / 60)}m ago"
    return f"{int(diff / 3600)}h ago"


def _fmt_in(dt: Optional[datetime]) -> str:
    if dt is None:
        return "—"
    diff = (dt - datetime.now(timezone.utc)).total_seconds()
    if diff <= 0:
        return "now"
    if diff < 60:
        return f"in {int(diff)}s"
    if diff < 3600:
        return f"in {int(diff / 60)}m"
    return f"in {int(diff / 3600)}h"


def _fmt_uptime(start: datetime) -> str:
    diff = int((datetime.now(timezone.utc) - start).total_seconds())
    h, rem = divmod(diff, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


# ---------------------------------------------------------------------------
# Top-left: Status panel
# ---------------------------------------------------------------------------

class StatusPanel(Static):
    """Combined topic overview + crawl schedule."""

    def compose(self) -> ComposeResult:
        yield Label("STATUS", id="st-title")
        yield Label("", id="st-topic")
        yield Label("", id="st-running")
        yield Label("", id="st-urls")
        yield Label("", id="st-extracted")
        yield Label("", id="st-cost")
        yield Label("", id="st-directives")
        yield Label("", id="st-sep")
        yield Label("", id="st-last-crawl")
        yield Label("", id="st-next-crawl")
        yield Label("", id="st-report-interval")
        yield Label("", id="st-last-report")

    def refresh_all(
        self,
        topic_name: str,
        running: bool,
        urls: int,
        extracted: int,
        pending: int,
        cost: float,
        cost_limit: float,
        directives: int,
        last_crawl: Optional[datetime],
        next_crawl: Optional[datetime],
        report_interval: str,
        last_report_at: Optional[datetime],
    ) -> None:
        status = "[green]● running[/green]" if running else "[dim]● idle[/dim]"
        q = self.query_one
        q("#st-topic",           Label).update(f"[dim]Topic[/dim]        {topic_name}")
        q("#st-running",         Label).update(f"[dim]Status[/dim]       {status}")
        q("#st-urls",            Label).update(f"[dim]URLs known[/dim]   {urls}")
        q("#st-extracted",       Label).update(
            f"[dim]Extracted[/dim]    {extracted}  [dim]pending {pending}[/dim]"
        )
        q("#st-cost",            Label).update(
            f"[dim]Cost[/dim]         ${cost:.4f} / ${cost_limit:.2f}"
        )
        q("#st-directives",      Label).update(f"[dim]Directives[/dim]   {directives}")
        q("#st-sep",             Label).update("")
        q("#st-last-crawl",      Label).update(f"[dim]Last crawl[/dim]   {_fmt_ago(last_crawl)}")
        q("#st-next-crawl",      Label).update(f"[dim]Next crawl[/dim]   {_fmt_in(next_crawl)}")
        q("#st-report-interval", Label).update(f"[dim]Report[/dim]       {report_interval}")
        q("#st-last-report",     Label).update(f"[dim]Last report[/dim]  {_fmt_ago(last_report_at)}")


# ---------------------------------------------------------------------------
# Top-right: Working log (brief liveness indicator)
# ---------------------------------------------------------------------------

class WorkingLog(Static):
    """Brief live activity events — proof the system is not stuck."""

    def compose(self) -> ComposeResult:
        yield Label("WORKING LOG", id="wl-title")
        yield RichLog(id="wl-log", highlight=True, markup=True, wrap=False)


# ---------------------------------------------------------------------------
# Main App
# ---------------------------------------------------------------------------

class TipsterApp(App):
    """Tipster TUI — keyboard-only conversation interface."""

    TITLE = "Tipster"
    ENABLE_COMMAND_PALETTE = False

    CSS = """
    Screen {
        background: $background;
    }

    /* ── top two-panel row ── */
    #top-panels {
        height: 14;
        min-height: 12;
    }
    StatusPanel {
        width: 1fr;
        height: 100%;
        border: solid $primary-darken-2;
        padding: 0 1;
    }
    WorkingLog {
        width: 1fr;
        height: 100%;
        border: solid $primary-darken-2;
        padding: 0 1;
    }
    #st-title, #wl-title {
        text-style: bold;
        color: $accent;
    }
    #wl-log {
        height: 1fr;
    }

    /* ── main window ── */
    #main-log {
        height: 1fr;
        border: solid $primary-darken-2;
        padding: 0 1;
    }

    /* ── input bar ── */
    #input-bar {
        height: 3;
        border: solid $accent;
        padding: 0 1;
        align: left middle;
    }
    #input-prompt {
        width: 3;
        color: $accent;
        text-style: bold;
        content-align: left middle;
    }
    #conv-input {
        width: 1fr;
        border: none;
        background: transparent;
    }
    """

    BINDINGS = [
        Binding("q", "quit", "Quit"),
        Binding("escape", "focus_input", "Focus input", show=False),
    ]

    def __init__(
        self,
        cfg: TipsterConfig,
        topic_id: int,
        topic_name: str,
        bus: EventBus,
        stats: CrawlStats,
        scheduler: CrawlScheduler,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self._cfg = cfg
        self._topic_id = topic_id
        self._topic_name = topic_name
        self._bus = bus
        self._stats = stats
        self._scheduler = scheduler
        self._app_start_time = datetime.now(timezone.utc)
        self._current_report: Optional[dict] = None

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        with Horizontal(id="top-panels"):
            yield StatusPanel(id="status")
            yield WorkingLog(id="worklog")
        yield RichLog(id="main-log", highlight=True, markup=True, wrap=True)
        with Horizontal(id="input-bar"):
            yield Label("▶", id="input-prompt")
            yield Input(
                id="conv-input",
                placeholder="type a command  (help — list all commands)",
            )
        yield Footer()

    def on_mount(self) -> None:
        self.title = f"Tipster  ●  {self._topic_name}"
        self.sub_title = "starting…"
        self.run_worker(self._run_scheduler(), exclusive=False)
        self.run_worker(self._consume_events(), exclusive=False)
        self.run_worker(self._refresh_stats(), exclusive=False)
        self.set_interval(1, self._tick_uptime)
        self.query_one("#conv-input", Input).focus()

    def _tick_uptime(self) -> None:
        self.sub_title = f"uptime {_fmt_uptime(self._app_start_time)}"

    # ------------------------------------------------------------------
    # Background workers
    # ------------------------------------------------------------------

    async def _run_scheduler(self) -> None:
        await self._scheduler.run()

    async def _consume_events(self) -> None:
        main_log: RichLog = self.query_one("#main-log", RichLog)
        work_log: RichLog = self.query_one("#wl-log", RichLog)

        while True:
            event: Event = await self._bus.receive()

            if event.kind == EventKind.REPORT_READY and event.data:
                # Brief note in working log
                work_log.write(_brief_line(event))

                # Full report rendered in main window
                self._current_report = event.data
                narrative_md = event.data.get("narrative_md", "")
                items = event.data.get("items", [])
                sep = "━" * 62
                main_log.write("")
                main_log.write(f"[bold magenta]{sep}[/bold magenta]")
                main_log.write(
                    f"[bold magenta]  REPORT READY  —  {len(items)} item(s)[/bold magenta]"
                )
                main_log.write(f"[bold magenta]{sep}[/bold magenta]")
                for line in narrative_md.split("\n"):
                    main_log.write(line)
                main_log.write("")
                main_log.write("[bold]── Items in this report ──[/bold]")
                for i, item in enumerate(items, 1):
                    star = "[bold yellow]★[/bold yellow]" if item.get("is_new_source") else " "
                    url_short = item["url"][:68]
                    main_log.write(
                        f"  [bold]{i:>2}.[/bold] {star} [dim]{url_short}[/dim]"
                        f"  [dim]score={item['score']:.2f}[/dim]"
                    )
                main_log.write("")
                main_log.write(
                    "[dim]Feedback: [bold]<N>j[/bold] interesting  "
                    "[bold]<N>n[/bold] not interesting  "
                    "[bold]<N>c <text>[/bold] comment  "
                    "— type [bold]help[/bold] for all commands[/dim]"
                )
                main_log.write(f"[bold magenta]{sep}[/bold magenta]")
                continue

            # All other events go to the working log (brief format)
            work_log.write(_brief_line(event))

    async def _refresh_stats(self) -> None:
        st: StatusPanel = self.query_one("#status", StatusPanel)

        while True:
            try:
                db = get_db()
                url_count = UrlRegistryRepo(db).count_by_topic(self._topic_id)
                item_repo = ContentItemRepo(db)
                extracted = (
                    item_repo.count_by_topic(self._topic_id)
                    - item_repo.count_pending(self._topic_id)
                )
                pending = item_repo.count_pending(self._topic_id)
                directive_count = DirectiveRepo(db).count_active(self._topic_id)
                from tipster.db.repositories.reports import ReportRepo
                last_report = ReportRepo(db).get_last(self._topic_id)
                last_report_at = last_report.generated_at if last_report else None
                if last_report_at and last_report_at.tzinfo is None:
                    from datetime import timezone as _tz
                    last_report_at = last_report_at.replace(tzinfo=_tz.utc)
                db.close()
            except Exception:
                url_count = extracted = pending = directive_count = 0
                last_report_at = None

            st.refresh_all(
                topic_name=self._topic_name,
                running=self._stats.running,
                urls=url_count,
                extracted=extracted,
                pending=pending,
                cost=self._stats.cost_usd,
                cost_limit=self._cfg.budget.max_cost_per_slice_usd,
                directives=directive_count,
                last_crawl=self._stats.last_crawl_at,
                next_crawl=self._stats.next_crawl_at,
                report_interval=self._cfg.schedule.report_interval,
                last_report_at=last_report_at,
            )
            await asyncio.sleep(3)

    # ------------------------------------------------------------------
    # Actions
    # ------------------------------------------------------------------

    def action_focus_input(self) -> None:
        self.query_one("#conv-input", Input).focus()

    def action_quit(self) -> None:
        self._scheduler.stop()
        self.exit()

    # ------------------------------------------------------------------
    # Conversation input
    # ------------------------------------------------------------------

    async def on_input_submitted(self, event: Input.Submitted) -> None:
        text = event.value.strip()
        event.input.clear()
        if text:
            await self._handle_command(text)
        event.input.focus()

    async def _handle_command(self, text: str) -> None:
        """Route a conversation command and write the response to the main log."""
        log: RichLog = self.query_one("#main-log", RichLog)

        # Echo the input so it appears in scroll history
        log.write(f"[bold cyan]▶ {text}[/bold cyan]")

        cmd = text.lower().strip()

        if cmd in ("help", "?", "h"):
            self._cmd_help(log)
            return

        if cmd in ("report", "r"):
            await self._cmd_report(log)
            return

        # Feedback: <N>j / <N>n / <N>c <text>
        m = re.match(r"^(\d+)\s*([jJnNcC])(.*)?$", text)
        if m:
            await self._handle_feedback(m, log)
            return

        log.write(
            "[red]Unknown command.[/red] "
            "Type [bold]help[/bold] for available commands."
        )

    def _cmd_help(self, log: RichLog) -> None:
        log.write("[bold]Available commands:[/bold]")
        log.write("  [bold cyan]<N>j[/bold cyan]            mark item N as interesting (+)")
        log.write("  [bold cyan]<N>n[/bold cyan]            mark item N as not interesting (−)")
        log.write("  [bold cyan]<N>c <text>[/bold cyan]     comment on item N — LLM infers directives")
        log.write("  [bold cyan]report[/bold cyan]  [dim]r[/dim]     trigger report generation now")
        log.write("  [bold cyan]help[/bold cyan]    [dim]?[/dim]     show this help")
        log.write("  [bold cyan]q[/bold cyan]               quit Tipster")

    async def _cmd_report(self, log: RichLog) -> None:
        from tipster.reporter import generate_report
        log.write("[dim]Generating report…[/dim]")
        result = await generate_report(self._topic_id, self._cfg, self._bus)
        if result is None:
            log.write("[yellow]No unreported items to report yet.[/yellow]")

    async def _handle_feedback(self, m: re.Match, log: RichLog) -> None:
        from tipster.feedback import process_comment, process_judgement

        if self._current_report is None:
            log.write(
                "[yellow]No active report — wait for a report to arrive "
                "or type [bold]report[/bold] to generate one now.[/yellow]"
            )
            return

        idx = int(m.group(1)) - 1
        action = m.group(2).lower()
        extra = (m.group(3) or "").strip()
        items = self._current_report.get("items", [])

        if idx < 0 or idx >= len(items):
            log.write(
                f"[red]No item #{idx + 1} in the current report "
                f"({len(items)} item(s)).[/red]"
            )
            return

        item = items[idx]
        item_id = item["item_id"]
        url_id = item["url_id"]
        domain = item.get("domain", "")
        snippet = item.get("summary", "")[:200]

        loop = asyncio.get_event_loop()

        if action == "j":
            await loop.run_in_executor(
                None,
                lambda: process_judgement(
                    self._topic_id, item_id, url_id, +1, snippet, domain
                ),
            )
            log.write(f"[green]✓ Item #{idx + 1} marked interesting.[/green]")

        elif action == "n":
            await loop.run_in_executor(
                None,
                lambda: process_judgement(
                    self._topic_id, item_id, url_id, -1, snippet, domain
                ),
            )
            log.write(f"[yellow]✓ Item #{idx + 1} marked not interesting.[/yellow]")

        elif action == "c":
            if not extra:
                log.write("[red]Usage: <N>c <your comment text>[/red]")
                return
            log.write(f"[dim]Interpreting comment for item #{idx + 1}…[/dim]")
            directives = await process_comment(
                self._topic_id, item_id, url_id, extra, self._cfg, self._bus
            )
            if directives:
                log.write(
                    f"[bold yellow]✓ {len(directives)} directive(s) created: "
                    f"{', '.join(directives)}[/bold yellow]"
                )
            else:
                log.write("[dim]Comment saved. No directives inferred.[/dim]")
