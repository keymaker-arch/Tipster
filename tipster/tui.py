"""Textual TUI dashboard — Phase 1–4 (finding-by-finding viewer).

Keyboard-only conversation interface.

Layout:
  ┌─ header ──────────────────────────────────────────────────────┐
  ├─ STATUS ───────────────────┬─ WORKING LOG ────────────────────┤
  │ Topic / URLs / Cost / etc  │ Brief live activity events       │
  ├────────────────────────────┴──────────────────────────────────┤
  │                                                               │
  │  FINDING VIEWER  (one finding at a time, markdown rendered,   │
  │                   scrollable up/down via PgUp/PgDn or mouse)  │
  │                                                               │
  ├───────────────────────────────────────────────────────────────┤
  │ HISTORY  [1] ★Title · [2] Title · …                          │
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
from textual.containers import Horizontal, VerticalScroll
from textual.widgets import Footer, Header, Input, Label, Markdown, RichLog, Static

from tipster.config import TipsterConfig
from tipster.db.repositories.content_items import ContentItemRepo
from tipster.db.repositories.directives import DirectiveRepo
from tipster.db.repositories.url_registry import UrlRegistryRepo
from tipster.db.session import get_db
from tipster.events import Event, EventBus, EventKind
from tipster.reporter import _render_item
from tipster.scheduler import CrawlScheduler, CrawlStats

# ---------------------------------------------------------------------------
# Brief icon + style for working log
# ---------------------------------------------------------------------------

_KIND_BRIEF: dict[EventKind, tuple[str, str]] = {
    EventKind.CRAWL_START:        ("dim cyan",     "→"),
    EventKind.CRAWL_OK:           ("green",        "✓"),
    EventKind.CRAWL_SKIP:         ("dim yellow",   "="),
    EventKind.CRAWL_DUPLICATE:    ("dim yellow",   "≡"),
    EventKind.CRAWL_ERROR:        ("red",          "✗"),
    EventKind.TRIAGE_RELEVANT:    ("bold green",   "★"),
    EventKind.TRIAGE_IRRELEVANT:  ("dim",          "·"),
    EventKind.EXTRACT_START:      ("dim cyan",     "↳"),
    EventKind.EXTRACT_OK:         ("cyan",         "✓"),
    EventKind.EXTRACT_DEFERRED:   ("yellow",       "⏸"),
    EventKind.EXTRACT_ERROR:      ("red",          "✗"),
    EventKind.LINK_DISCOVERED:    ("blue",         "+"),
    EventKind.LINK_DEFERRED:      ("dim yellow",   "~"),
    EventKind.SCHEDULER_TICK:     ("dim blue",     "●"),
    EventKind.STATS_UPDATE:       ("dim",          "·"),
    EventKind.REPORT_READY:       ("bold magenta", "★"),
    EventKind.DIRECTIVE_APPLIED:  ("bold yellow",  "⚡"),
}


def _domain(url: str) -> str:
    s = url.split("://", 1)[-1]
    return s.split("/")[0][:38]


_SHOW_REASON: frozenset[EventKind] = frozenset({
    EventKind.CRAWL_ERROR,
    EventKind.CRAWL_SKIP,
    EventKind.CRAWL_DUPLICATE,
    EventKind.EXTRACT_ERROR,
    EventKind.EXTRACT_DEFERRED,
})


def _brief_line(event: Event) -> str:
    ts = event.ts.strftime("%H:%M:%S")
    style, icon = _KIND_BRIEF.get(event.kind, ("dim", "·"))
    if event.url:
        desc = _domain(event.url)
        if event.kind in _SHOW_REASON and event.message:
            desc += f"  [dim]{event.message[:60]}[/dim]"
    else:
        desc = (event.message or event.kind.value)[:60]
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
# Main: Finding viewer (one finding at a time, markdown rendered)
# ---------------------------------------------------------------------------

class FindingViewer(Static):
    """Displays one finding at a time with rendered markdown, scrollable vertically.

    The widget border carries the counter (border_title) and keyboard hints
    (border_subtitle) — no inner header/footer boxes needed.
    """

    def compose(self) -> ComposeResult:
        with VerticalScroll(id="fv-scroll"):
            yield Markdown("", id="fv-content")

    def show_idle(self, message: str = "Waiting for findings…") -> None:
        self.border_title = ""
        self.border_subtitle = ""
        self.query_one("#fv-content", Markdown).update(f"*{message}*")
        self.query_one("#fv-scroll", VerticalScroll).scroll_home(animate=False)

    def show_finding(self, item: dict, position: int, total: int) -> None:
        star = "★ NEW SOURCE  " if item.get("is_new_source") else ""
        domain = item.get("domain") or item.get("url", "")
        score = item.get("score", 0.0)
        self.border_title = f" Finding {position}/{total}  {star}{domain} · {score:.2f} "
        self.border_subtitle = " j=interesting  n=not  c <text>=comment  skip  PgUp/PgDn=scroll  history "
        self.query_one("#fv-content", Markdown).update(_render_item(item))
        self.query_one("#fv-scroll", VerticalScroll).scroll_home(animate=False)

    def show_markdown(self, md: str, header: str = "") -> None:
        """Show arbitrary markdown content (help, history, messages, etc.)."""
        self.border_title = f" {header} " if header else ""
        self.border_subtitle = ""
        self.query_one("#fv-content", Markdown).update(md)
        self.query_one("#fv-scroll", VerticalScroll).scroll_home(animate=False)

    def scroll_up(self) -> None:
        self.query_one("#fv-scroll", VerticalScroll).scroll_page_up(animate=True)

    def scroll_down(self) -> None:
        self.query_one("#fv-scroll", VerticalScroll).scroll_page_down(animate=True)


# ---------------------------------------------------------------------------
# History bar (compact strip below the finding viewer)
# ---------------------------------------------------------------------------

class HistoryBar(Static):
    """One-line strip showing a compact summary of reviewed intel items."""

    def compose(self) -> ComposeResult:
        yield Label("[dim]HISTORY  —  no items reviewed yet[/dim]", id="hist-line")
        # no border — rendered as a flush colored strip between viewer and input

    def refresh_history(self, history: list[dict]) -> None:
        if not history:
            self.query_one("#hist-line", Label).update(
                "[dim]HISTORY  —  no items reviewed yet[/dim]"
            )
            return

        _FB_ICON = {"j": "[green]✓[/green]", "n": "[red]✗[/red]", "c": "[yellow]💬[/yellow]"}
        parts = ["[dim bold]HISTORY[/dim bold]"]
        for i, h in enumerate(history, 1):
            star = "★" if h.get("is_new_source") else ""
            title = (h.get("title") or h.get("domain") or "?")[:22]
            fb_icon = _FB_ICON.get(h.get("_feedback", ""), "[dim]·[/dim]")
            parts.append(f"  [dim]{i}.[/dim]{fb_icon}[dim]{star}{title}[/dim]")

        self.query_one("#hist-line", Label).update("".join(parts))


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

    /* ── finding viewer ── */
    FindingViewer {
        height: 1fr;
        border: solid $primary-darken-2;
    }
    #fv-scroll {
        height: 1fr;
        padding: 0 1;
        overflow-y: auto;
        overflow-x: hidden;
    }
    #fv-content {
        padding: 0 1;
    }

    /* ── history bar — borderless coloured strip ── */
    HistoryBar {
        height: 1;
        padding: 0 1;
        background: $primary-darken-3;
        overflow-x: hidden;
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
        Binding("pageup", "scroll_up", "Scroll up", show=False),
        Binding("pagedown", "scroll_down", "Scroll down", show=False),
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

        # Finding queue — items waiting to be reviewed
        self._pending_findings: list[dict] = []
        # Currently displayed finding (None when idle / all caught up)
        self._current_finding: Optional[dict] = None
        # History of findings that have been reviewed in this session
        self._intel_history: list[dict] = []

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        with Horizontal(id="top-panels"):
            yield StatusPanel(id="status")
            yield WorkingLog(id="worklog")
        yield FindingViewer(id="finding-viewer")
        yield HistoryBar(id="history-bar")
        with Horizontal(id="input-bar"):
            yield Label("▶", id="input-prompt")
            yield Input(
                id="conv-input",
                placeholder="j / n / c <text> / skip / report / history / help / q",
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
        self.query_one("#finding-viewer", FindingViewer).show_idle()
        self.query_one("#history-bar", HistoryBar).refresh_history([])

    def _tick_uptime(self) -> None:
        self.sub_title = f"uptime {_fmt_uptime(self._app_start_time)}"

    # ------------------------------------------------------------------
    # Background workers
    # ------------------------------------------------------------------

    async def _run_scheduler(self) -> None:
        await self._scheduler.run()

    async def _consume_events(self) -> None:
        work_log: RichLog = self.query_one("#wl-log", RichLog)

        while True:
            event: Event = await self._bus.receive()

            if event.kind == EventKind.REPORT_READY and event.data:
                work_log.write(_brief_line(event))
                items = event.data.get("items", [])
                if items:
                    self._pending_findings.extend(items)
                    # If no finding is currently on screen, show the first one
                    if self._current_finding is None:
                        self._show_next_finding()
                continue

            work_log.write(_brief_line(event))

    def _show_next_finding(self) -> None:
        """Pop the next pending finding and display it, or show idle state."""
        viewer = self.query_one("#finding-viewer", FindingViewer)

        if not self._pending_findings:
            self._current_finding = None
            n = len(self._intel_history)
            if n > 0:
                viewer.show_idle(f"All {n} finding(s) reviewed — waiting for next report…")
            else:
                viewer.show_idle("Waiting for findings…")
            return

        self._current_finding = self._pending_findings.pop(0)
        # position = how many reviewed + 1 (this one); total = reviewed + this + remaining
        position = len(self._intel_history) + 1
        total = position + len(self._pending_findings)
        viewer.show_finding(self._current_finding, position, total)

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
            await asyncio.sleep(1)

    # ------------------------------------------------------------------
    # Actions
    # ------------------------------------------------------------------

    def action_focus_input(self) -> None:
        self.query_one("#conv-input", Input).focus()

    def action_quit(self) -> None:
        self._scheduler.stop()
        self.exit()

    def action_scroll_up(self) -> None:
        self.query_one("#finding-viewer", FindingViewer).scroll_up()

    def action_scroll_down(self) -> None:
        self.query_one("#finding-viewer", FindingViewer).scroll_down()

    # ------------------------------------------------------------------
    # Command dispatch
    # ------------------------------------------------------------------

    async def on_input_submitted(self, event: Input.Submitted) -> None:
        text = event.value.strip()
        event.input.clear()
        if text:
            await self._handle_command(text)
        event.input.focus()

    async def _handle_command(self, text: str) -> None:
        cmd = text.lower().strip()

        if cmd in ("help", "?", "h"):
            self._cmd_help()
            return

        if cmd in ("report", "r"):
            await self._cmd_report()
            return

        if cmd in ("history", "hist"):
            self._cmd_history()
            return

        if cmd in ("skip", "s"):
            self._cmd_skip()
            return

        # Simple feedback on current finding
        if cmd == "j":
            await self._apply_feedback("j", "")
            return
        if cmd == "n":
            await self._apply_feedback("n", "")
            return
        m_c = re.match(r"^c\s+(.+)$", text, re.IGNORECASE)
        if m_c:
            await self._apply_feedback("c", m_c.group(1).strip())
            return

        # Legacy / history format: <N>j / <N>n / <N>c <text>
        m = re.match(r"^(\d+)\s*([jJnNcC])(.*)?$", text)
        if m:
            await self._handle_history_feedback(m)
            return

        self.query_one("#finding-viewer", FindingViewer).show_markdown(
            "**Unknown command.** Type `help` for available commands.",
            header="[red]Error[/red]",
        )

    # ------------------------------------------------------------------
    # Individual commands
    # ------------------------------------------------------------------

    def _cmd_help(self) -> None:
        self.query_one("#finding-viewer", FindingViewer).show_markdown(
            "## Commands\n\n"
            "### Feedback on the current finding\n"
            "- **j** — mark as interesting (+)\n"
            "- **n** — mark as not interesting (−)\n"
            "- **c \\<text\\>** — comment (LLM infers crawl directives)\n"
            "- **skip** / **s** — skip without feedback\n\n"
            "### Navigation\n"
            "- **history** / **hist** — show reviewed intel history\n"
            "- **PgUp** / **PgDn** — scroll the current finding (or use mouse)\n\n"
            "### Reports\n"
            "- **report** / **r** — trigger report generation now\n\n"
            "### Feedback on a history item\n"
            "- **\\<N\\>j** / **\\<N\\>n** / **\\<N\\>c \\<text\\>** "
            "— give feedback on history item N\n\n"
            "### Other\n"
            "- **help** / **?** / **h** — show this help\n"
            "- **q** — quit Tipster\n",
            header="[bold]HELP[/bold]",
        )

    async def _cmd_report(self) -> None:
        from tipster.reporter import generate_report
        viewer = self.query_one("#finding-viewer", FindingViewer)
        if self._current_finding is None:
            viewer.show_idle("Generating report…")
        result = await generate_report(self._topic_id, self._cfg, self._bus)
        if result is None and self._current_finding is None:
            viewer.show_idle("No unreported items to report yet.")

    def _cmd_history(self) -> None:
        viewer = self.query_one("#finding-viewer", FindingViewer)
        if not self._intel_history:
            viewer.show_markdown(
                "*No items reviewed yet in this session.*",
                header="[bold]INTEL HISTORY[/bold]",
            )
            return

        _FB_LABEL = {
            "j":    "✓ interesting",
            "n":    "✗ not interesting",
            "c":    "💬 comment",
            "skip": "— skipped",
        }
        lines: list[str] = ["## Intel History\n"]
        for i, h in enumerate(self._intel_history, 1):
            fb = h.get("_feedback", "")
            fb_label = _FB_LABEL.get(fb, "")
            star = "★ NEW SOURCE  " if h.get("is_new_source") else ""
            title = h.get("title") or h.get("domain") or h.get("url", "?")
            score = h.get("score", 0.0)
            url = h.get("url", "")
            lines.append(f"### {i}. {star}{title}")
            lines.append(f"*{h.get('domain', '')} · relevance {score:.2f}*  {fb_label}")
            if fb == "c" and h.get("_comment"):
                lines.append(f"\n> {h['_comment']}")
            if url:
                lines.append(f"\n[{url}]({url})")
            lines.append("\n---\n")

        viewer.show_markdown("\n".join(lines), header="[bold]INTEL HISTORY[/bold]")

    def _cmd_skip(self) -> None:
        if self._current_finding is None:
            self.query_one("#finding-viewer", FindingViewer).show_markdown(
                "*No active finding to skip.*"
            )
            return
        item = self._current_finding
        item["_feedback"] = "skip"
        self._intel_history.append(item)
        self.query_one("#history-bar", HistoryBar).refresh_history(self._intel_history)
        self._show_next_finding()

    async def _apply_feedback(self, action: str, comment: str) -> None:
        """Apply feedback (j/n/c) to the currently displayed finding."""
        from tipster.feedback import process_comment, process_judgement

        viewer = self.query_one("#finding-viewer", FindingViewer)

        if self._current_finding is None:
            viewer.show_markdown(
                "*No active finding — wait for a report or type **report** to generate one.*",
                header="[yellow]No finding[/yellow]",
            )
            return

        item = self._current_finding
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
            item["_feedback"] = "j"

        elif action == "n":
            await loop.run_in_executor(
                None,
                lambda: process_judgement(
                    self._topic_id, item_id, url_id, -1, snippet, domain
                ),
            )
            item["_feedback"] = "n"

        elif action == "c":
            if not comment:
                viewer.show_markdown(
                    "**Usage:** `c <your comment text>`",
                    header="[red]Error[/red]",
                )
                return
            item["_feedback"] = "c"
            item["_comment"] = comment
            # Run comment processing but don't block the UI flow
            directives = await process_comment(
                self._topic_id, item_id, url_id, comment, self._cfg, self._bus
            )
            # Note: result will appear in working log via DIRECTIVE_APPLIED event

        self._intel_history.append(item)
        self.query_one("#history-bar", HistoryBar).refresh_history(self._intel_history)
        self._show_next_finding()

    async def _handle_history_feedback(self, m: re.Match) -> None:
        """Handle <N>j / <N>n / <N>c feedback on a previously reviewed item."""
        from tipster.feedback import process_comment, process_judgement

        viewer = self.query_one("#finding-viewer", FindingViewer)
        idx = int(m.group(1)) - 1  # 1-based → 0-based
        action = m.group(2).lower()
        extra = (m.group(3) or "").strip()

        if idx < 0 or idx >= len(self._intel_history):
            viewer.show_markdown(
                f"**No history item #{idx + 1}** "
                f"({len(self._intel_history)} item(s) in history). "
                "Type `history` to see the list.",
                header="[red]Error[/red]",
            )
            return

        item = self._intel_history[idx]
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
            item["_feedback"] = "j"
            self.query_one("#history-bar", HistoryBar).refresh_history(self._intel_history)
            viewer.show_markdown(f"**✓ History item #{idx + 1} marked interesting.**")

        elif action == "n":
            await loop.run_in_executor(
                None,
                lambda: process_judgement(
                    self._topic_id, item_id, url_id, -1, snippet, domain
                ),
            )
            item["_feedback"] = "n"
            self.query_one("#history-bar", HistoryBar).refresh_history(self._intel_history)
            viewer.show_markdown(f"**✓ History item #{idx + 1} marked not interesting.**")

        elif action == "c":
            if not extra:
                viewer.show_markdown(
                    "**Usage:** `<N>c <your comment text>`",
                    header="[red]Error[/red]",
                )
                return
            item["_feedback"] = "c"
            item["_comment"] = extra
            self.query_one("#history-bar", HistoryBar).refresh_history(self._intel_history)
            directives = await process_comment(
                self._topic_id, item_id, url_id, extra, self._cfg, self._bus
            )
            if directives:
                viewer.show_markdown(
                    f"**✓ {len(directives)} directive(s) created:** {', '.join(directives)}"
                )
            else:
                viewer.show_markdown("*Comment saved. No directives inferred.*")
