"""Interactive TUI for claude-recall using textual."""

from __future__ import annotations

import os
import subprocess
import sys

from textual import on, work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.screen import ModalScreen
from textual.widgets import (
    Button,
    Footer,
    Input,
    Label,
    ListItem,
    ListView,
    OptionList,
    Static,
    Switch,
)
from textual.widgets.option_list import Option

from claude_recall.models import SearchResult
from claude_recall.utils import clean_display_text, format_date, format_size


def _session_title(s, max_len: int = 80) -> str:
    """Build a clean title for a session."""
    title = clean_display_text(s.summary) or clean_display_text(s.first_prompt) or "Untitled"
    if len(title) > max_len:
        title = title[:max_len] + "..."
    return title


def _score_color(score: float) -> str:
    """Return a color name based on score."""
    if score >= 0.8:
        return "green"
    elif score >= 0.5:
        return "yellow"
    elif score >= 0.2:
        return "dark_orange"
    return "dim"


class SessionItem(ListItem):
    """A single search result in the list."""

    def __init__(self, result: SearchResult, rank: int) -> None:
        super().__init__()
        self.result = result
        self.rank = rank

    def compose(self) -> ComposeResult:
        s = self.result.session
        score = self.result.score
        color = _score_color(score)

        title = _session_title(s)

        # Meta line
        meta_parts = []
        if s.modified:
            meta_parts.append(format_date(s.modified))
        meta_parts.append(f"{s.message_count} msgs")
        if s.file_size:
            meta_parts.append(format_size(s.file_size))
        project = self.result.display_project

        # Last activity
        last = ""
        if s.last_prompt and s.last_prompt != s.first_prompt:
            cleaned = clean_display_text(s.last_prompt)
            if cleaned:
                last = cleaned[:100]

        lines = [
            f"[bold]{self.rank}.[/bold] [bold]{title}[/bold]",
            f"   [{color}]{'█' * max(1, int(score * 10))}[/{color}]"
            f" [{color}]{score:.0%}[/{color}]"
            f"  [dim]{project} · {' · '.join(meta_parts)}[/dim]",
        ]
        if last:
            lines.append(f"   [italic dim]↳ {last}[/italic dim]")

        yield Static("\n".join(lines), markup=True)


class PreviewPanel(Static):
    """Shows detailed info about the selected session."""

    def update_preview(self, result: SearchResult | None) -> None:
        if result is None:
            self.update("[dim]Select a session to preview[/dim]")
            return

        s = result.session
        lines = []

        # Header
        title = _session_title(s, 120)
        lines.append(f"[bold underline]{title}[/bold underline]\n")

        # Metadata
        lines.append(f"[bold]Project:[/bold]  {result.display_project}")
        if s.git_branch:
            lines.append(f"[bold]Branch:[/bold]   {s.git_branch}")
        if s.modified:
            lines.append(f"[bold]Date:[/bold]     {format_date(s.modified)}")
        lines.append(f"[bold]Messages:[/bold] {s.message_count}")
        lines.append(f"[bold]Size:[/bold]     {format_size(s.file_size)}")
        lines.append(f"[bold]Score:[/bold]    {result.score:.0%}")

        # First prompt
        if s.first_prompt:
            fp = clean_display_text(s.first_prompt) or ""
            if fp:
                lines.append(f"\n[bold]Started with:[/bold]\n{fp[:400]}")

        # Last activity
        if s.last_prompt and s.last_prompt != s.first_prompt:
            lp = clean_display_text(s.last_prompt) or ""
            if lp:
                lines.append(f"\n[bold]Left off at:[/bold]\n{lp[:400]}")

        lines.append(f"\n[bold green]↵ Enter to resume this session[/bold green]")
        lines.append(f"[dim]ID: {s.session_id}[/dim]")

        self.update("\n".join(lines))


class SettingsScreen(ModalScreen):
    """Settings modal overlay with arrow-key navigation."""

    CSS = """
    SettingsScreen {
        align: center middle;
    }
    #settings-dialog {
        width: 80;
        height: auto;
        max-height: 90%;
        border: tall $primary;
        background: $surface;
        padding: 1 2;
    }
    #settings-title {
        text-align: center;
        text-style: bold;
        padding: 0 0 1 0;
        color: $text-primary;
    }
    .section-label {
        text-style: bold;
        padding: 1 0 0 0;
    }
    #mode-list {
        height: 8;
        margin: 0 0 0 2;
        border: tall $border-blurred;
    }
    #mode-list:focus {
        border: tall $border;
    }
    .setting-row {
        height: 3;
        margin: 0 0 0 0;
    }
    .setting-key {
        width: 22;
        padding: 1 1 0 0;
    }
    .setting-value {
        width: 1fr;
    }
    .toggle-row {
        height: 3;
        margin: 0 0 0 0;
    }
    .toggle-label {
        width: 1fr;
        padding: 1 0 0 0;
    }
    Switch {
        width: auto;
    }
    #settings-buttons {
        height: 3;
        align: center middle;
        padding: 1 0 0 0;
    }
    #settings-buttons Button {
        margin: 0 1;
    }
    """

    BINDINGS = [
        Binding("escape", "cancel", "Cancel", show=True),
    ]

    def compose(self) -> ComposeResult:
        from claude_recall.config import SEARCH_MODES, load_config

        self._config = load_config()
        self._mode_keys = list(SEARCH_MODES.keys())

        with VerticalScroll(id="settings-dialog"):
            yield Static("Settings", id="settings-title")

            # Search mode — OptionList (arrow-key navigable)
            yield Static("Search Mode", classes="section-label")
            options = []
            highlighted = 0
            for i, (mode, desc) in enumerate(SEARCH_MODES.items()):
                options.append(Option(f"{mode}  —  {desc}"))
                if mode == self._config["search_mode"]:
                    highlighted = i
            ol = OptionList(*options, id="mode-list")
            yield ol

            # Numeric inputs
            yield Static("Search Options", classes="section-label")

            with Horizontal(classes="setting-row"):
                yield Static("Results Limit", classes="setting-key")
                yield Input(
                    value=str(self._config["limit"]),
                    type="integer",
                    id="limit-input",
                    classes="setting-value",
                )

            with Horizontal(classes="setting-row"):
                yield Static("Relevance Cutoff (0.0–1.0)", classes="setting-key")
                yield Input(
                    value=str(self._config["relevance_cutoff"]),
                    id="cutoff-input",
                    classes="setting-value",
                )

            # Toggles — Switch widgets (arrow-key & space friendly)
            yield Static("Toggles", classes="section-label")

            with Horizontal(classes="toggle-row"):
                yield Static("Show subagent sessions", classes="toggle-label")
                yield Switch(
                    value=self._config["show_subagents"],
                    id="switch-subagents",
                )

            with Horizontal(classes="toggle-row"):
                yield Static("Auto-install SessionEnd hook", classes="toggle-label")
                yield Switch(
                    value=self._config["auto_index_hook"],
                    id="switch-hook",
                )

            with Horizontal(id="settings-buttons"):
                yield Button("Save", variant="primary", id="save-btn")
                yield Button("Cancel", variant="default", id="cancel-btn")

    def on_mount(self) -> None:
        # Pre-select the current mode
        ol = self.query_one("#mode-list", OptionList)
        current_idx = 0
        for i, key in enumerate(self._mode_keys):
            if key == self._config["search_mode"]:
                current_idx = i
                break
        ol.highlighted = current_idx
        ol.focus()

    @on(Button.Pressed, "#save-btn")
    def on_save(self, event: Button.Pressed) -> None:
        self._save_settings()

    @on(Button.Pressed, "#cancel-btn")
    def on_cancel(self, event: Button.Pressed) -> None:
        self.dismiss(False)

    def action_cancel(self) -> None:
        self.dismiss(False)

    def _save_settings(self) -> None:
        from claude_recall.config import save_config

        # Search mode from OptionList
        ol = self.query_one("#mode-list", OptionList)
        if ol.highlighted is not None and ol.highlighted < len(self._mode_keys):
            self._config["search_mode"] = self._mode_keys[ol.highlighted]

        # Inputs
        try:
            self._config["limit"] = int(self.query_one("#limit-input", Input).value)
        except ValueError:
            pass
        try:
            self._config["relevance_cutoff"] = float(
                self.query_one("#cutoff-input", Input).value
            )
        except ValueError:
            pass

        # Switches
        self._config["show_subagents"] = self.query_one("#switch-subagents", Switch).value
        self._config["auto_index_hook"] = self.query_one("#switch-hook", Switch).value

        save_config(self._config)
        self.dismiss(True)


class RecallApp(App):
    """claude-recall interactive session search."""

    CSS = """
    #search-box {
        dock: top;
        height: 3;
        padding: 0 1;
    }
    #search-input {
        width: 1fr;
    }
    #status {
        dock: top;
        height: 1;
        padding: 0 2;
        color: $text-muted;
    }
    #main {
        height: 1fr;
    }
    #results {
        width: 1fr;
        min-width: 40;
    }
    #preview {
        width: 45%;
        border-left: tall $primary;
        padding: 1 2;
        overflow-y: auto;
        display: none;
    }
    #preview.visible {
        display: block;
    }
    SessionItem {
        padding: 0 1;
        height: auto;
        margin: 0 0 1 0;
    }
    SessionItem:hover {
        background: $surface-lighten-1;
    }
    SessionItem.-highlight {
        background: $primary-darken-2;
    }
    """

    BINDINGS = [
        Binding("escape", "quit", "Quit", show=True, priority=True),
        Binding("tab", "toggle_preview", "Preview", show=True),
        Binding("ctrl+d", "summarize", "AI Summary", show=True),
        Binding("ctrl+s", "open_settings", "Settings", show=True),
        Binding("ctrl+c", "quit", "Quit", show=False),
    ]

    def __init__(self, initial_query: str = "", initial_results: list[SearchResult] | None = None):
        super().__init__()
        self.initial_query = initial_query
        self._results = initial_results or []
        self._selected_result: SearchResult | None = None

    def compose(self) -> ComposeResult:
        yield Input(
            placeholder="Search your Claude Code sessions...",
            value=self.initial_query,
            id="search-input",
        )
        yield Label("Type to search — Enter to select — Tab for preview — Esc to quit", id="status")
        with Horizontal(id="main"):
            yield ListView(id="results")
            yield PreviewPanel(id="preview")
        yield Footer()

    def on_mount(self) -> None:
        self.title = "claude-recall"
        if self._results and self.initial_query.strip():
            self._display_results(self._results)
        self.query_one("#search-input", Input).focus()

    @on(Input.Changed, "#search-input")
    def on_search_changed(self, event: Input.Changed) -> None:
        """Debounced search on input change."""
        query = event.value.strip()
        status = self.query_one("#status", Label)

        if not query:
            status.update("Type to search your Claude Code sessions")
            self.query_one("#results", ListView).clear()
            self._selected_result = None
            # Clear preview
            preview = self.query_one("#preview", PreviewPanel)
            preview.update("[dim]Type a query to search[/dim]")
            return

        # Show what we're searching for — user knows search is happening
        status.update(f'Searching for "{query}"...')
        # Dim old results to indicate they're stale
        list_view = self.query_one("#results", ListView)
        list_view.styles.opacity = 0.4
        self._debounced_search(event.value)

    @on(Input.Submitted, "#search-input")
    def on_search_submitted(self, event: Input.Submitted) -> None:
        """When Enter is pressed in search input, focus the results list."""
        list_view = self.query_one("#results", ListView)
        if list_view.children:
            list_view.focus()
            list_view.index = 0

    def on_key(self, event) -> None:
        """Keyboard navigation and shortcuts."""
        input_widget = self.query_one("#search-input", Input)

        # Cmd+Backspace: clear entire search
        if event.key == "super+backspace" or event.key == "cmd+backspace":
            if self.focused == input_widget:
                input_widget.value = ""
                event.prevent_default()
                return

        # Ctrl+Backspace: delete last word
        if event.key == "ctrl+w" or event.key == "ctrl+backspace":
            if self.focused == input_widget:
                text = input_widget.value
                stripped = text.rstrip()
                if " " in stripped:
                    input_widget.value = stripped[:stripped.rfind(" ") + 1]
                else:
                    input_widget.value = ""
                input_widget.cursor_position = len(input_widget.value)
                event.prevent_default()
                return

        # Ctrl+D: AI summary (works from anywhere when a result is selected)
        if event.key == "ctrl+d":
            if self._selected_result:
                self.action_summarize()
                event.prevent_default()
                return

        if event.key == "down":
            focused = self.focused
            if focused and focused.id == "search-input":
                list_view = self.query_one("#results", ListView)
                if list_view.children:
                    list_view.focus()
                    list_view.index = 0
                    event.prevent_default()
        elif event.key == "up":
            list_view = self.query_one("#results", ListView)
            if self.focused == list_view and list_view.index == 0:
                input_widget.focus()
                input_widget.cursor_position = len(input_widget.value)
                event.prevent_default()

    @on(ListView.Selected, "#results")
    def on_result_selected(self, event: ListView.Selected) -> None:
        """When Enter is pressed on a result, resume that session."""
        if event.item and isinstance(event.item, SessionItem):
            session_id = event.item.result.session.session_id
            self.exit(session_id)

    @on(ListView.Highlighted, "#results")
    def on_result_highlighted(self, event: ListView.Highlighted) -> None:
        """Update preview when a result is highlighted."""
        if event.item and isinstance(event.item, SessionItem):
            self._selected_result = event.item.result
            # Auto-show preview panel
            preview = self.query_one("#preview", PreviewPanel)
            if "visible" not in preview.classes:
                preview.add_class("visible")
            preview.update_preview(event.item.result)
            # Update status with context-aware hints
            self.query_one("#status", Label).update(
                "Enter=Resume  Ctrl+D=AI Summary  Tab=Hide Preview  Esc=Quit"
            )

    @work(exclusive=True, thread=True)
    def _debounced_search(self, query: str) -> None:
        """Search with debounce (runs in a thread)."""
        import time

        time.sleep(0.5)

        if not query.strip():
            self.call_from_thread(self._display_results, [], "")
            return

        # Check if the query changed while we were waiting
        current = self.query_one("#search-input", Input).value.strip()
        if current != query.strip():
            return  # User kept typing — skip this search, next one will run

        from claude_recall.searcher import search as do_search

        results = do_search(query=query, limit=20)
        self._results = results
        self.call_from_thread(self._display_results, results, query)

    def _display_results(self, results: list[SearchResult], query: str = "") -> None:
        """Update the results list."""
        list_view = self.query_one("#results", ListView)
        list_view.clear()
        list_view.styles.opacity = 1.0  # Restore full opacity

        status = self.query_one("#status", Label)

        if not results:
            if query:
                status.update(f'No results for "{query}"')
            else:
                status.update("No results")
            return

        status.update(
            f'Found {len(results)} sessions for "{query}" — '
            f"↓ to navigate, Enter to resume, Ctrl+D for AI summary"
        )

        for i, result in enumerate(results, 1):
            list_view.append(SessionItem(result, i))

    def action_toggle_preview(self) -> None:
        """Toggle the preview panel."""
        preview = self.query_one("#preview", PreviewPanel)
        preview.toggle_class("visible")

    def action_summarize(self) -> None:
        """Generate an AI summary of the selected session using claude -p."""
        if not self._selected_result:
            return
        preview = self.query_one("#preview", PreviewPanel)
        preview.toggle_class("visible")
        if "visible" not in preview.classes:
            preview.add_class("visible")
        preview.update("[bold]Generating AI summary...[/bold]")
        self._run_summarize(self._selected_result)

    @work(exclusive=True, thread=True)
    def _run_summarize(self, result: SearchResult) -> None:
        """Run claude -p to summarize a session."""
        import shutil
        import subprocess

        s = result.session
        claude_bin = shutil.which("claude")
        if not claude_bin:
            self.call_from_thread(
                self.query_one("#preview", PreviewPanel).update,
                "[red]Claude CLI not found[/red]",
            )
            return

        prompt = (
            f"Summarize this Claude Code session in 3-5 bullet points. "
            f"What was the goal? What was accomplished? What was the last thing worked on?\n\n"
            f"Project: {result.display_project}\n"
            f"First message: {(s.first_prompt or '')[:300]}\n"
            f"Last message: {(s.last_prompt or '')[:300]}\n"
            f"Messages: {s.message_count}\n"
        )

        try:
            proc = subprocess.run(
                [claude_bin, "-p", "--model", "haiku", "--no-session-persistence"],
                input=prompt,
                capture_output=True,
                text=True,
                timeout=20,
            )
            summary = proc.stdout.strip() if proc.returncode == 0 else "Summary failed"
        except Exception as e:
            summary = f"Error: {e}"

        title = _session_title(s, 120)
        output = (
            f"[bold underline]{title}[/bold underline]\n\n"
            f"[bold]Project:[/bold]  {result.display_project}\n"
            f"[bold]Date:[/bold]     {format_date(s.modified)}\n"
            f"[bold]Messages:[/bold] {s.message_count}\n\n"
            f"[bold]AI Summary:[/bold]\n{summary}\n\n"
            f"[bold green]↵ Enter to resume[/bold green]  "
            f"[dim]ID: {s.session_id}[/dim]"
        )
        self.call_from_thread(
            self.query_one("#preview", PreviewPanel).update,
            output,
        )

    def action_open_settings(self) -> None:
        """Open the settings modal."""
        def on_settings_dismissed(saved: bool) -> None:
            if saved:
                status = self.query_one("#status", Label)
                status.update("Settings saved")

        self.push_screen(SettingsScreen(), on_settings_dismissed)

    def action_quit(self) -> None:
        self.exit(None)


def run_tui(query: str, results: list[SearchResult]) -> None:
    """Launch the TUI and handle the result."""
    result_map = {r.session.session_id: r for r in results}

    app = RecallApp(initial_query=query, initial_results=results)
    session_id = app.run()

    if session_id:
        result = result_map.get(session_id) or (
            next((r for r in app._results if r.session.session_id == session_id), None)
        )
        project_path = result.session.project_path if result else None

        if project_path:
            # Try the exact path first, then walk up to find an existing parent
            resume_dir = project_path
            while resume_dir and not os.path.isdir(resume_dir):
                resume_dir = os.path.dirname(resume_dir)

            if resume_dir and os.path.isdir(resume_dir):
                os.chdir(resume_dir)
                if resume_dir != project_path:
                    print(f"\nNote: {project_path} no longer exists", file=sys.stderr)
                    print(f"cd {resume_dir} (nearest parent)", file=sys.stderr)
                else:
                    print(f"\ncd {resume_dir}", file=sys.stderr)

        print(f"claude --resume {session_id}", file=sys.stderr)
        if sys.platform == "win32":
            subprocess.run(["claude", "--resume", session_id])
        else:
            os.execvp("claude", ["claude", "--resume", session_id])
