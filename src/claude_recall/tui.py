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
from textual.widgets import Button, Footer, Input, Label, ListItem, ListView, Static

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
    """Settings modal overlay."""

    CSS = """
    SettingsScreen {
        align: center middle;
    }
    #settings-dialog {
        width: 65;
        height: auto;
        max-height: 85%;
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
    .setting-row {
        height: 3;
        padding: 0 0 0 0;
    }
    .setting-key {
        width: 22;
        text-style: bold;
        padding: 1 1 0 0;
    }
    .setting-value {
        width: 1fr;
    }
    .setting-value Input {
        width: 100%;
    }
    #settings-buttons {
        height: 3;
        align: center middle;
        padding: 1 0 0 0;
    }
    #settings-buttons Button {
        margin: 0 1;
    }
    .mode-option {
        height: 1;
        padding: 0 0 0 2;
    }
    .mode-option.selected {
        color: $success;
        text-style: bold;
    }
    """

    BINDINGS = [
        Binding("escape", "cancel", "Cancel", show=True),
    ]

    def compose(self) -> ComposeResult:
        from claude_recall.config import SEARCH_MODES, load_config

        self._config = load_config()

        with VerticalScroll(id="settings-dialog"):
            yield Static("Settings", id="settings-title")

            # Search mode as a selectable list
            yield Static("[bold]Search Mode[/bold]", markup=True)
            for mode, desc in SEARCH_MODES.items():
                selected = mode == self._config["search_mode"]
                prefix = "[green bold]>[/green bold]" if selected else " "
                yield Static(
                    f"  {prefix} [bold]{mode}[/bold] — [dim]{desc}[/dim]",
                    markup=True,
                    classes=f"mode-option {'selected' if selected else ''}",
                    id=f"mode-{mode}",
                )

            yield Static("")

            # Inputs
            with Horizontal(classes="setting-row"):
                yield Static("Results Limit", classes="setting-key")
                yield Input(
                    value=str(self._config["limit"]),
                    type="integer",
                    id="limit-input",
                    classes="setting-value",
                )

            with Horizontal(classes="setting-row"):
                yield Static("Relevance Cutoff", classes="setting-key")
                yield Input(
                    value=str(self._config["relevance_cutoff"]),
                    id="cutoff-input",
                    classes="setting-value",
                )

            yield Static("")

            # Toggles as clickable labels
            sub = "on" if self._config["show_subagents"] else "off"
            yield Static(
                f"  [{'green' if self._config['show_subagents'] else 'dim'}]"
                f"[{'x' if self._config['show_subagents'] else ' '}][/] "
                f"Show subagent sessions",
                markup=True,
                id="toggle-subagents",
            )
            hook = "on" if self._config["auto_index_hook"] else "off"
            yield Static(
                f"  [{'green' if self._config['auto_index_hook'] else 'dim'}]"
                f"[{'x' if self._config['auto_index_hook'] else ' '}][/] "
                f"Auto-install SessionEnd hook",
                markup=True,
                id="toggle-hook",
            )

            yield Static("")

            with Horizontal(id="settings-buttons"):
                yield Button("Save", variant="primary", id="save-btn")
                yield Button("Cancel", variant="default", id="cancel-btn")

    @on(Static.Click)
    def on_static_click(self, event: Static.Click) -> None:
        """Handle clicks on mode options and toggles."""
        widget_id = event.static.id or ""

        if widget_id.startswith("mode-"):
            mode = widget_id[5:]
            self._config["search_mode"] = mode
            # Update visual state
            from claude_recall.config import SEARCH_MODES
            for m in SEARCH_MODES:
                w = self.query_one(f"#mode-{m}", Static)
                selected = m == mode
                prefix = "[green bold]>[/green bold]" if selected else " "
                desc = SEARCH_MODES[m]
                w.update(
                    f"  {prefix} [bold]{m}[/bold] — [dim]{desc}[/dim]"
                )

        elif widget_id == "toggle-subagents":
            self._config["show_subagents"] = not self._config["show_subagents"]
            v = self._config["show_subagents"]
            event.static.update(
                f"  [{'green' if v else 'dim'}][{'x' if v else ' '}][/] "
                f"Show subagent sessions"
            )

        elif widget_id == "toggle-hook":
            self._config["auto_index_hook"] = not self._config["auto_index_hook"]
            v = self._config["auto_index_hook"]
            event.static.update(
                f"  [{'green' if v else 'dim'}][{'x' if v else ' '}][/] "
                f"Auto-install SessionEnd hook"
            )

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

        # Read inputs
        try:
            self._config["limit"] = int(self.query_one("#limit-input", Input).value)
        except ValueError:
            pass
        try:
            self._config["relevance_cutoff"] = float(self.query_one("#cutoff-input", Input).value)
        except ValueError:
            pass

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
        status = self.query_one("#status", Label)
        if event.value.strip():
            status.update("Searching...")
        else:
            status.update("Type to search your Claude Code sessions")
            # Clear results when search is empty
            self.query_one("#results", ListView).clear()
            return
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
                input_widget = self.query_one("#search-input", Input)
                input_widget.focus()
                # Move cursor to end instead of selecting all
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
            preview = self.query_one("#preview", PreviewPanel)
            preview.update_preview(event.item.result)

    @work(exclusive=True, thread=True)
    def _debounced_search(self, query: str) -> None:
        """Search with debounce (runs in a thread)."""
        import time

        time.sleep(0.5)

        if not query.strip():
            self.call_from_thread(self._display_results, [])
            return

        from claude_recall.searcher import search as do_search

        results = do_search(query=query, limit=20)
        self._results = results
        self.call_from_thread(self._display_results, results)

    def _display_results(self, results: list[SearchResult]) -> None:
        """Update the results list."""
        list_view = self.query_one("#results", ListView)
        list_view.clear()

        status = self.query_one("#status", Label)

        if not results:
            status.update("No results")
            return

        status.update(
            f"Found {len(results)} sessions — "
            f"Enter to focus list, ↑↓ to navigate, Enter to resume"
        )

        for i, result in enumerate(results, 1):
            list_view.append(SessionItem(result, i))

    def action_toggle_preview(self) -> None:
        """Toggle the preview panel."""
        preview = self.query_one("#preview", PreviewPanel)
        preview.toggle_class("visible")

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

        if project_path and os.path.isdir(project_path):
            os.chdir(project_path)
            print(f"\ncd {project_path}", file=sys.stderr)

        print(f"claude --resume {session_id}", file=sys.stderr)
        if sys.platform == "win32":
            subprocess.run(["claude", "--resume", session_id])
        else:
            os.execvp("claude", ["claude", "--resume", session_id])
