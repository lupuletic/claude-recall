"""Interactive TUI for claude-recall using textual."""

from __future__ import annotations

import os
import subprocess
import sys

from textual import on, work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.widgets import Footer, Input, Label, ListItem, ListView, Static

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
