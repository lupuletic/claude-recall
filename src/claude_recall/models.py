"""Data models for claude-recall."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Session:
    """A Claude Code session extracted from a .jsonl transcript."""

    session_id: str
    project_path: str
    project_dir: str
    file_path: str
    summary: str | None = None
    first_prompt: str | None = None
    first_reply: str | None = None
    last_prompt: str | None = None
    last_reply: str | None = None
    messages_text: str | None = None
    git_branch: str | None = None
    message_count: int = 0
    file_size: int = 0
    created: str | None = None
    modified: str | None = None
    mtime: float = 0.0
    is_subagent: bool = False
    parent_session: str | None = None


@dataclass
class SearchResult:
    """A search result with relevance scoring and context."""

    session: Session
    score: float = 0.0
    fts_rank: float | None = None
    vec_score: float | None = None
    snippets: list[str] = field(default_factory=list)

    @property
    def display_project(self) -> str:
        """Short project path for display (~/Projects/foo)."""
        path = self.session.project_path
        import os

        home = os.path.expanduser("~")
        if path.startswith(home):
            return "~" + path[len(home) :]
        return path

    @property
    def resume_command(self) -> str:
        return f"claude --resume {self.session.session_id}"
