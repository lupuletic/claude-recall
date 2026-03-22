"""Shared fixtures for claude-recall tests."""

from __future__ import annotations

import json
import os
import sqlite3
from pathlib import Path

import pytest

from claude_recall.db import get_connection, upsert_chunks, upsert_session
from claude_recall.models import Session


# ---------------------------------------------------------------------------
# Helper: build a JSONL session line
# ---------------------------------------------------------------------------

def _user_line(text: str) -> str:
    """Build a JSONL line for a user message."""
    return json.dumps({
        "type": "user",
        "message": {"role": "user", "content": text},
    })


def _user_line_blocks(blocks: list[dict]) -> str:
    """Build a JSONL line for a user message with content blocks."""
    return json.dumps({
        "type": "user",
        "message": {"role": "user", "content": blocks},
    })


def _assistant_line(text: str) -> str:
    """Build a JSONL line for an assistant message."""
    return json.dumps({
        "type": "assistant",
        "message": {
            "role": "assistant",
            "content": [{"type": "text", "text": text}],
        },
    })


def _system_line() -> str:
    """Build a JSONL line for a system/init message (should be ignored)."""
    return json.dumps({"type": "system", "message": {"role": "system", "content": "init"}})


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def db_path(tmp_path: Path) -> Path:
    """Return a temporary DB path (file does not exist yet)."""
    return tmp_path / "test_index.db"


@pytest.fixture
def db_conn(db_path: Path) -> sqlite3.Connection:
    """Return a connection to a freshly-created test database."""
    conn = get_connection(db_path)
    yield conn
    conn.close()


@pytest.fixture
def sample_session() -> Session:
    """A single realistic Session object."""
    return Session(
        session_id="abc123",
        project_path="/Users/testuser/Projects/myapp",
        project_dir="-Users-testuser-Projects-myapp",
        file_path="/tmp/fake/abc123.jsonl",
        summary="Debugging auth middleware",
        first_prompt="Help me debug the auth middleware",
        first_reply="I can help with that. Let me look at the code.",
        last_prompt="Can you also add tests?",
        last_reply="Sure, here are the tests.",
        messages_text="Help me debug the auth middleware\nCan you also add tests?",
        git_branch="fix/auth-bug",
        message_count=5,
        file_size=2048,
        created="2025-01-15T10:00:00Z",
        modified="2025-01-15T11:30:00Z",
        mtime=1736935800.0,
        is_subagent=False,
        parent_session=None,
    )


@pytest.fixture
def sample_session_2() -> Session:
    """A second Session for multi-result tests."""
    return Session(
        session_id="def456",
        project_path="/Users/testuser/Projects/webapp",
        project_dir="-Users-testuser-Projects-webapp",
        file_path="/tmp/fake/def456.jsonl",
        summary="Setting up React router",
        first_prompt="Set up React router for our app",
        first_reply="Let me configure the routes.",
        last_prompt="Add a 404 page",
        last_reply="Done, added NotFound component.",
        messages_text="Set up React router for our app\nAdd a 404 page",
        git_branch="feature/routing",
        message_count=8,
        file_size=4096,
        created="2025-02-01T08:00:00Z",
        modified="2025-02-01T10:00:00Z",
        mtime=1738396800.0,
        is_subagent=False,
        parent_session=None,
    )


@pytest.fixture
def subagent_session() -> Session:
    """A subagent session."""
    return Session(
        session_id="sub789",
        project_path="/Users/testuser/Projects/myapp",
        project_dir="-Users-testuser-Projects-myapp",
        file_path="/tmp/fake/sub789.jsonl",
        summary=None,
        first_prompt="Run the linter",
        first_reply="Linting complete.",
        last_prompt="Run the linter",
        last_reply="Linting complete.",
        messages_text="Run the linter",
        git_branch=None,
        message_count=1,
        file_size=512,
        created="2025-01-15T10:30:00Z",
        modified="2025-01-15T10:31:00Z",
        mtime=1736937060.0,
        is_subagent=True,
        parent_session="abc123",
    )


@pytest.fixture
def populated_db(db_conn, sample_session, sample_session_2, subagent_session):
    """A DB with three sessions (2 main + 1 subagent) and their chunks."""
    upsert_session(db_conn, sample_session)
    upsert_chunks(db_conn, sample_session.session_id, [
        "User: Help me debug the auth middleware\nAssistant: I can help with that.",
        "User: Can you also add tests?\nAssistant: Sure, here are the tests.",
    ])

    upsert_session(db_conn, sample_session_2)
    upsert_chunks(db_conn, sample_session_2.session_id, [
        "User: Set up React router for our app\nAssistant: Let me configure the routes.",
        "User: Add a 404 page\nAssistant: Done, added NotFound component.",
    ])

    upsert_session(db_conn, subagent_session)
    upsert_chunks(db_conn, subagent_session.session_id, [
        "User: Run the linter\nAssistant: Linting complete.",
    ])

    db_conn.commit()
    return db_conn


@pytest.fixture
def sample_jsonl_file(tmp_path: Path) -> Path:
    """Create a realistic session .jsonl file and return its path."""
    lines = [
        _system_line(),
        _user_line("Help me debug the authentication middleware"),
        _assistant_line("I'll help you debug the auth middleware. Let me look at the code."),
        _user_line("The login endpoint returns 401 even with valid credentials"),
        _assistant_line("I see the issue. The token validation is checking the wrong header."),
        _user_line("Can you fix it?"),
        _assistant_line("Fixed! The middleware now reads from the Authorization header correctly."),
    ]
    file_path = tmp_path / "session-abc.jsonl"
    file_path.write_text("\n".join(lines) + "\n")
    return file_path


@pytest.fixture
def sample_jsonl_file_with_markup(tmp_path: Path) -> Path:
    """Create a session file with internal Claude Code markup in messages."""
    lines = [
        _user_line(
            "<system-reminder>You are Claude.</system-reminder>"
            "Fix the bug in utils.py"
        ),
        _assistant_line(
            "<local-command-caveat>Running command</local-command-caveat>"
            "I found the issue in line 42."
        ),
        _user_line(
            "<task-notification>Task completed</task-notification>"
            "Thanks, now add tests"
        ),
        _assistant_line("Tests added."),
    ]
    file_path = tmp_path / "session-markup.jsonl"
    file_path.write_text("\n".join(lines) + "\n")
    return file_path


@pytest.fixture
def empty_jsonl_file(tmp_path: Path) -> Path:
    """Create a session file with no user messages."""
    lines = [_system_line()]
    file_path = tmp_path / "session-empty.jsonl"
    file_path.write_text("\n".join(lines) + "\n")
    return file_path


@pytest.fixture
def projects_dir(tmp_path: Path) -> Path:
    """Create a realistic Claude projects directory structure.

    Structure:
        projects_dir/
            -Users-test-Projects-myapp/
                sessions-index.json
                session-001.jsonl
                session-002.jsonl
                session-001/
                    subagents/
                        agent-sub1.jsonl
            -Users-test-Projects-webapp/
                session-003.jsonl
    """
    proj1 = tmp_path / "-Users-test-Projects-myapp"
    proj1.mkdir()

    # sessions-index.json
    idx = {
        "originalPath": "/Users/test/Projects/myapp",
        "entries": [
            {
                "sessionId": "session-001",
                "summary": "Auth middleware debugging",
                "created": "2025-01-15T10:00:00Z",
                "modified": "2025-01-15T11:30:00Z",
                "gitBranch": "fix/auth",
            },
            {
                "sessionId": "session-002",
                "summary": "Database migration",
                "created": "2025-01-20T09:00:00Z",
                "modified": "2025-01-20T10:00:00Z",
            },
        ],
    }
    (proj1 / "sessions-index.json").write_text(json.dumps(idx))

    # Session files
    lines1 = [
        _user_line("Help me debug the auth middleware"),
        _assistant_line("Looking at the code now."),
        _user_line("The login returns 401"),
        _assistant_line("Fixed the token check."),
    ]
    (proj1 / "session-001.jsonl").write_text("\n".join(lines1) + "\n")

    lines2 = [
        _user_line("Run the database migration"),
        _assistant_line("Migration completed successfully."),
    ]
    (proj1 / "session-002.jsonl").write_text("\n".join(lines2) + "\n")

    # Subagent
    sub_dir = proj1 / "session-001" / "subagents"
    sub_dir.mkdir(parents=True)
    sub_lines = [
        _user_line("Run linter"),
        _assistant_line("No issues found."),
    ]
    (sub_dir / "agent-sub1.jsonl").write_text("\n".join(sub_lines) + "\n")

    # Second project (no sessions-index.json)
    proj2 = tmp_path / "-Users-test-Projects-webapp"
    proj2.mkdir()
    lines3 = [
        _user_line("Set up React routing"),
        _assistant_line("Routes configured."),
        _user_line("Add navigation component"),
        _assistant_line("Navigation component added."),
    ]
    (proj2 / "session-003.jsonl").write_text("\n".join(lines3) + "\n")

    return tmp_path


@pytest.fixture
def sessions_index_path(tmp_path: Path) -> Path:
    """Create a standalone sessions-index.json."""
    proj_dir = tmp_path / "-Users-alice-my-project"
    proj_dir.mkdir()
    idx = {
        "originalPath": "/Users/alice/my-project",
        "entries": [],
    }
    (proj_dir / "sessions-index.json").write_text(json.dumps(idx))
    return tmp_path
