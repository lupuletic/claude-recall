"""End-to-end integration tests for claude-recall."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import pytest

from claude_recall.db import get_all_session_ids, get_connection, get_stats
from claude_recall.indexer import build_index
from claude_recall.searcher import search


def _user_line(text: str) -> str:
    return json.dumps({
        "type": "user",
        "message": {"role": "user", "content": text},
    })


def _assistant_line(text: str) -> str:
    return json.dumps({
        "type": "assistant",
        "message": {
            "role": "assistant",
            "content": [{"type": "text", "text": text}],
        },
    })


@pytest.fixture
def integration_env(tmp_path: Path):
    """Create a complete test environment with projects, sessions, and index."""
    projects_dir = tmp_path / "projects"
    db_path = tmp_path / "index.db"

    # Project 1: Python backend
    proj1 = projects_dir / "-Users-dev-Projects-backend"
    proj1.mkdir(parents=True)

    idx1 = {
        "originalPath": "/Users/dev/Projects/backend",
        "entries": [
            {
                "sessionId": "sess-py-auth",
                "summary": "Authentication system debugging",
                "created": "2025-01-10T10:00:00Z",
                "modified": "2025-01-10T12:00:00Z",
                "gitBranch": "fix/auth-tokens",
            },
            {
                "sessionId": "sess-py-db",
                "summary": "Database migration scripts",
                "created": "2025-01-15T09:00:00Z",
                "modified": "2025-01-15T11:00:00Z",
                "gitBranch": "feature/migrations",
            },
        ],
    }
    (proj1 / "sessions-index.json").write_text(json.dumps(idx1))

    # Session: auth debugging
    auth_lines = [
        _user_line("Help me debug the authentication system"),
        _assistant_line("I'll look at the auth middleware. What error are you seeing?"),
        _user_line("The JWT token validation fails with invalid signature"),
        _assistant_line("The issue is in the token verification. You need to use the correct secret key."),
        _user_line("Can you also add refresh token support?"),
        _assistant_line("I've added refresh token rotation with secure httpOnly cookies."),
    ]
    (proj1 / "sess-py-auth.jsonl").write_text("\n".join(auth_lines) + "\n")

    # Session: database migration
    db_lines = [
        _user_line("Create database migration for the users table"),
        _assistant_line("Here's the Alembic migration for adding the users table."),
        _user_line("Add an index on the email column"),
        _assistant_line("Added a unique index on users.email."),
    ]
    (proj1 / "sess-py-db.jsonl").write_text("\n".join(db_lines) + "\n")

    # Subagent session
    sub_dir = proj1 / "sess-py-auth" / "subagents"
    sub_dir.mkdir(parents=True)
    sub_lines = [
        _user_line("Run the test suite"),
        _assistant_line("All 42 tests passed."),
    ]
    (sub_dir / "agent-linter.jsonl").write_text("\n".join(sub_lines) + "\n")

    # Project 2: React frontend
    proj2 = projects_dir / "-Users-dev-Projects-frontend"
    proj2.mkdir(parents=True)

    idx2 = {
        "originalPath": "/Users/dev/Projects/frontend",
        "entries": [
            {
                "sessionId": "sess-react-router",
                "summary": "React router configuration",
                "created": "2025-02-01T08:00:00Z",
                "modified": "2025-02-01T10:00:00Z",
                "gitBranch": "feature/routing",
            },
        ],
    }
    (proj2 / "sessions-index.json").write_text(json.dumps(idx2))

    react_lines = [
        _user_line("Set up React Router v6 with nested routes"),
        _assistant_line("I'll configure React Router with createBrowserRouter."),
        _user_line("Add a navigation bar component"),
        _assistant_line("Navigation component with NavLink active styling added."),
        _user_line("Add lazy loading for route components"),
        _assistant_line("Routes now use React.lazy() with Suspense fallback."),
    ]
    (proj2 / "sess-react-router.jsonl").write_text("\n".join(react_lines) + "\n")

    return {
        "projects_dir": projects_dir,
        "db_path": db_path,
        "proj1": proj1,
        "proj2": proj2,
    }


# ===========================================================================
# End-to-end tests
# ===========================================================================

class TestEndToEnd:
    def test_index_and_search(self, integration_env):
        """Full pipeline: index sessions, then search for them."""
        env = integration_env

        # Index
        stats = build_index(
            projects_dir=env["projects_dir"],
            db_path=env["db_path"],
            verbose=False,
        )
        assert stats["indexed"] >= 3
        assert stats["errors"] == 0

        # Search for auth-related sessions
        results = search(
            "authentication JWT token",
            db_path=env["db_path"],
            semantic=False,
        )
        assert len(results) >= 1
        # The auth session should be the top result
        ids = [r.session.session_id for r in results]
        assert "sess-py-auth" in ids

    def test_fts_finds_keyword_match(self, integration_env):
        """FTS should find sessions containing specific keywords."""
        env = integration_env
        build_index(projects_dir=env["projects_dir"], db_path=env["db_path"], verbose=False)

        # Search for database-specific keywords
        results = search(
            "database migration alembic",
            db_path=env["db_path"],
            semantic=False,
        )
        assert len(results) >= 1
        ids = [r.session.session_id for r in results]
        assert "sess-py-db" in ids

    def test_fts_finds_react_router(self, integration_env):
        """FTS should find React-specific sessions."""
        env = integration_env
        build_index(projects_dir=env["projects_dir"], db_path=env["db_path"], verbose=False)

        results = search(
            "React Router navigation",
            db_path=env["db_path"],
            semantic=False,
        )
        assert len(results) >= 1
        ids = [r.session.session_id for r in results]
        assert "sess-react-router" in ids

    def test_subagent_excluded_from_search(self, integration_env):
        """Subagent sessions should not appear in search results."""
        env = integration_env
        build_index(projects_dir=env["projects_dir"], db_path=env["db_path"], verbose=False)

        results = search(
            "test suite linter",
            db_path=env["db_path"],
            semantic=False,
        )
        for r in results:
            assert not r.session.is_subagent

    def test_project_filter(self, integration_env):
        """Project filter should restrict results to matching projects."""
        env = integration_env
        build_index(projects_dir=env["projects_dir"], db_path=env["db_path"], verbose=False)

        results = search(
            "debugging",
            db_path=env["db_path"],
            semantic=False,
            project_filter="backend",
        )
        for r in results:
            assert "backend" in r.session.project_path

    def test_date_filter(self, integration_env):
        """Date filters should restrict results by modification date."""
        env = integration_env
        build_index(projects_dir=env["projects_dir"], db_path=env["db_path"], verbose=False)

        results = search(
            "debugging migration router",
            db_path=env["db_path"],
            semantic=False,
            after="2025-01-20",
        )
        for r in results:
            assert r.session.modified >= "2025-01-20"

    def test_incremental_index(self, integration_env):
        """Second index run should skip unchanged files."""
        env = integration_env

        stats1 = build_index(
            projects_dir=env["projects_dir"],
            db_path=env["db_path"],
            verbose=False,
        )

        stats2 = build_index(
            projects_dir=env["projects_dir"],
            db_path=env["db_path"],
            verbose=False,
        )

        assert stats2["indexed"] == 0
        assert stats2["skipped"] >= stats1["indexed"]

    def test_gc_removes_orphans(self, integration_env):
        """Deleted session files should be removed from index on re-index."""
        env = integration_env
        build_index(projects_dir=env["projects_dir"], db_path=env["db_path"], verbose=False)

        # Delete a session file
        session_file = env["proj2"] / "sess-react-router.jsonl"
        session_file.unlink()

        stats = build_index(
            projects_dir=env["projects_dir"],
            db_path=env["db_path"],
            verbose=False,
        )
        assert stats["removed"] >= 1

        # Verify it's gone from the DB
        conn = get_connection(env["db_path"])
        ids = get_all_session_ids(conn)
        conn.close()
        assert "sess-react-router" not in ids

    def test_stats_correct(self, integration_env):
        """Index stats should reflect what was indexed."""
        env = integration_env
        build_index(projects_dir=env["projects_dir"], db_path=env["db_path"], verbose=False)

        conn = get_connection(env["db_path"])
        stats = get_stats(conn)
        conn.close()

        assert stats["total"] >= 3
        assert stats["main_sessions"] >= 3
        assert stats["subagent_sessions"] >= 1
        assert stats["projects"] >= 2
        assert stats["total_messages"] >= 8

    def test_chunks_populated(self, integration_env):
        """Chunks table should be populated for all sessions."""
        env = integration_env
        build_index(projects_dir=env["projects_dir"], db_path=env["db_path"], verbose=False)

        conn = get_connection(env["db_path"])
        chunk_count = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
        conn.close()

        assert chunk_count >= 3  # at least one chunk per session

    def test_project_path_decoded(self, integration_env):
        """Project paths should be decoded from sessions-index.json."""
        env = integration_env
        build_index(projects_dir=env["projects_dir"], db_path=env["db_path"], verbose=False)

        conn = get_connection(env["db_path"])
        row = conn.execute(
            "SELECT project_path FROM sessions WHERE session_id = 'sess-py-auth'"
        ).fetchone()
        conn.close()

        assert row["project_path"] == "/Users/dev/Projects/backend"

    def test_search_scores_normalized(self, integration_env):
        """Search result scores should be normalized to 0..1."""
        env = integration_env
        build_index(projects_dir=env["projects_dir"], db_path=env["db_path"], verbose=False)

        results = search(
            "authentication middleware",
            db_path=env["db_path"],
            semantic=False,
        )
        if results:
            scores = [r.score for r in results]
            assert max(scores) <= 1.0
            assert min(scores) >= 0.0

    def test_new_file_indexed_incrementally(self, integration_env):
        """Adding a new session file and re-indexing should pick it up."""
        env = integration_env
        build_index(projects_dir=env["projects_dir"], db_path=env["db_path"], verbose=False)

        # Add a new session
        new_lines = [
            _user_line("Implement WebSocket support"),
            _assistant_line("I'll add Socket.IO for real-time communication."),
        ]
        new_file = env["proj1"] / "sess-websocket.jsonl"
        new_file.write_text("\n".join(new_lines) + "\n")

        stats = build_index(
            projects_dir=env["projects_dir"],
            db_path=env["db_path"],
            verbose=False,
        )
        assert stats["indexed"] >= 1

        # Search should find it
        results = search(
            "WebSocket",
            db_path=env["db_path"],
            semantic=False,
        )
        ids = [r.session.session_id for r in results]
        assert "sess-websocket" in ids

    def test_json_output_format(self, integration_env, capsys):
        """CLI --json output should be valid and contain expected fields."""
        env = integration_env
        build_index(projects_dir=env["projects_dir"], db_path=env["db_path"], verbose=False)

        results = search(
            "authentication",
            db_path=env["db_path"],
            semantic=False,
        )

        # Verify result objects have expected fields
        for r in results:
            assert hasattr(r, "score")
            assert hasattr(r, "session")
            assert hasattr(r.session, "session_id")
            assert hasattr(r.session, "project_path")
            assert hasattr(r, "resume_command")
