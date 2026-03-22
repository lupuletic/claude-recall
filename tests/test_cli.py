"""Tests for claude_recall.cli."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from claude_recall.cli import main


@pytest.fixture(autouse=True)
def isolate_env(tmp_path, monkeypatch):
    """Isolate all CLI tests from real user data."""
    db_path = tmp_path / "test.db"
    projects_dir = tmp_path / "projects"
    projects_dir.mkdir()
    config_path = tmp_path / "config.json"
    hooks_marker = tmp_path / ".hooks-installed"

    monkeypatch.setattr("claude_recall.cli.DB_PATH", db_path)
    monkeypatch.setattr("claude_recall.cli.PROJECTS_DIR", projects_dir)
    monkeypatch.setattr("claude_recall.cli.HOOKS_MARKER", hooks_marker)
    monkeypatch.setattr("claude_recall.config.CONFIG_PATH", config_path)
    monkeypatch.setattr("claude_recall.db.DB_PATH", db_path)
    monkeypatch.setattr("claude_recall.utils.PROJECTS_DIR", projects_dir)

    return {
        "db_path": db_path,
        "projects_dir": projects_dir,
        "config_path": config_path,
    }


# ===========================================================================
# --version
# ===========================================================================

class TestVersion:
    def test_version_flag(self, capsys):
        with pytest.raises(SystemExit) as exc_info:
            main(["--version"])
        assert exc_info.value.code == 0
        out = capsys.readouterr().out
        assert "claude-recall" in out


# ===========================================================================
# index subcommand
# ===========================================================================

class TestIndexCommand:
    def test_index_runs(self, capsys, isolate_env):
        """'index' subcommand should run without error."""
        # Suppress ensure_models_downloaded
        with patch("claude_recall.cli.build_index", return_value={
            "indexed": 0, "skipped": 0, "removed": 0, "errors": 0,
            "embeddings": 0, "elapsed": 0.1, "total_discovered": 0,
        }):
            main(["index", "--quiet"])

    def test_index_force(self, isolate_env):
        with patch("claude_recall.cli.build_index", return_value={
            "indexed": 5, "skipped": 0, "removed": 0, "errors": 0,
            "embeddings": 0, "elapsed": 0.5, "total_discovered": 5,
        }) as mock_build:
            main(["index", "--force", "--quiet"])
            _, kwargs = mock_build.call_args
            assert kwargs.get("force") is True


# ===========================================================================
# search (direct query routing)
# ===========================================================================

class TestSearchCommand:
    @patch("claude_recall.cli._first_run_setup")
    @patch("claude_recall.cli.search", return_value=[])
    def test_direct_query(self, mock_search, mock_setup, capsys, isolate_env):
        """Positional args without subcommand should route to search."""
        main(["auth", "middleware", "--no-tui"])
        mock_search.assert_called_once()
        call_kwargs = mock_search.call_args
        assert call_kwargs[1]["query"] == "auth middleware"

    @patch("claude_recall.cli._first_run_setup")
    @patch("claude_recall.cli.search", return_value=[])
    def test_search_subcommand(self, mock_search, mock_setup, capsys, isolate_env):
        """'search' subcommand should strip the command and search."""
        main(["search", "debug", "auth", "--no-tui"])
        mock_search.assert_called_once()
        assert mock_search.call_args[1]["query"] == "debug auth"

    @patch("claude_recall.cli._first_run_setup")
    @patch("claude_recall.cli.search", return_value=[])
    def test_s_alias(self, mock_search, mock_setup, capsys, isolate_env):
        """'s' should be an alias for 'search'."""
        main(["s", "test", "query", "--no-tui"])
        mock_search.assert_called_once()
        assert mock_search.call_args[1]["query"] == "test query"


# ===========================================================================
# --json output
# ===========================================================================

class TestJsonOutput:
    @patch("claude_recall.cli._first_run_setup")
    @patch("claude_recall.cli.search")
    def test_json_output_valid(self, mock_search, mock_setup, capsys, isolate_env):
        """--json should output valid JSON."""
        from claude_recall.models import SearchResult, Session

        mock_search.return_value = [
            SearchResult(
                session=Session(
                    session_id="test1",
                    project_path="/test",
                    project_dir="test",
                    file_path="/tmp/test.jsonl",
                    summary="Test session",
                    first_prompt="hello",
                    message_count=3,
                    file_size=100,
                    modified="2025-01-01T00:00:00Z",
                ),
                score=0.95,
                snippets=["test snippet"],
            )
        ]
        main(["auth", "--json"])
        output = capsys.readouterr().out
        data = json.loads(output)
        assert isinstance(data, list)
        assert len(data) == 1
        assert data[0]["session_id"] == "test1"
        assert data[0]["score"] == 0.95

    @patch("claude_recall.cli._first_run_setup")
    @patch("claude_recall.cli.search", return_value=[])
    def test_json_empty_results(self, mock_search, mock_setup, capsys, isolate_env):
        main(["nonsense", "--json"])
        output = capsys.readouterr().out
        data = json.loads(output)
        assert data == []


# ===========================================================================
# --no-tui
# ===========================================================================

class TestNoTui:
    @patch("claude_recall.cli._first_run_setup")
    @patch("claude_recall.cli.search", return_value=[])
    def test_no_tui_flag(self, mock_search, mock_setup, capsys, isolate_env):
        """--no-tui should output plain text."""
        main(["test", "query", "--no-tui"])
        out = capsys.readouterr().out
        assert "No sessions found" in out


# ===========================================================================
# filters passed through
# ===========================================================================

class TestFilterPassthrough:
    @patch("claude_recall.cli._first_run_setup")
    @patch("claude_recall.cli.search", return_value=[])
    def test_project_filter(self, mock_search, mock_setup, isolate_env):
        main(["test", "--project", "myapp", "--no-tui"])
        assert mock_search.call_args[1]["project_filter"] == "myapp"

    @patch("claude_recall.cli._first_run_setup")
    @patch("claude_recall.cli.search", return_value=[])
    def test_after_filter(self, mock_search, mock_setup, isolate_env):
        main(["test", "--after", "2025-01-01", "--no-tui"])
        assert mock_search.call_args[1]["after"] == "2025-01-01"

    @patch("claude_recall.cli._first_run_setup")
    @patch("claude_recall.cli.search", return_value=[])
    def test_before_filter(self, mock_search, mock_setup, isolate_env):
        main(["test", "--before", "2025-12-31", "--no-tui"])
        assert mock_search.call_args[1]["before"] == "2025-12-31"

    @patch("claude_recall.cli._first_run_setup")
    @patch("claude_recall.cli.search", return_value=[])
    def test_limit(self, mock_search, mock_setup, isolate_env):
        main(["test", "-n", "5", "--no-tui"])
        assert mock_search.call_args[1]["limit"] == 5


# ===========================================================================
# info subcommand
# ===========================================================================

class TestInfoCommand:
    def test_info_no_db(self, capsys, isolate_env):
        """Info with no DB should print a helpful message."""
        main(["info"])
        out = capsys.readouterr().out
        assert "No index found" in out

    def test_info_with_db(self, capsys, isolate_env):
        """Info with a DB should show stats."""
        from claude_recall.db import get_connection, upsert_session
        from claude_recall.models import Session

        db_path = isolate_env["db_path"]
        conn = get_connection(db_path)
        s = Session(
            session_id="info1",
            project_path="/test",
            project_dir="test",
            file_path="/tmp/info1.jsonl",
            message_count=5,
            file_size=1024,
            created="2025-01-01T00:00:00Z",
            modified="2025-01-01T01:00:00Z",
        )
        upsert_session(conn, s)
        conn.commit()
        conn.close()

        main(["info"])
        out = capsys.readouterr().out
        assert "Sessions:" in out
        assert "1" in out


# ===========================================================================
# config subcommand
# ===========================================================================

class TestConfigCommand:
    def test_config_view(self, capsys, isolate_env):
        main(["config"])
        out = capsys.readouterr().out
        assert "search_mode" in out

    def test_config_set(self, capsys, isolate_env):
        main(["config", "limit", "25"])
        out = capsys.readouterr().out
        assert "25" in out

    def test_config_set_invalid(self, capsys, isolate_env):
        with pytest.raises(SystemExit):
            main(["config", "search_mode", "bogus"])
