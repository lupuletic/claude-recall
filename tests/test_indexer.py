"""Tests for claude_recall.indexer."""

from __future__ import annotations

import time
from pathlib import Path
from unittest.mock import patch

import pytest

from claude_recall.db import get_all_session_ids, get_connection, get_session_mtime
from claude_recall.indexer import build_index


class TestBuildIndex:
    def test_discovers_and_indexes_sessions(self, projects_dir, db_path):
        """build_index should discover and index all valid session files."""
        stats = build_index(
            projects_dir=projects_dir,
            db_path=db_path,
            force=False,
            verbose=False,
        )
        assert stats["total_discovered"] == 4  # 3 main + 1 subagent
        assert stats["indexed"] >= 3  # at least the sessions with user messages
        assert stats["errors"] == 0

    def test_all_sessions_in_db(self, projects_dir, db_path):
        """After indexing, all sessions should be in the database."""
        build_index(projects_dir=projects_dir, db_path=db_path, verbose=False)

        conn = get_connection(db_path)
        ids = get_all_session_ids(conn)
        conn.close()

        assert "session-001" in ids
        assert "session-002" in ids
        assert "session-003" in ids
        assert "agent-sub1" in ids

    def test_incremental_skips_unchanged(self, projects_dir, db_path):
        """Second run should skip already-indexed, unchanged files."""
        stats1 = build_index(
            projects_dir=projects_dir,
            db_path=db_path,
            verbose=False,
        )
        indexed_first = stats1["indexed"]

        stats2 = build_index(
            projects_dir=projects_dir,
            db_path=db_path,
            verbose=False,
        )
        assert stats2["skipped"] >= indexed_first
        assert stats2["indexed"] == 0

    def test_force_reindex_processes_all(self, projects_dir, db_path):
        """Force reindex should process all files again."""
        build_index(
            projects_dir=projects_dir,
            db_path=db_path,
            verbose=False,
        )

        stats = build_index(
            projects_dir=projects_dir,
            db_path=db_path,
            force=True,
            verbose=False,
        )
        # With force, all discovered sessions should be re-indexed
        assert stats["indexed"] >= 3

    def test_empty_sessions_skipped(self, projects_dir, db_path, tmp_path):
        """Sessions with no user messages should be skipped."""
        import json

        # Create a session file with only system messages
        proj = projects_dir / "-Users-test-Projects-empty"
        proj.mkdir()
        empty_session = json.dumps({
            "type": "system",
            "message": {"role": "system", "content": "init"},
        })
        (proj / "empty-session.jsonl").write_text(empty_session + "\n")

        stats = build_index(
            projects_dir=projects_dir,
            db_path=db_path,
            verbose=False,
        )
        # The empty session should be skipped
        conn = get_connection(db_path)
        ids = get_all_session_ids(conn)
        conn.close()
        assert "empty-session" not in ids

    def test_subagent_detection(self, projects_dir, db_path):
        """Subagent sessions should be marked as such."""
        build_index(
            projects_dir=projects_dir,
            db_path=db_path,
            verbose=False,
        )

        conn = get_connection(db_path)
        row = conn.execute(
            "SELECT * FROM sessions WHERE session_id = 'agent-sub1'"
        ).fetchone()
        conn.close()

        assert row is not None
        assert row["is_subagent"] == 1
        assert row["parent_session"] == "session-001"

    def test_project_path_decoded(self, projects_dir, db_path):
        """Project path should be decoded from sessions-index.json."""
        build_index(
            projects_dir=projects_dir,
            db_path=db_path,
            verbose=False,
        )

        conn = get_connection(db_path)
        row = conn.execute(
            "SELECT * FROM sessions WHERE session_id = 'session-001'"
        ).fetchone()
        conn.close()

        # sessions-index.json has originalPath = /Users/test/Projects/myapp
        assert row["project_path"] == "/Users/test/Projects/myapp"

    def test_gc_removes_orphaned(self, projects_dir, db_path):
        """Removed session files should be cleaned up on re-index."""
        build_index(
            projects_dir=projects_dir,
            db_path=db_path,
            verbose=False,
        )

        # Delete a session file
        session_file = projects_dir / "-Users-test-Projects-webapp" / "session-003.jsonl"
        session_file.unlink()

        stats = build_index(
            projects_dir=projects_dir,
            db_path=db_path,
            verbose=False,
        )
        assert stats["removed"] >= 1

        conn = get_connection(db_path)
        ids = get_all_session_ids(conn)
        conn.close()
        assert "session-003" not in ids

    def test_modified_file_reindexed(self, projects_dir, db_path):
        """A file with changed mtime should be re-indexed."""
        build_index(
            projects_dir=projects_dir,
            db_path=db_path,
            verbose=False,
        )

        # Touch the file to update mtime
        import json
        import os

        session_file = projects_dir / "-Users-test-Projects-myapp" / "session-001.jsonl"
        # Append a new message
        with open(session_file, "a") as f:
            f.write(json.dumps({
                "type": "user",
                "message": {"role": "user", "content": "new question added"},
            }) + "\n")

        # Ensure mtime is different (on some filesystems, 1s resolution)
        future_time = time.time() + 10
        os.utime(session_file, (future_time, future_time))

        stats = build_index(
            projects_dir=projects_dir,
            db_path=db_path,
            verbose=False,
        )
        assert stats["indexed"] >= 1

    def test_chunks_stored(self, projects_dir, db_path):
        """Chunks should be stored in the chunks table."""
        build_index(
            projects_dir=projects_dir,
            db_path=db_path,
            verbose=False,
        )

        conn = get_connection(db_path)
        chunk_count = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
        conn.close()

        assert chunk_count > 0

    @patch("claude_recall.indexer.has_semantic", return_value=False)
    def test_no_semantic_deps(self, mock_semantic, projects_dir, db_path):
        """Should work fine without semantic dependencies."""
        stats = build_index(
            projects_dir=projects_dir,
            db_path=db_path,
            verbose=False,
        )
        assert stats["indexed"] >= 3
        assert stats["embeddings"] == 0

    def test_sessions_index_metadata_used(self, projects_dir, db_path):
        """Metadata from sessions-index.json should be incorporated."""
        build_index(
            projects_dir=projects_dir,
            db_path=db_path,
            verbose=False,
        )

        conn = get_connection(db_path)
        row = conn.execute(
            "SELECT * FROM sessions WHERE session_id = 'session-001'"
        ).fetchone()
        conn.close()

        assert row["summary"] == "Auth middleware debugging"
        assert row["git_branch"] == "fix/auth"
        assert row["created"] == "2025-01-15T10:00:00Z"
