"""Tests for claude_recall.db."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from claude_recall.db import (
    delete_session,
    get_all_session_ids,
    get_connection,
    get_session_mtime,
    get_stats,
    upsert_chunks,
    upsert_session,
)
from claude_recall.models import Session


# ===========================================================================
# get_connection & schema
# ===========================================================================

class TestGetConnection:
    def test_creates_db_file(self, db_path):
        conn = get_connection(db_path)
        assert db_path.exists()
        conn.close()

    def test_creates_parent_dirs(self, tmp_path):
        nested = tmp_path / "a" / "b" / "test.db"
        conn = get_connection(nested)
        assert nested.exists()
        conn.close()

    def test_returns_row_factory(self, db_path):
        conn = get_connection(db_path)
        assert conn.row_factory == sqlite3.Row
        conn.close()

    def test_wal_mode(self, db_path):
        conn = get_connection(db_path)
        mode = conn.execute("PRAGMA journal_mode").fetchone()[0]
        assert mode == "wal"
        conn.close()


class TestSchema:
    def test_sessions_table_exists(self, db_conn):
        tables = {
            row[0]
            for row in db_conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        }
        assert "sessions" in tables

    def test_chunks_table_exists(self, db_conn):
        tables = {
            row[0]
            for row in db_conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        }
        assert "chunks" in tables

    def test_sessions_fts_table_exists(self, db_conn):
        tables = {
            row[0]
            for row in db_conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        }
        assert "sessions_fts" in tables

    def test_meta_table_exists(self, db_conn):
        tables = {
            row[0]
            for row in db_conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        }
        assert "meta" in tables

    def test_schema_version_set(self, db_conn):
        row = db_conn.execute(
            "SELECT value FROM meta WHERE key = 'schema_version'"
        ).fetchone()
        assert row is not None
        assert row["value"] == "2"

    def test_sessions_columns(self, db_conn):
        cols = {
            row[1]
            for row in db_conn.execute("PRAGMA table_info(sessions)").fetchall()
        }
        expected = {
            "session_id", "project_path", "project_dir", "file_path",
            "summary", "first_prompt", "first_reply", "last_prompt", "last_reply",
            "messages_text", "git_branch", "message_count", "file_size",
            "created", "modified", "mtime", "is_subagent", "parent_session",
        }
        assert expected.issubset(cols)

    def test_chunks_columns(self, db_conn):
        cols = {
            row[1]
            for row in db_conn.execute("PRAGMA table_info(chunks)").fetchall()
        }
        expected = {"chunk_id", "session_id", "chunk_index", "chunk_text"}
        assert expected.issubset(cols)


# ===========================================================================
# upsert_session
# ===========================================================================

class TestUpsertSession:
    def test_insert(self, db_conn, sample_session):
        upsert_session(db_conn, sample_session)
        db_conn.commit()

        row = db_conn.execute(
            "SELECT * FROM sessions WHERE session_id = ?",
            (sample_session.session_id,),
        ).fetchone()
        assert row is not None
        assert row["session_id"] == "abc123"
        assert row["project_path"] == "/Users/testuser/Projects/myapp"
        assert row["first_prompt"] == "Help me debug the auth middleware"
        assert row["message_count"] == 5
        assert row["is_subagent"] == 0

    def test_update(self, db_conn, sample_session):
        upsert_session(db_conn, sample_session)
        db_conn.commit()

        # Update the session
        sample_session.summary = "Updated summary"
        sample_session.message_count = 10
        upsert_session(db_conn, sample_session)
        db_conn.commit()

        row = db_conn.execute(
            "SELECT * FROM sessions WHERE session_id = ?",
            (sample_session.session_id,),
        ).fetchone()
        assert row["summary"] == "Updated summary"
        assert row["message_count"] == 10

    def test_upsert_preserves_single_row(self, db_conn, sample_session):
        upsert_session(db_conn, sample_session)
        upsert_session(db_conn, sample_session)
        db_conn.commit()

        count = db_conn.execute("SELECT COUNT(*) FROM sessions").fetchone()[0]
        assert count == 1


# ===========================================================================
# upsert_chunks
# ===========================================================================

class TestUpsertChunks:
    def test_insert_chunks(self, db_conn, sample_session):
        upsert_session(db_conn, sample_session)
        chunks = ["chunk one text", "chunk two text"]
        upsert_chunks(db_conn, sample_session.session_id, chunks)
        db_conn.commit()

        rows = db_conn.execute(
            "SELECT * FROM chunks WHERE session_id = ? ORDER BY chunk_index",
            (sample_session.session_id,),
        ).fetchall()
        assert len(rows) == 2
        assert rows[0]["chunk_text"] == "chunk one text"
        assert rows[0]["chunk_index"] == 0
        assert rows[1]["chunk_text"] == "chunk two text"
        assert rows[1]["chunk_index"] == 1

    def test_replace_chunks(self, db_conn, sample_session):
        upsert_session(db_conn, sample_session)
        upsert_chunks(db_conn, sample_session.session_id, ["old chunk"])
        db_conn.commit()

        upsert_chunks(db_conn, sample_session.session_id, ["new chunk 1", "new chunk 2"])
        db_conn.commit()

        rows = db_conn.execute(
            "SELECT * FROM chunks WHERE session_id = ?",
            (sample_session.session_id,),
        ).fetchall()
        assert len(rows) == 2
        texts = {row["chunk_text"] for row in rows}
        assert "new chunk 1" in texts
        assert "new chunk 2" in texts
        assert "old chunk" not in texts

    def test_empty_chunks(self, db_conn, sample_session):
        upsert_session(db_conn, sample_session)
        upsert_chunks(db_conn, sample_session.session_id, [])
        db_conn.commit()

        count = db_conn.execute(
            "SELECT COUNT(*) FROM chunks WHERE session_id = ?",
            (sample_session.session_id,),
        ).fetchone()[0]
        assert count == 0


# ===========================================================================
# get_session_mtime
# ===========================================================================

class TestGetSessionMtime:
    def test_returns_mtime(self, db_conn, sample_session):
        upsert_session(db_conn, sample_session)
        db_conn.commit()

        mtime = get_session_mtime(db_conn, sample_session.session_id)
        assert mtime is not None
        assert abs(mtime - sample_session.mtime) < 0.01

    def test_returns_none_for_missing(self, db_conn):
        assert get_session_mtime(db_conn, "nonexistent") is None


# ===========================================================================
# get_all_session_ids
# ===========================================================================

class TestGetAllSessionIds:
    def test_empty_db(self, db_conn):
        assert get_all_session_ids(db_conn) == set()

    def test_returns_all_ids(self, populated_db):
        ids = get_all_session_ids(populated_db)
        assert ids == {"abc123", "def456", "sub789"}


# ===========================================================================
# delete_session
# ===========================================================================

class TestDeleteSession:
    def test_delete_existing(self, db_conn, sample_session):
        upsert_session(db_conn, sample_session)
        upsert_chunks(db_conn, sample_session.session_id, ["chunk"])
        db_conn.commit()

        delete_session(db_conn, sample_session.session_id)
        db_conn.commit()

        row = db_conn.execute(
            "SELECT * FROM sessions WHERE session_id = ?",
            (sample_session.session_id,),
        ).fetchone()
        assert row is None

    def test_cascade_deletes_chunks(self, db_conn, sample_session):
        """Foreign key cascade should delete chunks when session is deleted."""
        # Enable foreign keys (needed for CASCADE)
        db_conn.execute("PRAGMA foreign_keys = ON")
        upsert_session(db_conn, sample_session)
        upsert_chunks(db_conn, sample_session.session_id, ["chunk1", "chunk2"])
        db_conn.commit()

        delete_session(db_conn, sample_session.session_id)
        db_conn.commit()

        count = db_conn.execute(
            "SELECT COUNT(*) FROM chunks WHERE session_id = ?",
            (sample_session.session_id,),
        ).fetchone()[0]
        assert count == 0

    def test_delete_nonexistent(self, db_conn):
        # Should not raise
        delete_session(db_conn, "nonexistent")


# ===========================================================================
# get_stats
# ===========================================================================

class TestGetStats:
    def test_empty_db(self, db_conn):
        stats = get_stats(db_conn)
        assert stats["total"] == 0

    def test_populated_db(self, populated_db):
        stats = get_stats(populated_db)
        assert stats["total"] == 3
        assert stats["main_sessions"] == 2
        assert stats["subagent_sessions"] == 1
        assert stats["projects"] == 2
        assert stats["total_messages"] == 14  # 5 + 8 + 1

    def test_total_size(self, populated_db):
        stats = get_stats(populated_db)
        assert stats["total_size"] == 2048 + 4096 + 512

    def test_date_range(self, populated_db):
        stats = get_stats(populated_db)
        assert stats["earliest"] == "2025-01-15T10:00:00Z"
        assert stats["latest"] == "2025-02-01T10:00:00Z"


# ===========================================================================
# FTS triggers
# ===========================================================================

class TestFtsTriggers:
    def test_fts_populated_on_insert(self, db_conn, sample_session):
        upsert_session(db_conn, sample_session)
        db_conn.commit()

        rows = db_conn.execute(
            "SELECT * FROM sessions_fts WHERE sessions_fts MATCH ?",
            ('"auth middleware"',),
        ).fetchall()
        assert len(rows) >= 1

    def test_fts_updated_on_update(self, db_conn, sample_session):
        upsert_session(db_conn, sample_session)
        db_conn.commit()

        # Update the session with new content
        sample_session.first_prompt = "Help with database migration"
        sample_session.messages_text = "database migration schema"
        upsert_session(db_conn, sample_session)
        db_conn.commit()

        # Use the JOIN approach (same as the searcher) to verify FTS is updated.
        # Direct FTS queries on content-synced tables can be fragile after
        # INSERT OR REPLACE, so we use the JOIN to validate.
        new_results = db_conn.execute(
            """SELECT s.session_id
               FROM sessions_fts
               JOIN sessions s ON s.rowid = sessions_fts.rowid
               WHERE sessions_fts MATCH ?""",
            ('"database migration"',),
        ).fetchall()
        assert len(new_results) >= 1
        assert new_results[0]["session_id"] == sample_session.session_id

    def test_fts_cleaned_on_delete(self, db_conn, sample_session):
        upsert_session(db_conn, sample_session)
        db_conn.commit()

        delete_session(db_conn, sample_session.session_id)
        db_conn.commit()

        rows = db_conn.execute(
            "SELECT * FROM sessions_fts WHERE sessions_fts MATCH ?",
            ('"auth middleware"',),
        ).fetchall()
        assert len(rows) == 0

    def test_fts_search_summary(self, db_conn, sample_session):
        upsert_session(db_conn, sample_session)
        db_conn.commit()

        rows = db_conn.execute(
            "SELECT * FROM sessions_fts WHERE sessions_fts MATCH ?",
            ('"Debugging"',),
        ).fetchall()
        assert len(rows) >= 1

    def test_fts_search_messages_text(self, db_conn, sample_session):
        upsert_session(db_conn, sample_session)
        db_conn.commit()

        rows = db_conn.execute(
            "SELECT * FROM sessions_fts WHERE sessions_fts MATCH ?",
            ('"tests"',),
        ).fetchall()
        assert len(rows) >= 1
