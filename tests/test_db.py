"""Tests for claude_recall.db."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from claude_recall.db import (
    build_session_chains,
    delete_session,
    get_all_session_ids,
    get_connection,
    get_related_sessions,
    get_session_mtime,
    get_stats,
    upsert_chunks,
    upsert_graph_edges,
    upsert_session,
    upsert_session_commands,
    upsert_session_files,
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
        assert row["value"] == "3"

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


# ===========================================================================
# Graph tables: session_files, session_commands, graph_edges, session_chains
# ===========================================================================

class TestGraphTablesExist:
    def test_session_files_table(self, db_conn):
        tables = {
            row[0]
            for row in db_conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        }
        assert "session_files" in tables

    def test_session_commands_table(self, db_conn):
        tables = {
            row[0]
            for row in db_conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        }
        assert "session_commands" in tables

    def test_graph_edges_table(self, db_conn):
        tables = {
            row[0]
            for row in db_conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        }
        assert "graph_edges" in tables

    def test_session_chains_table(self, db_conn):
        tables = {
            row[0]
            for row in db_conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        }
        assert "session_chains" in tables


class TestUpsertSessionFiles:
    def test_insert_files(self, db_conn, sample_session):
        upsert_session(db_conn, sample_session)
        files = [
            {"path": "src/auth.py", "name": "auth.py", "action": "edit"},
            {"path": "tests/test_auth.py", "name": "test_auth.py", "action": "edit"},
        ]
        upsert_session_files(db_conn, sample_session.session_id, files)
        db_conn.commit()

        rows = db_conn.execute(
            "SELECT * FROM session_files WHERE session_id = ?",
            (sample_session.session_id,),
        ).fetchall()
        assert len(rows) == 2
        names = {row["file_name"] for row in rows}
        assert "auth.py" in names
        assert "test_auth.py" in names

    def test_replace_files(self, db_conn, sample_session):
        upsert_session(db_conn, sample_session)
        upsert_session_files(db_conn, sample_session.session_id, [
            {"path": "old.py", "name": "old.py", "action": "edit"},
        ])
        db_conn.commit()

        upsert_session_files(db_conn, sample_session.session_id, [
            {"path": "new.py", "name": "new.py", "action": "edit"},
        ])
        db_conn.commit()

        rows = db_conn.execute(
            "SELECT * FROM session_files WHERE session_id = ?",
            (sample_session.session_id,),
        ).fetchall()
        assert len(rows) == 1
        assert rows[0]["file_name"] == "new.py"

    def test_empty_files(self, db_conn, sample_session):
        upsert_session(db_conn, sample_session)
        upsert_session_files(db_conn, sample_session.session_id, [])
        db_conn.commit()

        count = db_conn.execute(
            "SELECT COUNT(*) FROM session_files WHERE session_id = ?",
            (sample_session.session_id,),
        ).fetchone()[0]
        assert count == 0

    def test_search_by_file_name(self, db_conn, sample_session):
        upsert_session(db_conn, sample_session)
        upsert_session_files(db_conn, sample_session.session_id, [
            {"path": "src/auth.py", "name": "auth.py", "action": "edit"},
        ])
        db_conn.commit()

        rows = db_conn.execute(
            "SELECT * FROM session_files WHERE file_name = ?",
            ("auth.py",),
        ).fetchall()
        assert len(rows) == 1
        assert rows[0]["session_id"] == sample_session.session_id


class TestUpsertSessionCommands:
    def test_insert_commands(self, db_conn, sample_session):
        upsert_session(db_conn, sample_session)
        cmds = [
            {"command": "pytest tests/", "command_name": "pytest"},
            {"command": "git status", "command_name": "git"},
        ]
        upsert_session_commands(db_conn, sample_session.session_id, cmds)
        db_conn.commit()

        rows = db_conn.execute(
            "SELECT * FROM session_commands WHERE session_id = ?",
            (sample_session.session_id,),
        ).fetchall()
        assert len(rows) == 2
        names = {row["command_name"] for row in rows}
        assert "pytest" in names
        assert "git" in names

    def test_empty_commands(self, db_conn, sample_session):
        upsert_session(db_conn, sample_session)
        upsert_session_commands(db_conn, sample_session.session_id, [])
        db_conn.commit()

        count = db_conn.execute(
            "SELECT COUNT(*) FROM session_commands WHERE session_id = ?",
            (sample_session.session_id,),
        ).fetchone()[0]
        assert count == 0


class TestUpsertGraphEdges:
    def test_insert_edges(self, db_conn, sample_session):
        upsert_session(db_conn, sample_session)
        edges = [
            {
                "src_type": "session", "src_name": sample_session.session_id,
                "dst_type": "file", "dst_name": "auth.py",
                "rel": "edited",
            },
            {
                "src_type": "session", "src_name": sample_session.session_id,
                "dst_type": "command", "dst_name": "pytest",
                "rel": "ran",
            },
        ]
        upsert_graph_edges(db_conn, sample_session.session_id, edges)
        db_conn.commit()

        rows = db_conn.execute(
            "SELECT * FROM graph_edges WHERE session_id = ?",
            (sample_session.session_id,),
        ).fetchall()
        assert len(rows) == 2

    def test_replace_edges(self, db_conn, sample_session):
        upsert_session(db_conn, sample_session)
        upsert_graph_edges(db_conn, sample_session.session_id, [
            {
                "src_type": "session", "src_name": sample_session.session_id,
                "dst_type": "file", "dst_name": "old.py",
                "rel": "edited",
            },
        ])
        db_conn.commit()

        upsert_graph_edges(db_conn, sample_session.session_id, [
            {
                "src_type": "session", "src_name": sample_session.session_id,
                "dst_type": "file", "dst_name": "new.py",
                "rel": "edited",
            },
        ])
        db_conn.commit()

        rows = db_conn.execute(
            "SELECT * FROM graph_edges WHERE session_id = ?",
            (sample_session.session_id,),
        ).fetchall()
        assert len(rows) == 1
        assert rows[0]["target_name"] == "new.py"

    def test_empty_edges(self, db_conn, sample_session):
        upsert_session(db_conn, sample_session)
        upsert_graph_edges(db_conn, sample_session.session_id, [])
        db_conn.commit()

        count = db_conn.execute(
            "SELECT COUNT(*) FROM graph_edges WHERE session_id = ?",
            (sample_session.session_id,),
        ).fetchone()[0]
        assert count == 0


class TestBuildSessionChains:
    def test_chains_same_project_branch(self, db_conn):
        """Sessions in the same project/branch within 4h should chain."""
        s1 = Session(
            session_id="chain1",
            project_path="/test",
            project_dir="test",
            file_path="/tmp/chain1.jsonl",
            git_branch="main",
            created="2025-01-15T10:00:00Z",
            modified="2025-01-15T10:30:00Z",
            is_subagent=False,
        )
        s2 = Session(
            session_id="chain2",
            project_path="/test",
            project_dir="test",
            file_path="/tmp/chain2.jsonl",
            git_branch="main",
            created="2025-01-15T12:00:00Z",
            modified="2025-01-15T12:30:00Z",
            is_subagent=False,
        )
        upsert_session(db_conn, s1)
        upsert_session(db_conn, s2)
        db_conn.commit()

        build_session_chains(db_conn)

        rows = db_conn.execute(
            "SELECT * FROM session_chains ORDER BY chain_order"
        ).fetchall()
        assert len(rows) == 2
        # Both should share the same chain_id
        assert rows[0]["chain_id"] == rows[1]["chain_id"]
        assert rows[0]["chain_order"] == 0
        assert rows[1]["chain_order"] == 1

    def test_chains_different_projects(self, db_conn):
        """Sessions in different projects should have separate chains."""
        s1 = Session(
            session_id="proj_a",
            project_path="/test/a",
            project_dir="test-a",
            file_path="/tmp/a.jsonl",
            git_branch="main",
            created="2025-01-15T10:00:00Z",
            modified="2025-01-15T10:30:00Z",
            is_subagent=False,
        )
        s2 = Session(
            session_id="proj_b",
            project_path="/test/b",
            project_dir="test-b",
            file_path="/tmp/b.jsonl",
            git_branch="main",
            created="2025-01-15T10:00:00Z",
            modified="2025-01-15T10:30:00Z",
            is_subagent=False,
        )
        upsert_session(db_conn, s1)
        upsert_session(db_conn, s2)
        db_conn.commit()

        build_session_chains(db_conn)

        rows = db_conn.execute(
            "SELECT * FROM session_chains ORDER BY session_id"
        ).fetchall()
        assert len(rows) == 2
        assert rows[0]["chain_id"] != rows[1]["chain_id"]

    def test_chains_time_gap_breaks_chain(self, db_conn):
        """Sessions >4h apart should be in separate chains."""
        s1 = Session(
            session_id="gap1",
            project_path="/test",
            project_dir="test",
            file_path="/tmp/gap1.jsonl",
            git_branch="main",
            created="2025-01-15T10:00:00Z",
            modified="2025-01-15T10:30:00Z",
            is_subagent=False,
        )
        s2 = Session(
            session_id="gap2",
            project_path="/test",
            project_dir="test",
            file_path="/tmp/gap2.jsonl",
            git_branch="main",
            created="2025-01-15T20:00:00Z",
            modified="2025-01-15T20:30:00Z",
            is_subagent=False,
        )
        upsert_session(db_conn, s1)
        upsert_session(db_conn, s2)
        db_conn.commit()

        build_session_chains(db_conn)

        rows = db_conn.execute(
            "SELECT * FROM session_chains ORDER BY session_id"
        ).fetchall()
        assert len(rows) == 2
        assert rows[0]["chain_id"] != rows[1]["chain_id"]


class TestGetRelatedSessions:
    def test_finds_related_by_shared_files(self, db_conn):
        """Sessions editing the same file should be related."""
        s1 = Session(
            session_id="rel1",
            project_path="/test",
            project_dir="test",
            file_path="/tmp/rel1.jsonl",
            summary="Auth work",
            message_count=5,
            modified="2025-01-15T10:00:00Z",
            is_subagent=False,
        )
        s2 = Session(
            session_id="rel2",
            project_path="/test",
            project_dir="test",
            file_path="/tmp/rel2.jsonl",
            summary="More auth work",
            message_count=3,
            modified="2025-01-16T10:00:00Z",
            is_subagent=False,
        )
        upsert_session(db_conn, s1)
        upsert_session(db_conn, s2)

        # Both sessions edited auth.py
        upsert_graph_edges(db_conn, "rel1", [
            {"src_type": "session", "src_name": "rel1",
             "dst_type": "file", "dst_name": "auth.py", "rel": "edited"},
        ])
        upsert_graph_edges(db_conn, "rel2", [
            {"src_type": "session", "src_name": "rel2",
             "dst_type": "file", "dst_name": "auth.py", "rel": "edited"},
        ])
        db_conn.commit()

        related = get_related_sessions(db_conn, "rel1")
        assert len(related) == 1
        assert related[0]["session_id"] == "rel2"
        assert related[0]["shared_files"] == 1

    def test_no_related_sessions(self, db_conn, sample_session):
        upsert_session(db_conn, sample_session)
        db_conn.commit()

        related = get_related_sessions(db_conn, sample_session.session_id)
        assert len(related) == 0
