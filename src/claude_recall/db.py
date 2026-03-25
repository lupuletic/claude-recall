"""SQLite database management for claude-recall."""

from __future__ import annotations

import sqlite3
from pathlib import Path

from claude_recall.models import Session

DB_DIR = Path.home() / ".claude-recall"
DB_PATH = DB_DIR / "index.db"

SCHEMA_VERSION = 3

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS meta (
    key TEXT PRIMARY KEY,
    value TEXT
);

CREATE TABLE IF NOT EXISTS sessions (
    session_id TEXT PRIMARY KEY,
    project_path TEXT,
    project_dir TEXT,
    file_path TEXT,
    summary TEXT,
    first_prompt TEXT,
    first_reply TEXT,
    last_prompt TEXT,
    last_reply TEXT,
    messages_text TEXT,
    git_branch TEXT,
    message_count INTEGER DEFAULT 0,
    file_size INTEGER DEFAULT 0,
    created TEXT,
    modified TEXT,
    mtime REAL DEFAULT 0,
    is_subagent INTEGER DEFAULT 0,
    parent_session TEXT,
    files_modified TEXT,
    commands_run TEXT,
    git_branch_detected TEXT
);

-- Chunks table for multi-embed semantic search
CREATE TABLE IF NOT EXISTS chunks (
    chunk_id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL REFERENCES sessions(session_id) ON DELETE CASCADE,
    chunk_index INTEGER NOT NULL,
    chunk_text TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_chunks_session ON chunks(session_id);

-- Normalized file edits (searchable by exact file path)
CREATE TABLE IF NOT EXISTS session_files (
    session_id TEXT NOT NULL,
    file_path TEXT NOT NULL,
    file_name TEXT NOT NULL,
    action TEXT NOT NULL DEFAULT 'edit',
    PRIMARY KEY (session_id, file_path, action)
);
CREATE INDEX IF NOT EXISTS idx_sf_name ON session_files(file_name);
CREATE INDEX IF NOT EXISTS idx_sf_path ON session_files(file_path);

-- Normalized commands
CREATE TABLE IF NOT EXISTS session_commands (
    session_id TEXT NOT NULL,
    command TEXT NOT NULL,
    command_name TEXT NOT NULL,
    PRIMARY KEY (session_id, command)
);
CREATE INDEX IF NOT EXISTS idx_sc_name ON session_commands(command_name);

-- Knowledge graph edges
CREATE TABLE IF NOT EXISTS graph_edges (
    source_type TEXT NOT NULL,
    source_name TEXT NOT NULL,
    target_type TEXT NOT NULL,
    target_name TEXT NOT NULL,
    relationship TEXT NOT NULL,
    session_id TEXT NOT NULL,
    weight REAL DEFAULT 1.0
);
CREATE INDEX IF NOT EXISTS idx_ge_source ON graph_edges(source_type, source_name);
CREATE INDEX IF NOT EXISTS idx_ge_target ON graph_edges(target_type, target_name);
CREATE INDEX IF NOT EXISTS idx_ge_session ON graph_edges(session_id);

-- Session chains (related sessions in same project)
CREATE TABLE IF NOT EXISTS session_chains (
    session_id TEXT NOT NULL,
    chain_id TEXT NOT NULL,
    chain_order INTEGER NOT NULL,
    PRIMARY KEY (session_id)
);
CREATE INDEX IF NOT EXISTS idx_chain ON session_chains(chain_id);

CREATE VIRTUAL TABLE IF NOT EXISTS sessions_fts USING fts5(
    summary,
    first_prompt,
    last_prompt,
    messages_text,
    project_path,
    content=sessions,
    content_rowid=rowid,
    tokenize='porter unicode61'
);

-- Triggers to keep FTS in sync with the sessions table
CREATE TRIGGER IF NOT EXISTS sessions_ai AFTER INSERT ON sessions BEGIN
    INSERT INTO sessions_fts(rowid, summary, first_prompt, last_prompt, messages_text, project_path)
    VALUES (new.rowid, new.summary, new.first_prompt, new.last_prompt, new.messages_text, new.project_path);
END;

CREATE TRIGGER IF NOT EXISTS sessions_ad AFTER DELETE ON sessions BEGIN
    INSERT INTO sessions_fts(sessions_fts, rowid, summary, first_prompt, last_prompt, messages_text, project_path)
    VALUES ('delete', old.rowid, old.summary, old.first_prompt, old.last_prompt, old.messages_text, old.project_path);
END;

CREATE TRIGGER IF NOT EXISTS sessions_au AFTER UPDATE ON sessions BEGIN
    INSERT INTO sessions_fts(sessions_fts, rowid, summary, first_prompt, last_prompt, messages_text, project_path)
    VALUES ('delete', old.rowid, old.summary, old.first_prompt, old.last_prompt, old.messages_text, old.project_path);
    INSERT INTO sessions_fts(rowid, summary, first_prompt, last_prompt, messages_text, project_path)
    VALUES (new.rowid, new.summary, new.first_prompt, new.last_prompt, new.messages_text, new.project_path);
END;
"""


def get_connection(db_path: Path = DB_PATH) -> sqlite3.Connection:
    """Get a database connection, creating the DB if needed."""
    db_path.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(str(db_path))
    try:
        conn.enable_load_extension(True)
    except AttributeError:
        # Python built without SQLITE_ENABLE_LOAD_EXTENSION
        # Semantic search won't be available, but FTS5 still works
        pass
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys=ON")
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA busy_timeout=10000")  # wait up to 10s for locks

    _ensure_schema(conn)
    return conn


def _ensure_schema(conn: sqlite3.Connection) -> None:
    """Create tables if they don't exist, run migrations if needed."""
    conn.executescript(SCHEMA_SQL)

    version = _get_meta(conn, "schema_version")
    if version is None:
        _set_meta(conn, "schema_version", str(SCHEMA_VERSION))


def _get_meta(conn: sqlite3.Connection, key: str) -> str | None:
    row = conn.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
    return row["value"] if row else None


def _set_meta(conn: sqlite3.Connection, key: str, value: str) -> None:
    conn.execute(
        "INSERT OR REPLACE INTO meta (key, value) VALUES (?, ?)",
        (key, value),
    )
    conn.commit()


def upsert_session(conn: sqlite3.Connection, session: Session) -> None:
    """Insert or update a session in the database."""
    conn.execute(
        """INSERT OR REPLACE INTO sessions
           (session_id, project_path, project_dir, file_path,
            summary, first_prompt, first_reply, last_prompt, last_reply,
            messages_text, git_branch, message_count, file_size,
            created, modified, mtime, is_subagent, parent_session,
            files_modified, commands_run, git_branch_detected)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            session.session_id,
            session.project_path,
            session.project_dir,
            session.file_path,
            session.summary,
            session.first_prompt,
            session.first_reply,
            session.last_prompt,
            session.last_reply,
            session.messages_text,
            session.git_branch,
            session.message_count,
            session.file_size,
            session.created,
            session.modified,
            session.mtime,
            int(session.is_subagent),
            session.parent_session,
            session.files_modified,
            session.commands_run,
            session.git_branch_detected,
        ),
    )


def upsert_chunks(conn: sqlite3.Connection, session_id: str, chunks: list[str]) -> None:
    """Store conversation chunks for a session (replaces existing)."""
    # Delete orphaned vector entries before deleting chunks
    try:
        conn.execute(
            """DELETE FROM chunks_vec
               WHERE chunk_rowid IN (
                   SELECT chunk_id FROM chunks WHERE session_id = ?
               )""",
            (session_id,),
        )
    except Exception:
        pass  # chunks_vec may not exist yet

    conn.execute("DELETE FROM chunks WHERE session_id = ?", (session_id,))
    conn.executemany(
        "INSERT INTO chunks (session_id, chunk_index, chunk_text) VALUES (?, ?, ?)",
        [(session_id, i, text) for i, text in enumerate(chunks)],
    )


def upsert_session_files(conn: sqlite3.Connection, session_id: str, files: list[dict]) -> None:
    """Store normalized file edits. files = [{"path": "...", "name": "...", "action": "edit"}]"""
    conn.execute("DELETE FROM session_files WHERE session_id = ?", (session_id,))
    if files:
        conn.executemany(
            "INSERT OR IGNORE INTO session_files (session_id, file_path, file_name, action) VALUES (?, ?, ?, ?)",
            [(session_id, f["path"], f["name"], f["action"]) for f in files],
        )


def upsert_session_commands(conn: sqlite3.Connection, session_id: str, commands: list[dict]) -> None:
    """Store normalized commands. commands = [{"command": "...", "command_name": "..."}]"""
    conn.execute("DELETE FROM session_commands WHERE session_id = ?", (session_id,))
    if commands:
        conn.executemany(
            "INSERT OR IGNORE INTO session_commands (session_id, command, command_name) VALUES (?, ?, ?)",
            [(session_id, c["command"], c["command_name"]) for c in commands],
        )


def upsert_graph_edges(conn: sqlite3.Connection, session_id: str, edges: list[dict]) -> None:
    """Store knowledge graph edges."""
    conn.execute("DELETE FROM graph_edges WHERE session_id = ?", (session_id,))
    if edges:
        conn.executemany(
            "INSERT INTO graph_edges (source_type, source_name, target_type, target_name, relationship, session_id) VALUES (?, ?, ?, ?, ?, ?)",
            [(e["src_type"], e["src_name"], e["dst_type"], e["dst_name"], e["rel"], session_id) for e in edges],
        )


def build_session_chains(conn: sqlite3.Connection) -> None:
    """Link sessions in the same project + branch within a 4-hour window."""
    conn.execute("DELETE FROM session_chains")
    rows = conn.execute("""
        SELECT session_id, project_dir, git_branch, created
        FROM sessions WHERE is_subagent = 0
        ORDER BY project_dir, git_branch, created
    """).fetchall()

    chain_id = None
    chain_order = 0
    prev = None

    for row in rows:
        key = (row["project_dir"], row["git_branch"] or "")
        start_new_chain = True

        if prev and prev["key"] == key and prev["created"] and row["created"]:
            # Check time gap — within 4 hours = same chain
            try:
                from datetime import datetime

                prev_dt = datetime.fromisoformat(prev["created"].replace("Z", "+00:00"))
                curr_dt = datetime.fromisoformat(row["created"].replace("Z", "+00:00"))
                gap_hours = abs((curr_dt - prev_dt).total_seconds()) / 3600
                if gap_hours <= 4:
                    start_new_chain = False
                    chain_order += 1
            except (ValueError, TypeError):
                pass

        if start_new_chain:
            chain_id = row["session_id"]  # first session is the chain ID
            chain_order = 0

        conn.execute(
            "INSERT OR REPLACE INTO session_chains (session_id, chain_id, chain_order) VALUES (?, ?, ?)",
            (row["session_id"], chain_id, chain_order),
        )
        prev = {"key": key, "created": row["created"]}

    conn.commit()


def get_related_sessions(conn: sqlite3.Connection, session_id: str, limit: int = 5) -> list:
    """Find sessions related to this one via shared files."""
    rows = conn.execute("""
        SELECT s.session_id, s.project_path, s.summary, s.message_count, s.modified,
               COUNT(DISTINCT ge2.target_name) as shared_files
        FROM graph_edges ge1
        JOIN graph_edges ge2 ON ge1.target_name = ge2.target_name
            AND ge1.target_type = ge2.target_type
            AND ge1.session_id != ge2.session_id
        JOIN sessions s ON s.session_id = ge2.session_id AND s.is_subagent = 0
        WHERE ge1.session_id = ? AND ge1.relationship = 'edited'
        GROUP BY s.session_id
        ORDER BY shared_files DESC
        LIMIT ?
    """, (session_id, limit)).fetchall()
    return rows


def get_session_mtime(conn: sqlite3.Connection, session_id: str) -> float | None:
    """Get the stored mtime for a session, or None if not indexed."""
    row = conn.execute(
        "SELECT mtime FROM sessions WHERE session_id = ?", (session_id,)
    ).fetchone()
    return row["mtime"] if row else None


def get_all_session_ids(conn: sqlite3.Connection) -> set[str]:
    """Get all indexed session IDs."""
    rows = conn.execute("SELECT session_id FROM sessions").fetchall()
    return {row["session_id"] for row in rows}


def delete_session(conn: sqlite3.Connection, session_id: str) -> None:
    """Remove a session and all related data from the index."""
    # Delete vec entries BEFORE chunks (chunks are needed to find vec rows)
    try:
        conn.execute(
            """DELETE FROM chunks_vec WHERE chunk_rowid IN
               (SELECT chunk_id FROM chunks WHERE session_id = ?)""",
            (session_id,),
        )
    except Exception:
        pass  # chunks_vec may not exist yet
    conn.execute("DELETE FROM chunks WHERE session_id = ?", (session_id,))
    conn.execute("DELETE FROM session_files WHERE session_id = ?", (session_id,))
    conn.execute("DELETE FROM session_commands WHERE session_id = ?", (session_id,))
    conn.execute("DELETE FROM graph_edges WHERE session_id = ?", (session_id,))
    conn.execute("DELETE FROM sessions WHERE session_id = ?", (session_id,))


def get_stats(conn: sqlite3.Connection) -> dict:
    """Get index statistics."""
    row = conn.execute(
        """SELECT
            COUNT(*) as total,
            COUNT(CASE WHEN is_subagent = 0 THEN 1 END) as main_sessions,
            COUNT(CASE WHEN is_subagent = 1 THEN 1 END) as subagent_sessions,
            COUNT(DISTINCT project_dir) as projects,
            SUM(file_size) as total_size,
            SUM(message_count) as total_messages,
            MIN(created) as earliest,
            MAX(modified) as latest
        FROM sessions"""
    ).fetchone()
    return dict(row) if row else {}


def load_vec_extension(conn: sqlite3.Connection) -> bool:
    """Load sqlite-vec extension into a connection.

    Safe to call multiple times — silently succeeds if already loaded.
    """
    try:
        import sqlite_vec

        sqlite_vec.load(conn)
        return True
    except ImportError:
        return False
    except (sqlite3.OperationalError, AttributeError):
        # OperationalError: extension already loaded, or init failed
        # AttributeError: enable_load_extension not available
        try:
            conn.execute("SELECT 1 FROM chunks_vec LIMIT 0")
            return True
        except sqlite3.OperationalError:
            return False


def setup_vec_table(conn: sqlite3.Connection) -> None:
    """Create the vector table for semantic search. Requires sqlite-vec."""
    if not load_vec_extension(conn):
        return
    conn.execute(
        """CREATE VIRTUAL TABLE IF NOT EXISTS chunks_vec USING vec0(
            chunk_rowid INTEGER PRIMARY KEY,
            embedding float[384] distance_metric=cosine
        )"""
    )
    # Drop old sessions_vec if migrating from v1
    try:
        conn.execute("DROP TABLE IF EXISTS sessions_vec")
    except Exception:
        pass
    conn.commit()


def has_vec_table(conn: sqlite3.Connection) -> bool:
    """Check if the vector table exists."""
    row = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='chunks_vec'"
    ).fetchone()
    return row is not None
