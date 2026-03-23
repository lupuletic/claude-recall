"""Session indexer for claude-recall."""

from __future__ import annotations

import sys
import time
from pathlib import Path

from claude_recall import has_semantic
from claude_recall.db import (
    DB_PATH,
    delete_session,
    get_all_session_ids,
    get_connection,
    get_session_mtime,
    setup_vec_table,
    upsert_chunks,
    upsert_session,
)
from claude_recall.models import Session
from claude_recall.utils import (
    PROJECTS_DIR,
    decode_project_path,
    discover_sessions,
    load_sessions_index,
    parse_session_file,
)


def _mtime_to_iso(mtime: float) -> str:
    """Convert a file mtime to ISO 8601 string."""
    from datetime import datetime, timezone

    return datetime.fromtimestamp(mtime, tz=timezone.utc).isoformat()


def build_index(
    projects_dir: Path = PROJECTS_DIR,
    db_path: Path = DB_PATH,
    force: bool = False,
    verbose: bool = True,
    defer_embeddings: bool = False,
) -> dict:
    """Build or update the session index.

    Returns stats dict with counts of indexed/skipped/removed sessions.
    """
    start = time.monotonic()
    conn = get_connection(db_path)

    # Set up vector table if semantic deps available
    if has_semantic():
        setup_vec_table(conn)

    # Discover all session files on disk
    discovered = discover_sessions(projects_dir)

    if verbose:
        print(f"Found {len(discovered)} session files", file=sys.stderr)

    # Load existing index state
    existing_ids = get_all_session_ids(conn) if not force else set()

    # Cache sessions-index.json data per project
    index_cache: dict[str, dict[str, dict]] = {}

    # Pre-build set of session IDs that are parents of subagents
    parent_ids = {
        s["parent_session"]
        for s in discovered
        if s.get("is_subagent") and s.get("parent_session")
    }

    indexed = 0
    skipped = 0
    errors = 0

    for i, session_info in enumerate(discovered):
        session_id = session_info["session_id"]
        project_dir = session_info["project_dir"]
        file_mtime = session_info["mtime"]

        # Skip if already indexed and file hasn't changed
        if not force and session_id in existing_ids:
            stored_mtime = get_session_mtime(conn, session_id)
            if stored_mtime is not None and abs(stored_mtime - file_mtime) < 0.01:
                skipped += 1
                continue

        # Load sessions-index metadata (cached per project)
        if project_dir not in index_cache:
            index_cache[project_dir] = load_sessions_index(project_dir, projects_dir)
        idx_meta = index_cache[project_dir].get(session_id, {})

        # Parse the session file
        try:
            parsed = parse_session_file(session_info["file_path"])
        except Exception:
            errors += 1
            continue

        # Skip sessions with no user messages — UNLESS they're parents of subagents
        if parsed["message_count"] == 0 and session_id not in parent_ids:
            skipped += 1
            continue

        # Decode project path
        project_path = decode_project_path(project_dir, projects_dir)

        # Use sessions-index summary if available, otherwise auto-generate
        summary = idx_meta.get("summary") or parsed.get("summary")

        session = Session(
            session_id=session_id,
            project_path=project_path,
            project_dir=project_dir,
            file_path=session_info["file_path"],
            summary=summary,
            first_prompt=parsed["first_prompt"],
            first_reply=parsed["first_reply"],
            last_prompt=parsed["last_prompt"],
            last_reply=parsed["last_reply"],
            messages_text=parsed["messages_text"],
            git_branch=idx_meta.get("gitBranch"),
            message_count=parsed["message_count"],
            file_size=session_info["file_size"],
            created=idx_meta.get("created") or _mtime_to_iso(file_mtime),
            modified=idx_meta.get("modified") or _mtime_to_iso(file_mtime),
            mtime=file_mtime,
            is_subagent=session_info["is_subagent"],
            parent_session=session_info["parent_session"],
        )

        upsert_session(conn, session)
        upsert_chunks(conn, session_id, parsed["chunks"])
        indexed += 1

        # Commit in batches so a kill doesn't lose everything
        if indexed % 100 == 0:
            conn.commit()

        # Progress indicator
        if verbose and (indexed % 50 == 0 or i == len(discovered) - 1):
            print(
                f"\r  Indexed {indexed} sessions ({skipped} unchanged, {errors} errors)...",
                end="",
                file=sys.stderr,
            )

    # Enrich parent sessions with subagent content
    # This ensures searching for terms that only appear in subagents
    # still finds the parent session
    if indexed > 0:
        _enrich_parents_with_subagent_content(conn, verbose)

    # Generate embeddings if semantic is available (unless deferred)
    embeddings_generated = 0
    if has_semantic() and indexed > 0 and not defer_embeddings:
        embeddings_generated = _generate_embeddings(conn, force, verbose)

    # Remove sessions that no longer exist on disk
    discovered_ids = {s["session_id"] for s in discovered}
    removed = 0
    for old_id in existing_ids - discovered_ids:
        delete_session(conn, old_id)
        removed += 1

    conn.commit()

    elapsed = time.monotonic() - start

    if verbose:
        print(file=sys.stderr)  # newline after progress
        print(
            f"  Done in {elapsed:.1f}s: {indexed} indexed, "
            f"{skipped} unchanged, {removed} removed, {errors} errors",
            file=sys.stderr,
        )
        if embeddings_generated:
            print(
                f"  Generated {embeddings_generated} embeddings",
                file=sys.stderr,
            )

    conn.close()

    return {
        "indexed": indexed,
        "skipped": skipped,
        "removed": removed,
        "errors": errors,
        "embeddings": embeddings_generated,
        "elapsed": elapsed,
        "total_discovered": len(discovered),
    }


def ensure_index(
    projects_dir: Path = PROJECTS_DIR,
    db_path: Path = DB_PATH,
    verbose: bool = True,
) -> None:
    """Ensure the index exists and is reasonably up-to-date.

    Called automatically before search. Builds FTS index immediately,
    defers embedding generation to a background process on first run.
    """
    is_first_run = not db_path.exists()

    if is_first_run:
        if verbose:
            print("Building index for the first time...", file=sys.stderr)
        # Build FTS index immediately (fast, ~2s), skip embeddings
        build_index(
            projects_dir, db_path, force=False, verbose=verbose,
            defer_embeddings=True,
        )
        # Generate embeddings in background
        if has_semantic():
            _spawn_background_embeddings(db_path, projects_dir, verbose)
    else:
        # Quick incremental check — always defer embeddings (only generate during explicit `index`)
        try:
            build_index(
                projects_dir, db_path, force=False, verbose=False,
                defer_embeddings=True,
            )
        except Exception:
            pass  # DB locked by background embeddings — search with existing index


def _spawn_background_embeddings(db_path: Path, projects_dir: Path, verbose: bool) -> None:
    """Spawn a background process to generate embeddings."""
    import subprocess as sp

    if verbose:
        print(
            "  Generating embeddings in background (search works now with keywords)...",
            file=sys.stderr,
        )

    # Run `claude-recall index --quiet` in background
    import shutil

    claude_recall_bin = shutil.which("claude-recall")
    if claude_recall_bin:
        sp.Popen(
            [claude_recall_bin, "index", "--quiet"],
            stdout=sp.DEVNULL,
            stderr=sp.DEVNULL,
            start_new_session=True,
        )
    else:
        # Fallback: run via python -m
        sp.Popen(
            [sys.executable, "-m", "claude_recall", "index", "--quiet"],
            stdout=sp.DEVNULL,
            stderr=sp.DEVNULL,
            start_new_session=True,
        )


def _enrich_parents_with_subagent_content(conn, verbose: bool = False) -> None:
    """Append subagent first_prompts to parent session messages_text.

    This ensures that searching for terms only used in subagent sessions
    (e.g. project names, specific tools) still finds the parent session.
    """
    rows = conn.execute(
        """SELECT s.session_id AS sub_id, s.parent_session, s.first_prompt
           FROM sessions s
           WHERE s.is_subagent = 1 AND s.parent_session IS NOT NULL
           AND s.first_prompt IS NOT NULL"""
    ).fetchall()

    if not rows:
        return

    # Group subagent prompts by parent
    parent_extras: dict[str, list[str]] = {}
    for row in rows:
        parent_id = row["parent_session"]
        prompt = row["first_prompt"]
        if prompt and prompt.strip():
            parent_extras.setdefault(parent_id, []).append(prompt[:200])

    enriched = 0
    for parent_id, extras in parent_extras.items():
        extra_text = "\n".join(extras)
        # Append to the parent's messages_text
        conn.execute(
            """UPDATE sessions
               SET messages_text = COALESCE(messages_text, '') || ? || ?
               WHERE session_id = ? AND is_subagent = 0""",
            ("\n", extra_text, parent_id),
        )
        enriched += 1

    if enriched:
        conn.commit()
        # Rebuild FTS for enriched parents
        conn.execute("INSERT INTO sessions_fts(sessions_fts) VALUES('rebuild')")
        conn.commit()

    if verbose and enriched:
        print(f"\n  Enriched {enriched} parent sessions with subagent content", file=sys.stderr)


def _generate_embeddings(
    conn, force: bool = False, verbose: bool = True
) -> int:
    """Generate embeddings for sessions that don't have them yet."""
    try:
        from claude_recall.embedder import get_embedder
    except ImportError:
        return 0

    embedder = get_embedder()
    if embedder is None:
        return 0

    from claude_recall.db import load_vec_extension

    load_vec_extension(conn)

    # Find chunks needing embeddings (chunks without vec entries)
    if force:
        conn.execute("DELETE FROM chunks_vec")
        rows = conn.execute(
            "SELECT chunk_id, chunk_text FROM chunks"
        ).fetchall()
    else:
        rows = conn.execute(
            """SELECT c.chunk_id, c.chunk_text
               FROM chunks c
               WHERE c.chunk_id NOT IN (
                   SELECT chunk_rowid FROM chunks_vec
               )"""
        ).fetchall()

    if not rows:
        return 0

    if verbose:
        print(f"\n  Generating embeddings for {len(rows)} chunks...", file=sys.stderr)

    # Prepare texts and IDs
    texts = [row["chunk_text"] for row in rows if row["chunk_text"].strip()]
    chunk_ids = [row["chunk_id"] for row in rows if row["chunk_text"].strip()]

    if not texts:
        return 0

    # Batch embed
    embeddings = embedder.embed(texts)

    # Store in vec table with periodic commits
    for i, (chunk_id, embedding) in enumerate(zip(chunk_ids, embeddings)):
        conn.execute(
            "INSERT OR REPLACE INTO chunks_vec (chunk_rowid, embedding) VALUES (?, ?)",
            (chunk_id, embedding.tobytes()),
        )
        if (i + 1) % 50 == 0:
            conn.commit()
            if verbose:
                print(
                    f"\r  Embedded {i + 1}/{len(chunk_ids)} chunks...",
                    end="",
                    file=sys.stderr,
                )

    conn.commit()
    return len(chunk_ids)
