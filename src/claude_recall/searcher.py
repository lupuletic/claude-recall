"""Search engine for claude-recall."""

from __future__ import annotations

import sqlite3
from pathlib import Path

from claude_recall.db import DB_PATH, get_connection, has_vec_table
from claude_recall.models import SearchResult, Session


def search(
    query: str,
    db_path: Path = DB_PATH,
    limit: int = 20,
    project_filter: str | None = None,
    after: str | None = None,
    before: str | None = None,
    semantic: bool | None = None,
    min_messages: int = 1,
) -> list[SearchResult]:
    """Search sessions using FTS5 and optionally vector similarity.

    Args:
        query: Search query string
        db_path: Path to the SQLite database
        limit: Maximum results to return
        project_filter: Substring filter on project path
        after: ISO date string, only sessions modified after this date
        before: ISO date string, only sessions modified before this date
        semantic: Force semantic search on/off. None = auto (use if available)
        min_messages: Minimum user messages for a session to be included
    """
    if not query or not query.strip():
        return []

    conn = get_connection(db_path)

    # Determine if we should do semantic search
    use_semantic = semantic if semantic is not None else has_vec_table(conn)
    if use_semantic:
        try:
            from claude_recall.embedder import get_embedder

            if get_embedder() is None:
                use_semantic = False
        except ImportError:
            use_semantic = False

    # Phase 1: Strict FTS (AND — all meaningful terms must appear)
    fts_results = _fts_search(conn, query, limit * 3, project_filter, after, before, min_messages)

    # Phase 2: If strict FTS found too few, try relaxed FTS (OR)
    if len(fts_results) < 3:
        relaxed = _fts_search_relaxed(
            conn, query, limit * 3, project_filter, after, before, min_messages
        )
        # Add relaxed results that aren't already in strict results
        seen = {r.session.session_id for r in fts_results}
        for r in relaxed:
            if r.session.session_id not in seen:
                r.score *= 0.7  # penalize relaxed matches
                fts_results.append(r)
                seen.add(r.session.session_id)

    if not use_semantic:
        # Re-normalize scores
        if fts_results:
            max_s = max(r.score for r in fts_results)
            if max_s > 0:
                for r in fts_results:
                    r.score /= max_s
        conn.close()
        return fts_results[:limit]

    # Phase 3: Semantic search (no threshold — let RRF handle ranking)
    vec_results = _vec_search(conn, query, limit * 3, project_filter, after, before, min_messages)

    # Phase 4: Hybrid ranking — weight semantic MORE when FTS found few results
    fts_strength = min(len(fts_results) / 5, 1.0)  # 0..1 based on FTS result count
    alpha = 0.3 + 0.3 * fts_strength  # 0.3 (semantic-heavy) to 0.6 (balanced)

    combined = _reciprocal_rank_fusion(fts_results, vec_results, alpha=alpha, k=60)

    # Phase 5: Cross-encoder reranking (most accurate, applied to top candidates)
    combined = _cross_encoder_rerank(query, combined[:limit * 2])

    conn.close()
    return combined[:limit]


def _fts_search(
    conn: sqlite3.Connection,
    query: str,
    limit: int,
    project_filter: str | None,
    after: str | None,
    before: str | None,
    min_messages: int,
) -> list[SearchResult]:
    """Full-text search using FTS5 with BM25 ranking."""
    # Build WHERE clauses for filters
    where_parts = []
    params: list = []

    # Exclude subagent sessions — show only main sessions
    where_parts.append("s.is_subagent = 0")

    if project_filter:
        where_parts.append("s.project_path LIKE ?")
        params.append(f"%{project_filter}%")
    if after:
        where_parts.append("s.modified >= ?")
        params.append(after)
    if before:
        where_parts.append("s.modified <= ?")
        params.append(before)
    if min_messages > 0:
        where_parts.append("s.message_count >= ?")
        params.append(min_messages)

    where_clause = ""
    if where_parts:
        where_clause = "AND " + " AND ".join(where_parts)

    # Escape FTS5 special characters in query for safe matching
    fts_query = _prepare_fts_query(query)

    sql = f"""
        SELECT
            s.*,
            sessions_fts.rank as fts_rank,
            snippet(sessions_fts, 0, '**', '**', '...', 20) as summary_snippet,
            snippet(sessions_fts, 1, '**', '**', '...', 20) as prompt_snippet,
            snippet(sessions_fts, 2, '**', '**', '...', 20) as last_prompt_snippet,
            snippet(sessions_fts, 3, '**', '**', '...', 20) as messages_snippet
        FROM sessions_fts
        JOIN sessions s ON s.rowid = sessions_fts.rowid
        WHERE sessions_fts MATCH ?
        {where_clause}
        ORDER BY sessions_fts.rank
        LIMIT ?
    """

    try:
        rows = conn.execute(sql, [fts_query, *params, limit]).fetchall()
    except sqlite3.OperationalError:
        # If the FTS query syntax is invalid, fall back to simple terms
        fts_query = " OR ".join(query.split())
        try:
            rows = conn.execute(sql, [fts_query, *params, limit]).fetchall()
        except sqlite3.OperationalError:
            return []

    results = []
    for i, row in enumerate(rows):
        session = _row_to_session(row)
        snippets = _collect_snippets(row)

        results.append(SearchResult(
            session=session,
            score=0.0,  # Will be set by RRF or used as-is
            fts_rank=row["fts_rank"],
            snippets=snippets,
        ))

    # Normalize scores: FTS5 rank is negative (more negative = better match)
    # Convert to 0..1 where 1 = best match
    if results:
        abs_ranks = [abs(r.fts_rank or 0) for r in results]
        min_abs = min(abs_ranks)  # worst match
        max_abs = max(abs_ranks)  # best match
        spread = max_abs - min_abs
        if spread > 0:
            for r in results:
                r.score = (abs(r.fts_rank or 0) - min_abs) / spread
        else:
            for r in results:
                r.score = 1.0

    return results


def _fts_search_relaxed(
    conn: sqlite3.Connection,
    query: str,
    limit: int,
    project_filter: str | None,
    after: str | None,
    before: str | None,
    min_messages: int,
) -> list[SearchResult]:
    """Relaxed FTS search using OR instead of AND."""
    # Filter stop words but use OR
    terms = [
        t for t in query.lower().split()
        if t not in _STOP_WORDS and len(t) > 1
    ]
    if not terms:
        return []

    relaxed_query = " OR ".join(terms)

    # Build WHERE clauses
    where_parts = ["s.is_subagent = 0"]
    params: list = []
    if project_filter:
        where_parts.append("s.project_path LIKE ?")
        params.append(f"%{project_filter}%")
    if after:
        where_parts.append("s.modified >= ?")
        params.append(after)
    if before:
        where_parts.append("s.modified <= ?")
        params.append(before)
    if min_messages > 0:
        where_parts.append("s.message_count >= ?")
        params.append(min_messages)
    where_clause = "AND " + " AND ".join(where_parts)

    sql = f"""
        SELECT s.*,
            sessions_fts.rank as fts_rank,
            snippet(sessions_fts, 0, '**', '**', '...', 20) as summary_snippet,
            snippet(sessions_fts, 1, '**', '**', '...', 20) as prompt_snippet,
            snippet(sessions_fts, 2, '**', '**', '...', 20) as last_prompt_snippet,
            snippet(sessions_fts, 3, '**', '**', '...', 20) as messages_snippet
        FROM sessions_fts
        JOIN sessions s ON s.rowid = sessions_fts.rowid
        WHERE sessions_fts MATCH ?
        {where_clause}
        ORDER BY sessions_fts.rank
        LIMIT ?
    """

    try:
        rows = conn.execute(sql, [relaxed_query, *params, limit]).fetchall()
    except sqlite3.OperationalError:
        return []

    results = []
    for row in rows:
        session = _row_to_session(row)
        snippets = _collect_snippets(row)
        results.append(SearchResult(
            session=session,
            score=0.0,
            fts_rank=row["fts_rank"],
            snippets=snippets,
        ))

    # Normalize
    if results:
        abs_ranks = [abs(r.fts_rank or 0) for r in results]
        min_abs = min(abs_ranks)
        max_abs = max(abs_ranks)
        spread = max_abs - min_abs
        if spread > 0:
            for r in results:
                r.score = (abs(r.fts_rank or 0) - min_abs) / spread
        else:
            for r in results:
                r.score = 1.0

    return results


def _vec_search(
    conn: sqlite3.Connection,
    query: str,
    limit: int,
    project_filter: str | None,
    after: str | None,
    before: str | None,
    min_messages: int,
) -> list[SearchResult]:
    """Vector similarity search using sqlite-vec."""
    try:
        from claude_recall.db import load_vec_extension
        from claude_recall.embedder import get_embedder

        if not load_vec_extension(conn):
            return []
        embedder = get_embedder()
        if embedder is None:
            return []
    except ImportError:
        return []

    # Embed the query
    query_embedding = embedder.embed_single(query)

    # Search chunks (not sessions) — parent-child retrieval pattern
    # Find best matching chunks, then group by parent session
    rows = conn.execute(
        """SELECT v.chunk_rowid, v.distance, c.session_id, c.chunk_text
           FROM chunks_vec v
           JOIN chunks c ON c.chunk_id = v.chunk_rowid
           WHERE v.embedding MATCH ?
           AND k = ?
           ORDER BY v.distance""",
        (query_embedding.tobytes(), limit * 3),  # over-fetch to allow grouping
    ).fetchall()

    # Group by session, keeping best chunk score per session
    session_best: dict[str, tuple[float, str]] = {}  # session_id -> (similarity, chunk_text)
    for row in rows:
        similarity = 1.0 - row["distance"]
        sid = row["session_id"]
        if sid not in session_best or similarity > session_best[sid][0]:
            session_best[sid] = (similarity, row["chunk_text"][:200])

    # Fetch session data for matched sessions (main sessions only)
    results = []
    for sid, (similarity, chunk_snippet) in session_best.items():
        session_row = conn.execute(
            "SELECT * FROM sessions WHERE session_id = ? AND is_subagent = 0", (sid,)
        ).fetchone()
        if session_row is None:
            continue

        session = _row_to_session(session_row)

        if project_filter and project_filter.lower() not in session.project_path.lower():
            continue
        if after and (session.modified or "") < after:
            continue
        if before and (session.modified or "") > before:
            continue
        if session.message_count < min_messages:
            continue

        results.append(SearchResult(
            session=session,
            score=0.0,
            vec_score=similarity,
            snippets=[chunk_snippet],
        ))

    # Sort by vec score descending
    results.sort(key=lambda r: r.vec_score or 0, reverse=True)
    return results


def _reciprocal_rank_fusion(
    fts_results: list[SearchResult],
    vec_results: list[SearchResult],
    alpha: float = 0.6,
    k: int = 60,
) -> list[SearchResult]:
    """Combine FTS5 and vector search results using RRF.

    score = alpha / (k + fts_rank) + (1 - alpha) / (k + vec_rank)
    """
    # Build lookup by session_id
    fts_map: dict[str, tuple[int, SearchResult]] = {}
    for rank, r in enumerate(fts_results):
        fts_map[r.session.session_id] = (rank, r)

    vec_map: dict[str, tuple[int, SearchResult]] = {}
    for rank, r in enumerate(vec_results):
        vec_map[r.session.session_id] = (rank, r)

    all_ids = set(fts_map.keys()) | set(vec_map.keys())
    absent_rank = max(len(fts_results), len(vec_results)) + 100

    combined: list[SearchResult] = []
    for sid in all_ids:
        fts_rank = fts_map[sid][0] if sid in fts_map else absent_rank
        vec_rank = vec_map[sid][0] if sid in vec_map else absent_rank

        rrf_score = alpha / (k + fts_rank) + (1 - alpha) / (k + vec_rank)

        # Pick the result object with the most info
        if sid in fts_map:
            result = fts_map[sid][1]
        else:
            result = vec_map[sid][1]

        # Merge snippets from both sources
        if sid in fts_map and sid in vec_map:
            seen = set(result.snippets)
            for s in vec_map[sid][1].snippets:
                if s not in seen:
                    result.snippets.append(s)

        result.score = rrf_score
        result.fts_rank = fts_rank if sid in fts_map else None
        result.vec_score = vec_map[sid][1].vec_score if sid in vec_map else None

        combined.append(result)

    combined.sort(key=lambda r: r.score, reverse=True)

    # Normalize scores to 0..1 range
    if combined:
        max_score = combined[0].score
        min_score = combined[-1].score
        spread = max_score - min_score
        if spread > 0:
            for r in combined:
                r.score = (r.score - min_score) / spread
        else:
            for r in combined:
                r.score = 1.0

    return combined


# Common English stop words that pollute FTS results
_STOP_WORDS = {
    "a", "an", "the", "is", "it", "in", "on", "at", "to", "for", "of",
    "and", "or", "but", "not", "with", "from", "by", "as", "be", "was",
    "were", "been", "are", "am", "do", "did", "does", "has", "have", "had",
    "will", "would", "could", "should", "can", "may", "might", "shall",
    "we", "i", "you", "he", "she", "they", "them", "my", "your", "our",
    "me", "us", "this", "that", "these", "those", "there", "here",
    "where", "when", "what", "which", "who", "how", "why",
    "if", "then", "so", "just", "also", "very", "too",
    "about", "up", "out", "no", "yes", "all", "some", "any",
    "session", "sessions", "find", "search", "show", "get", "look",
    "one", "time", "did", "made", "built", "worked",
}


def _cross_encoder_rerank(query: str, results: list[SearchResult]) -> list[SearchResult]:
    """Rerank results using a cross-encoder model for precise relevance scoring.

    The cross-encoder takes (query, document) pairs with full cross-attention,
    producing much more accurate relevance scores than bi-encoder similarity.
    ~18ms for 20 documents on CPU.
    """
    if not results or len(results) <= 1:
        return results

    try:
        from claude_recall.embedder import get_reranker

        reranker = get_reranker()
        if reranker is None:
            return results
    except ImportError:
        return results

    # Build document texts for reranking
    documents = []
    for r in results:
        s = r.session
        parts = [s.summary or "", s.first_prompt or ""]
        if s.last_prompt and s.last_prompt != s.first_prompt:
            parts.append(s.last_prompt or "")
        doc = " ".join(p for p in parts if p)[:500]
        documents.append(doc)

    try:
        ranked = reranker.rerank(query, documents)
    except Exception:
        return results

    # Reorder results by cross-encoder score
    reranked = []
    for orig_idx, ce_score in ranked:
        if orig_idx < len(results):
            r = results[orig_idx]
            r.score = ce_score
            reranked.append(r)

    # Normalize to 0..1
    if reranked:
        scores = [r.score for r in reranked]
        min_s, max_s = min(scores), max(scores)
        spread = max_s - min_s
        if spread > 0:
            for r in reranked:
                r.score = (r.score - min_s) / spread
        else:
            for r in reranked:
                r.score = 1.0

    # Drop results that are clearly irrelevant compared to the top result
    # If the top result is strong and the rest are weak, trim the tail
    if len(reranked) >= 2 and reranked[0].score > 0.5:
        cutoff = reranked[0].score * 0.4  # must be at least 40% of top score
        reranked = [r for r in reranked if r.score >= cutoff]

    return reranked


def _prepare_fts_query(query: str) -> str:
    """Prepare a search query for FTS5.

    Extracts meaningful keywords, filters stop words,
    and joins with AND for precise matching.
    """
    query = query.strip()
    if not query:
        return ""

    # If user already used FTS5 syntax (AND, OR, NOT, quotes), pass through
    if any(op in query for op in [" AND ", " OR ", " NOT ", '"']):
        return query

    # Filter stop words and short terms
    terms = [
        t for t in query.lower().split()
        if t not in _STOP_WORDS and len(t) > 1
    ]

    if not terms:
        # All stop words — use original words as OR fallback
        terms = [t for t in query.split() if len(t) > 1]
        if not terms:
            return query
        return " OR ".join(terms)

    if len(terms) == 1:
        return terms[0]

    # Use AND for precision — all meaningful terms must appear
    return " AND ".join(terms)


def _row_to_session(row: sqlite3.Row) -> Session:
    """Convert a database row to a Session object."""
    return Session(
        session_id=row["session_id"],
        project_path=row["project_path"],
        project_dir=row["project_dir"],
        file_path=row["file_path"],
        summary=row["summary"],
        first_prompt=row["first_prompt"],
        first_reply=row["first_reply"],
        last_prompt=row["last_prompt"] if "last_prompt" in row.keys() else None,
        last_reply=row["last_reply"] if "last_reply" in row.keys() else None,
        messages_text=row["messages_text"],
        git_branch=row["git_branch"],
        message_count=row["message_count"],
        file_size=row["file_size"],
        created=row["created"],
        modified=row["modified"],
        mtime=row["mtime"],
        is_subagent=bool(row["is_subagent"]),
        parent_session=row["parent_session"],
    )


def _collect_snippets(row: sqlite3.Row) -> list[str]:
    """Collect non-empty snippets from FTS5 results."""
    snippets = []
    for key in ("summary_snippet", "prompt_snippet", "last_prompt_snippet", "messages_snippet"):
        val = row[key]
        if val and val.strip() and val != "...":
            snippets.append(val[:200])
    return snippets
