"""Search engine for claude-recall."""

from __future__ import annotations

import math
import sqlite3
from pathlib import Path

from claude_recall.db import DB_PATH, get_connection, get_related_sessions, has_vec_table
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
    try:
        return _search_pipeline(
            conn, query, limit, project_filter, after, before,
            semantic, min_messages,
        )
    finally:
        conn.close()


def _search_pipeline(
    conn, query, limit, project_filter, after, before, semantic, min_messages,
) -> list[SearchResult]:
    """Core search pipeline. Connection managed by caller."""
    # Check for structured query prefixes
    if query.startswith("file:"):
        return _file_search(conn, query[5:].strip(), limit, project_filter)
    if query.startswith("cmd:"):
        return _command_search(conn, query[4:].strip(), limit, project_filter)
    if query.startswith("branch:"):
        return _branch_search(conn, query[7:].strip(), limit, project_filter)

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
    # Fetch more candidates to survive depth-boost reranking (1-message
    # automated sessions often dominate BM25 and get penalized later)
    fts_fetch = max(limit * 5, 30)
    fts_results = _fts_search(conn, query, fts_fetch, project_filter, after, before, min_messages)

    # Phase 2: If strict FTS found too few, try relaxed FTS (OR)
    if len(fts_results) < 3:
        relaxed = _fts_search_relaxed(
            conn, query, fts_fetch, project_filter, after, before, min_messages
        )
        seen = {r.session.session_id for r in fts_results}
        for r in relaxed:
            if r.session.session_id not in seen:
                r.score *= 0.5  # heavier penalty for OR-only matches
                fts_results.append(r)
                seen.add(r.session.session_id)

    if not use_semantic:
        # Apply boosts before final normalization
        _apply_depth_boost(fts_results)
        _apply_project_path_boost(query, fts_results)
        _apply_prompt_match_boost(query, fts_results)
        if fts_results:
            max_s = max(r.score for r in fts_results)
            if max_s > 0:
                for r in fts_results:
                    r.score /= max_s
        fts_results.sort(key=lambda r: r.score, reverse=True)
        return fts_results[:limit]

    # Phase 3: Semantic search
    vec_results = _vec_search(conn, query, limit * 3, project_filter, after, before, min_messages)

    # Phase 4: Hybrid ranking — weight semantic MORE when FTS found few results
    fts_strength = min(len(fts_results) / 5, 1.0)
    alpha = 0.3 + 0.3 * fts_strength

    combined = _reciprocal_rank_fusion(fts_results, vec_results, alpha=alpha, k=60)

    # Phase 5: Cross-encoder reranking
    combined = _cross_encoder_rerank(query, combined[:limit * 2])

    # Phase 6: LLM reranking — only when explicitly set to "llm" mode
    from claude_recall.config import load_config

    config = load_config()
    if config.get("search_mode") == "llm" and combined:
        combined = _llm_rerank(query, combined[:limit])

    # Apply boosts as tiebreaker after reranking
    _apply_depth_boost(combined)
    _apply_project_path_boost(query, combined)
    _apply_prompt_match_boost(query, combined)
    if combined:
        max_s = max(r.score for r in combined)
        if max_s > 0:
            for r in combined:
                r.score /= max_s
        combined.sort(key=lambda r: r.score, reverse=True)

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

    # BM25 column weights: summary=5, first_prompt=3, last_prompt=3, messages_text=2, project_path=4
    sql = f"""
        SELECT
            s.*,
            bm25(sessions_fts, 5.0, 3.0, 3.0, 2.0, 4.0) as fts_rank,
            snippet(sessions_fts, 0, '**', '**', '...', 20) as summary_snippet,
            snippet(sessions_fts, 1, '**', '**', '...', 20) as prompt_snippet,
            snippet(sessions_fts, 2, '**', '**', '...', 20) as last_prompt_snippet,
            snippet(sessions_fts, 3, '**', '**', '...', 20) as messages_snippet
        FROM sessions_fts
        JOIN sessions s ON s.rowid = sessions_fts.rowid
        WHERE sessions_fts MATCH ?
        {where_clause}
        ORDER BY bm25(sessions_fts, 5.0, 3.0, 3.0, 2.0, 4.0)
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

    # Use prefix matching in relaxed mode for broader coverage
    parts = []
    for t in terms:
        if len(t) >= 3:
            parts.append(f'("{t}" OR "{t}"*)')
        else:
            parts.append(f'"{t}"')
    relaxed_query = " OR ".join(parts)

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
            bm25(sessions_fts, 5.0, 3.0, 3.0, 2.0, 4.0) as fts_rank,
            snippet(sessions_fts, 0, '**', '**', '...', 20) as summary_snippet,
            snippet(sessions_fts, 1, '**', '**', '...', 20) as prompt_snippet,
            snippet(sessions_fts, 2, '**', '**', '...', 20) as last_prompt_snippet,
            snippet(sessions_fts, 3, '**', '**', '...', 20) as messages_snippet
        FROM sessions_fts
        JOIN sessions s ON s.rowid = sessions_fts.rowid
        WHERE sessions_fts MATCH ?
        {where_clause}
        ORDER BY bm25(sessions_fts, 5.0, 3.0, 3.0, 2.0, 4.0)
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
    # For subagent chunks, map to their parent session
    session_best: dict[str, tuple[float, str]] = {}  # session_id -> (similarity, chunk_text)
    for row in rows:
        similarity = 1.0 - row["distance"]
        sid = row["session_id"]

        # If this is a subagent, map to the nearest main session
        sub_row = conn.execute(
            "SELECT parent_session, is_subagent, project_dir FROM sessions WHERE session_id = ?",
            (sid,),
        ).fetchone()
        if sub_row and sub_row["is_subagent"]:
            parent_id = sub_row["parent_session"]
            # Check if parent exists in index
            parent_exists = conn.execute(
                "SELECT 1 FROM sessions WHERE session_id = ? AND is_subagent = 0",
                (parent_id,),
            ).fetchone() if parent_id else None

            if parent_exists:
                sid = parent_id
            else:
                # Parent not indexed — find any main session in same project
                fallback = conn.execute(
                    """SELECT session_id FROM sessions
                       WHERE project_dir = ? AND is_subagent = 0
                       ORDER BY message_count DESC LIMIT 1""",
                    (sub_row["project_dir"],),
                ).fetchone()
                if fallback:
                    sid = fallback["session_id"]
                else:
                    continue  # no main session found

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


def _apply_depth_boost(results: list[SearchResult]) -> None:
    """Boost scores for sessions with more messages.

    Uses log2(message_count) as a multiplier:
      1 msg → 1.0x, 5 msgs → ~1.23x, 10 msgs → ~1.33x, 50 msgs → ~1.56x
    This penalizes single-message automated sessions (CI/issue bots)
    and rewards substantive human conversations.
    """
    for r in results:
        mc = max(r.session.message_count, 1)
        # Mild tiebreaker: 2 msgs → 1.05x, 10 msgs → 1.17x, 50 msgs → 1.28x
        boost = 1.0 + 0.05 * math.log2(mc)
        # Strong penalty for 1-message sessions — likely automated CI/bots
        if mc == 1:
            boost *= 0.5
        r.score *= boost


def _apply_project_path_boost(query: str, results: list[SearchResult]) -> None:
    """Boost results where query terms appear in the project path/name.

    When someone searches "trading bot", a project literally called
    "polymarket-copy-trading-bot" is highly likely to be the right match.
    This gives a strong boost proportional to how many query terms match.
    """
    terms = [
        t for t in query.lower().split()
        if t not in _STOP_WORDS and len(t) > 2
    ]
    if not terms:
        return

    for r in results:
        path = r.session.project_path.lower()
        # Check how many query terms appear in the project path
        matches = sum(1 for t in terms if t in path)
        if matches > 0:
            # Proportional boost: 1 term = 1.3x, 2 terms = 1.6x, 3+ = 1.9x
            boost = 1.0 + 0.3 * min(matches, 3)
            r.score *= boost


def _apply_prompt_match_boost(query: str, results: list[SearchResult]) -> None:
    """Boost results where the query closely matches first/last prompt.

    This helps "find my conversation" queries where the user remembers
    exactly what they typed, e.g. "give me the final version".
    Also boosts sessions where query terms are central to the conversation
    (high term density in summary/prompts), not just mentioned in passing.
    """
    query_lower = query.lower().strip()
    if len(query_lower) < 3:
        return

    # Extract meaningful query terms for density check
    terms = [
        t for t in query_lower.split()
        if t not in _STOP_WORDS and len(t) > 1
    ]

    for r in results:
        first_prompt = (r.session.first_prompt or "").lower()
        last_prompt = (r.session.last_prompt or "").lower()
        summary = (r.session.summary or "").lower()

        # Strong boost: query is a near-exact match of a prompt
        if len(query_lower) >= 5:
            if query_lower in last_prompt or last_prompt in query_lower:
                r.score *= 3.0
                continue
            if query_lower in first_prompt or first_prompt in query_lower:
                r.score *= 2.5
                continue

        # Term centrality boost: if query terms appear in the summary
        # or first prompt, the session is likely ABOUT this topic,
        # not just mentioning it in passing.
        if terms:
            summary_matches = sum(1 for t in terms if t in summary)
            prompt_matches = sum(1 for t in terms if t in first_prompt)
            best_match = max(summary_matches, prompt_matches)
            if best_match > 0:
                # Boost proportional to match fraction
                fraction = best_match / len(terms)
                r.score *= 1.0 + 0.5 * fraction


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
    "one", "time", "did", "made", "built", "worked", "give", "gave",
    "need", "want", "try", "use", "using", "used", "like", "thing",
}


def _cross_encoder_rerank(query: str, results: list[SearchResult]) -> list[SearchResult]:
    """Rerank results using a cross-encoder model for precise relevance scoring.

    The cross-encoder takes (query, document) pairs with full cross-attention,
    producing much more accurate relevance scores than bi-encoder similarity.
    ~18ms for 20 documents on CPU.
    """
    if not results or len(results) <= 1:
        return results

    # Skip cross-encoder for short keyword queries (1-2 meaningful terms)
    # Short queries like "reshot" or "git ssh" are exact matches where
    # BM25 ranking is already correct — the cross-encoder can misrank
    # by treating "reshot" as similar to "resize"
    meaningful_terms = [
        t for t in query.lower().split()
        if t not in _STOP_WORDS and len(t) > 1
    ]
    if len(meaningful_terms) <= 2:
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
        parts = [s.summary or ""]
        # Include matched chunk snippets
        if r.snippets:
            parts.append(r.snippets[0])
        parts.append(s.first_prompt or "")
        # Include a sample of messages_text (contains enriched subagent content)
        if s.messages_text:
            parts.append(s.messages_text[:200])
        # Include first_reply — this describes the actual work done
        if s.first_reply:
            parts.append(s.first_reply)
        if s.last_prompt and s.last_prompt != s.first_prompt:
            parts.append(s.last_prompt or "")
        doc = " ".join(p for p in parts if p)[:768]
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
            has_fts = r.fts_rank is not None
            has_vec = r.vec_score is not None
            # Strong boost for results matching BOTH keyword AND semantic
            if has_fts and has_vec:
                ce_score *= 2.0
            # Moderate boost for FTS-only matches (exact keyword match is strong signal)
            elif has_fts:
                ce_score *= 1.3
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
    # BUT keep results from the same project (they're likely related sessions)
    if len(reranked) >= 2 and reranked[0].score > 0.5:
        top_project = reranked[0].session.project_dir
        cutoff = reranked[0].score * 0.4
        reranked = [
            r for r in reranked
            if r.score >= cutoff or r.session.project_dir == top_project
        ]

    return reranked


def _llm_rerank(query: str, results: list[SearchResult]) -> list[SearchResult]:
    """Rerank using Claude via `claude -p` for highest quality results."""
    if not results:
        return results

    import sys

    print("  Reranking with Claude...", end="", file=sys.stderr, flush=True)

    from claude_recall.llm_reranker import llm_rerank

    candidates = []
    for r in results:
        s = r.session
        candidates.append({
            "summary": s.summary,
            "first_prompt": s.first_prompt,
            "last_prompt": s.last_prompt,
            "project_path": s.project_path,
            "message_count": s.message_count,
        })

    ranked_indices = llm_rerank(query, candidates)

    print(" done.", file=sys.stderr)

    reranked = []
    for rank, idx in enumerate(ranked_indices):
        if idx < len(results):
            r = results[idx]
            r.score = 1.0 - (rank / max(len(ranked_indices), 1))
            reranked.append(r)

    return reranked


def _file_search(
    conn: sqlite3.Connection,
    file_query: str,
    limit: int,
    project_filter: str | None,
) -> list[SearchResult]:
    """Search by file name/path using the normalized session_files table."""
    where_parts = ["s.is_subagent = 0"]
    params: list = [f"%{file_query}%", f"%{file_query}%"]
    if project_filter:
        where_parts.append("s.project_path LIKE ?")
        params.append(f"%{project_filter}%")

    where_clause = " AND ".join(where_parts)
    params.append(limit)

    rows = conn.execute(f"""
        SELECT DISTINCT sf.session_id, sf.file_path, sf.action, s.*
        FROM session_files sf
        JOIN sessions s ON s.session_id = sf.session_id
        WHERE (sf.file_name LIKE ? OR sf.file_path LIKE ?)
        AND {where_clause}
        ORDER BY s.modified DESC
        LIMIT ?
    """, params).fetchall()

    results = []
    for row in rows:
        session = _row_to_session(row)
        results.append(SearchResult(
            session=session,
            score=1.0,
            snippets=[f"{row['action'].title()}ed: {row['file_path']}"],
        ))

    # Normalize scores by recency
    for i, r in enumerate(results):
        r.score = 1.0 - (i / max(len(results), 1))
    return results


def _command_search(
    conn: sqlite3.Connection,
    cmd_query: str,
    limit: int,
    project_filter: str | None,
) -> list[SearchResult]:
    """Search by command name using the normalized session_commands table."""
    where_parts = ["s.is_subagent = 0"]
    params: list = [f"%{cmd_query}%", f"%{cmd_query}%"]
    if project_filter:
        where_parts.append("s.project_path LIKE ?")
        params.append(f"%{project_filter}%")

    where_clause = " AND ".join(where_parts)
    params.append(limit)

    rows = conn.execute(f"""
        SELECT DISTINCT sc.session_id, sc.command, sc.command_name, s.*
        FROM session_commands sc
        JOIN sessions s ON s.session_id = sc.session_id
        WHERE (sc.command_name LIKE ? OR sc.command LIKE ?)
        AND {where_clause}
        ORDER BY s.modified DESC
        LIMIT ?
    """, params).fetchall()

    results = []
    for row in rows:
        session = _row_to_session(row)
        results.append(SearchResult(
            session=session,
            score=1.0,
            snippets=[f"Ran: {row['command']}"],
        ))

    for i, r in enumerate(results):
        r.score = 1.0 - (i / max(len(results), 1))
    return results


def _branch_search(
    conn: sqlite3.Connection,
    branch_query: str,
    limit: int,
    project_filter: str | None,
) -> list[SearchResult]:
    """Search by git branch name."""
    where_parts = ["s.is_subagent = 0", "(s.git_branch LIKE ? OR s.git_branch_detected LIKE ?)"]
    params: list = [f"%{branch_query}%", f"%{branch_query}%"]
    if project_filter:
        where_parts.append("s.project_path LIKE ?")
        params.append(f"%{project_filter}%")

    where_clause = " AND ".join(where_parts)
    params.append(limit)

    rows = conn.execute(f"""
        SELECT s.*
        FROM sessions s
        WHERE {where_clause}
        ORDER BY s.modified DESC
        LIMIT ?
    """, params).fetchall()

    results = []
    for row in rows:
        session = _row_to_session(row)
        branch = session.git_branch or session.git_branch_detected or ""
        results.append(SearchResult(
            session=session,
            score=1.0,
            snippets=[f"Branch: {branch}"],
        ))

    for i, r in enumerate(results):
        r.score = 1.0 - (i / max(len(results), 1))
    return results


def _prepare_fts_query(query: str, use_prefix: bool = True) -> str:
    """Prepare a search query for FTS5.

    Extracts meaningful keywords, filters stop words,
    and joins with AND for precise matching.
    Uses prefix matching (*) so "auth" finds "authentication" etc.
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
        return " OR ".join(f'"{t}"' for t in terms)

    # Use prefix matching for terms >= 3 chars (avoids noise from very short prefixes)
    suffix = "*" if use_prefix else ""

    if len(terms) == 1:
        t = terms[0]
        # For single-term queries, exact match is sufficient.
        # Prefix matching adds noise (e.g. "reshot*" matching "resolution").
        # The relaxed search fallback handles the case where exact match
        # returns too few results.
        return f'"{t}"'

    # Quote each term to escape FTS5 special chars (colons, parens, etc.)
    # Use AND for precision — all meaningful terms must appear
    # Add prefix variants so "auth" matches "authentication"
    parts = []
    for t in terms:
        if use_prefix and len(t) >= 3:
            parts.append(f'("{t}" OR "{t}"{suffix})')
        else:
            parts.append(f'"{t}"')
    return " AND ".join(parts)


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
        files_modified=row["files_modified"] if "files_modified" in row.keys() else None,
        commands_run=row["commands_run"] if "commands_run" in row.keys() else None,
        git_branch_detected=row["git_branch_detected"] if "git_branch_detected" in row.keys() else None,
    )


def _collect_snippets(row: sqlite3.Row) -> list[str]:
    """Collect non-empty snippets from FTS5 results."""
    snippets = []
    for key in ("summary_snippet", "prompt_snippet", "last_prompt_snippet", "messages_snippet"):
        val = row[key]
        if val and val.strip() and val != "...":
            snippets.append(val[:200])
    return snippets
