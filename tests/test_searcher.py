"""Tests for claude_recall.searcher."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from claude_recall.db import (
    get_connection,
    upsert_chunks,
    upsert_graph_edges,
    upsert_session,
    upsert_session_commands,
    upsert_session_files,
)
from claude_recall.models import SearchResult, Session
from claude_recall.searcher import (
    _apply_depth_boost,
    _branch_search,
    _command_search,
    _cross_encoder_rerank,
    _file_search,
    _fts_search,
    _prepare_fts_query,
    _reciprocal_rank_fusion,
    search,
)


# ===========================================================================
# _prepare_fts_query
# ===========================================================================

class TestPrepareFtsQuery:
    def test_empty_query(self):
        assert _prepare_fts_query("") == ""

    def test_whitespace_query(self):
        assert _prepare_fts_query("   ") == ""

    def test_single_keyword(self):
        result = _prepare_fts_query("authentication")
        # Single terms use exact match only (prefix adds noise via stemming)
        assert '"authentication"' in result

    def test_single_keyword_no_prefix(self):
        result = _prepare_fts_query("authentication", use_prefix=False)
        assert result == '"authentication"'

    def test_multiple_keywords_and_join(self):
        result = _prepare_fts_query("debug middleware")
        assert '"debug"' in result
        assert '"middleware"' in result
        assert " AND " in result

    def test_prefix_matching_in_multi_keyword(self):
        result = _prepare_fts_query("auth middleware")
        # Each term >= 3 chars gets prefix variant
        assert '"auth"*' in result
        assert '"middleware"*' in result

    def test_stop_words_filtered(self):
        result = _prepare_fts_query("the bug in authentication")
        # "the" and "in" are stop words
        assert '"the"' not in result
        assert '"in"' not in result
        assert '"bug"' in result
        assert '"authentication"' in result

    def test_all_stop_words_uses_or_fallback(self):
        result = _prepare_fts_query("the is it")
        # All are stop words; should fall back to OR with original words > 1 char
        assert "OR" in result

    def test_fts5_passthrough_and(self):
        query = 'auth AND middleware'
        result = _prepare_fts_query(query)
        # Should pass through because it contains " AND "
        assert result == query

    def test_fts5_passthrough_or(self):
        query = 'auth OR middleware'
        result = _prepare_fts_query(query)
        assert result == query

    def test_fts5_passthrough_not(self):
        query = 'auth NOT test'
        result = _prepare_fts_query(query)
        assert result == query

    def test_fts5_passthrough_quotes(self):
        query = '"exact phrase"'
        result = _prepare_fts_query(query)
        assert result == query

    def test_special_chars_escaped_by_quoting(self):
        """Colons, parens, asterisks should be safe inside quotes."""
        result = _prepare_fts_query("file:main.py")
        # Should be quoted to escape the colon
        assert '"file:main.py"' in result

    def test_short_terms_filtered(self):
        result = _prepare_fts_query("a b debug")
        # single-char terms should be filtered
        assert '"debug"' in result

    def test_single_short_term_passthrough(self):
        """If all terms are short/stopwords, fall back gracefully."""
        result = _prepare_fts_query("a")
        # With only single-char terms, should return original query
        assert result == "a"


# ===========================================================================
# search (integration via FTS)
# ===========================================================================

class TestSearch:
    @pytest.fixture
    def search_db(self, db_path: Path):
        """Create a DB with sessions for search testing."""
        conn = get_connection(db_path)

        sessions = [
            Session(
                session_id="s1",
                project_path="/Users/test/myapp",
                project_dir="-Users-test-myapp",
                file_path="/tmp/s1.jsonl",
                summary="Auth middleware debugging",
                first_prompt="Debug the auth middleware",
                first_reply="Looking at it now.",
                last_prompt="Add error handling",
                last_reply="Done.",
                messages_text="Debug the auth middleware token validation error handling",
                message_count=5,
                file_size=1024,
                created="2025-01-15T10:00:00Z",
                modified="2025-01-15T11:00:00Z",
                mtime=1736935800.0,
                is_subagent=False,
            ),
            Session(
                session_id="s2",
                project_path="/Users/test/webapp",
                project_dir="-Users-test-webapp",
                file_path="/tmp/s2.jsonl",
                summary="React router setup",
                first_prompt="Set up routing",
                first_reply="Configuring routes.",
                last_prompt="Add 404 page",
                last_reply="Done.",
                messages_text="Set up routing navigation React router component",
                message_count=8,
                file_size=2048,
                created="2025-02-01T08:00:00Z",
                modified="2025-02-01T10:00:00Z",
                mtime=1738396800.0,
                is_subagent=False,
            ),
            Session(
                session_id="s3_sub",
                project_path="/Users/test/myapp",
                project_dir="-Users-test-myapp",
                file_path="/tmp/s3.jsonl",
                summary="Linter run",
                first_prompt="Run linter",
                first_reply="Done.",
                messages_text="Run linter check",
                message_count=1,
                file_size=256,
                created="2025-01-15T10:30:00Z",
                modified="2025-01-15T10:31:00Z",
                mtime=1736937060.0,
                is_subagent=True,
                parent_session="s1",
            ),
        ]

        for s in sessions:
            upsert_session(conn, s)

        upsert_chunks(conn, "s1", ["auth middleware debugging tokens"])
        upsert_chunks(conn, "s2", ["react router setup navigation"])
        upsert_chunks(conn, "s3_sub", ["linter check"])

        conn.commit()
        conn.close()
        return db_path

    def test_keyword_search_returns_results(self, search_db):
        results = search("auth middleware", db_path=search_db, semantic=False)
        assert len(results) >= 1
        assert results[0].session.session_id == "s1"

    def test_empty_query_returns_empty(self, search_db):
        assert search("", db_path=search_db) == []

    def test_whitespace_query_returns_empty(self, search_db):
        assert search("   ", db_path=search_db) == []

    def test_nonsense_query_returns_empty(self, search_db):
        results = search("xyzzyplugh9876543", db_path=search_db, semantic=False)
        assert results == []

    def test_project_filter(self, search_db):
        results = search("middleware router", db_path=search_db, semantic=False, project_filter="webapp")
        for r in results:
            assert "webapp" in r.session.project_path

    def test_date_after_filter(self, search_db):
        results = search(
            "routing",
            db_path=search_db,
            semantic=False,
            after="2025-01-20",
        )
        for r in results:
            assert r.session.modified >= "2025-01-20"

    def test_date_before_filter(self, search_db):
        results = search(
            "auth",
            db_path=search_db,
            semantic=False,
            before="2025-01-31",
        )
        for r in results:
            assert r.session.modified <= "2025-01-31"

    def test_min_messages_filter(self, search_db):
        results = search(
            "auth middleware routing",
            db_path=search_db,
            semantic=False,
            min_messages=6,
        )
        for r in results:
            assert r.session.message_count >= 6

    def test_subagent_excluded(self, search_db):
        """Subagent sessions should be excluded from results."""
        results = search("linter", db_path=search_db, semantic=False)
        for r in results:
            assert not r.session.is_subagent

    def test_results_have_scores(self, search_db):
        results = search("auth middleware", db_path=search_db, semantic=False)
        for r in results:
            assert isinstance(r.score, float)
            assert r.score >= 0

    def test_fts_special_chars_safe(self, search_db):
        """Special FTS5 characters should not crash the search."""
        for query in ["file:main.py", "func()", "test*", "a:b:c", "(parens)"]:
            # Should not raise
            results = search(query, db_path=search_db, semantic=False)
            assert isinstance(results, list)


# ===========================================================================
# _fts_search
# ===========================================================================

class TestFtsSearch:
    def test_returns_search_results(self, db_path):
        conn = get_connection(db_path)
        s = Session(
            session_id="fts1",
            project_path="/test",
            project_dir="test",
            file_path="/tmp/fts1.jsonl",
            first_prompt="python decorators",
            messages_text="python decorators metaclass",
            message_count=3,
            is_subagent=False,
        )
        upsert_session(conn, s)
        conn.commit()

        results = _fts_search(conn, '"python decorators"', 10, None, None, None, 1)
        conn.close()

        assert len(results) >= 1
        assert results[0].session.session_id == "fts1"

    def test_bm25_ranking(self, db_path):
        """Sessions with more keyword matches should rank higher."""
        conn = get_connection(db_path)

        # Session with many keyword occurrences
        s1 = Session(
            session_id="bm1",
            project_path="/test",
            project_dir="test",
            file_path="/tmp/bm1.jsonl",
            summary="python python python",
            first_prompt="python decorators advanced python",
            messages_text="python decorators metaclass python pattern python",
            message_count=5,
            is_subagent=False,
        )
        # Session with fewer occurrences
        s2 = Session(
            session_id="bm2",
            project_path="/test",
            project_dir="test",
            file_path="/tmp/bm2.jsonl",
            first_prompt="javascript basics",
            messages_text="javascript basics and also python once",
            message_count=3,
            is_subagent=False,
        )
        upsert_session(conn, s1)
        upsert_session(conn, s2)
        conn.commit()

        results = _fts_search(conn, '"python"', 10, None, None, None, 1)
        conn.close()

        assert len(results) >= 2
        # s1 should rank higher (more "python" occurrences)
        ids = [r.session.session_id for r in results]
        assert ids[0] == "bm1"


# ===========================================================================
# _reciprocal_rank_fusion
# ===========================================================================

class TestReciprocalRankFusion:
    def _make_result(self, session_id: str, score: float = 0.0) -> SearchResult:
        s = Session(
            session_id=session_id,
            project_path="/test",
            project_dir="test",
            file_path=f"/tmp/{session_id}.jsonl",
        )
        return SearchResult(session=s, score=score)

    def test_empty_inputs(self):
        result = _reciprocal_rank_fusion([], [])
        assert result == []

    def test_fts_only(self):
        fts = [self._make_result("a"), self._make_result("b")]
        result = _reciprocal_rank_fusion(fts, [])
        assert len(result) == 2

    def test_vec_only(self):
        vec = [self._make_result("a"), self._make_result("b")]
        result = _reciprocal_rank_fusion([], vec)
        assert len(result) == 2

    def test_combines_results(self):
        fts = [self._make_result("a"), self._make_result("b")]
        vec = [self._make_result("b"), self._make_result("c")]
        result = _reciprocal_rank_fusion(fts, vec)
        ids = {r.session.session_id for r in result}
        assert ids == {"a", "b", "c"}

    def test_shared_result_scores_higher(self):
        """A result in both FTS and vec should score higher than one in only one."""
        fts = [self._make_result("shared"), self._make_result("fts_only")]
        vec = [self._make_result("shared"), self._make_result("vec_only")]
        result = _reciprocal_rank_fusion(fts, vec, alpha=0.5)

        # "shared" should be first since it appears in both
        assert result[0].session.session_id == "shared"

    def test_scores_normalized_to_01(self):
        fts = [self._make_result("a"), self._make_result("b")]
        vec = [self._make_result("c")]
        result = _reciprocal_rank_fusion(fts, vec)
        scores = [r.score for r in result]
        assert max(scores) == pytest.approx(1.0)
        assert min(scores) >= 0.0

    def test_alpha_weighting(self):
        """Higher alpha should weight FTS more."""
        fts = [self._make_result("fts_top")]
        vec = [self._make_result("vec_top")]

        result_fts_heavy = _reciprocal_rank_fusion(fts, vec, alpha=0.9)
        # With alpha=0.9, fts_top should rank first
        assert result_fts_heavy[0].session.session_id == "fts_top"

        result_vec_heavy = _reciprocal_rank_fusion(fts, vec, alpha=0.1)
        # With alpha=0.1, vec_top should rank first
        assert result_vec_heavy[0].session.session_id == "vec_top"


# ===========================================================================
# _cross_encoder_rerank
# ===========================================================================

class TestCrossEncoderRerank:
    def _make_result(self, session_id: str, prompt: str, score: float = 0.5) -> SearchResult:
        s = Session(
            session_id=session_id,
            project_path="/test",
            project_dir="test",
            file_path=f"/tmp/{session_id}.jsonl",
            first_prompt=prompt,
        )
        return SearchResult(session=s, score=score)

    def test_empty_results(self):
        assert _cross_encoder_rerank("query", []) == []

    def test_single_result_unchanged(self):
        r = self._make_result("a", "hello")
        result = _cross_encoder_rerank("hello", [r])
        assert len(result) == 1

    @patch("claude_recall.embedder.get_reranker")
    def test_reranker_reorders(self, mock_get_reranker):
        """Mock reranker should reorder results."""
        mock_reranker = MagicMock()
        # Return items in reversed order with scores
        mock_reranker.rerank.return_value = [
            (1, 0.95),  # second result is now first
            (0, 0.30),  # first result is now second
        ]
        mock_get_reranker.return_value = mock_reranker

        r0 = self._make_result("a", "auth debugging")
        r1 = self._make_result("b", "middleware fix")
        result = _cross_encoder_rerank("fixing the middleware auth bug in production", [r0, r1])

        assert len(result) >= 1
        assert result[0].session.session_id == "b"

    @patch("claude_recall.embedder.get_reranker")
    def test_relevance_cutoff(self, mock_get_reranker):
        """Low-scoring results should be dropped if top score is strong."""
        mock_reranker = MagicMock()
        mock_reranker.rerank.return_value = [
            (0, 0.9),   # strong match
            (1, 0.1),   # very weak match — below 40% of top
        ]
        mock_get_reranker.return_value = mock_reranker

        r0 = self._make_result("strong", "auth debug")
        r1 = self._make_result("weak", "unrelated")
        r1.session.project_dir = "different_project"  # different project so cutoff applies
        result = _cross_encoder_rerank("debugging the authentication flow in production", [r0, r1])

        # The weak result should be dropped (0.1 is less than 0.4 * 0.9 normalized)
        # After normalization: strong=1.0, weak=0.0; cutoff = 1.0 * 0.4 = 0.4
        # weak (0.0) < 0.4 so it should be dropped
        ids = [r.session.session_id for r in result]
        assert "strong" in ids
        assert "weak" not in ids

    @patch("claude_recall.embedder.get_reranker", return_value=None)
    def test_no_reranker_returns_unchanged(self, mock_get_reranker):
        """If reranker is not available, return results unchanged."""
        r0 = self._make_result("a", "hello")
        r1 = self._make_result("b", "world")
        result = _cross_encoder_rerank("test", [r0, r1])
        assert len(result) == 2
        assert result[0].session.session_id == "a"

    @patch("claude_recall.embedder.get_reranker")
    def test_reranker_exception_returns_unchanged(self, mock_get_reranker):
        """If reranker raises, return results unchanged."""
        mock_reranker = MagicMock()
        mock_reranker.rerank.side_effect = RuntimeError("model error")
        mock_get_reranker.return_value = mock_reranker

        r0 = self._make_result("a", "hello")
        result = _cross_encoder_rerank("test", [r0])
        # Single result, should return as-is
        assert len(result) == 1


# ===========================================================================
# _apply_depth_boost
# ===========================================================================

class TestApplyDepthBoost:
    def _make_result(self, session_id: str, message_count: int, score: float) -> SearchResult:
        s = Session(
            session_id=session_id,
            project_path="/test",
            project_dir="test",
            file_path=f"/tmp/{session_id}.jsonl",
            message_count=message_count,
        )
        return SearchResult(session=s, score=score)

    def test_single_message_penalized(self):
        r = self._make_result("a", 1, 1.0)
        _apply_depth_boost([r])
        # 1 msg → log2(1)=0, base boost=1.0, then 0.5x penalty for automated
        assert r.score == pytest.approx(0.5)

    def test_multi_message_gets_boost(self):
        r1 = self._make_result("a", 1, 1.0)
        r10 = self._make_result("b", 10, 1.0)
        _apply_depth_boost([r1, r10])
        # 10 msgs → log2(10)≈3.32, boost≈1.332
        assert r10.score > r1.score

    def test_boost_is_mild(self):
        r = self._make_result("a", 100, 1.0)
        _apply_depth_boost([r])
        # Even 100 msgs shouldn't boost more than ~1.7x
        assert r.score < 2.0

    def test_preserves_relative_ordering_from_score(self):
        """A strong-scoring 1-msg session should still beat a weak-scoring 50-msg one."""
        r_strong = self._make_result("strong", 1, 1.0)
        r_weak = self._make_result("weak", 50, 0.3)
        _apply_depth_boost([r_strong, r_weak])
        assert r_strong.score > r_weak.score


# ===========================================================================
# _prepare_fts_query prefix matching
# ===========================================================================

class TestPrepareFtsQueryPrefix:
    def test_prefix_disabled(self):
        result = _prepare_fts_query("auth middleware", use_prefix=False)
        assert "*" not in result
        assert '"auth"' in result
        assert '"middleware"' in result

    def test_short_terms_no_prefix(self):
        """Terms shorter than 3 chars should not get prefix matching."""
        result = _prepare_fts_query("go db")
        assert '"go"' in result
        assert '"db"' in result
        # short terms don't get prefix
        assert '"go"*' not in result
        assert '"db"*' not in result


# ===========================================================================
# Integration: summary in search results
# ===========================================================================

class TestSearchWithSummary:
    @pytest.fixture
    def search_db_with_summary(self, db_path: Path):
        """Create a DB with sessions that have summaries."""
        conn = get_connection(db_path)

        sessions = [
            Session(
                session_id="sum1",
                project_path="/Users/test/myapp",
                project_dir="-Users-test-myapp",
                file_path="/tmp/sum1.jsonl",
                summary="Debugging JWT authentication token validation errors",
                first_prompt="Help me debug the auth",
                first_reply="Looking at the JWT validation code.",
                messages_text="auth debug jwt token validation",
                message_count=15,
                file_size=4096,
                created="2025-01-15T10:00:00Z",
                modified="2025-01-15T11:00:00Z",
                mtime=1736935800.0,
                is_subagent=False,
            ),
            Session(
                session_id="sum2",
                project_path="/Users/test/myapp",
                project_dir="-Users-test-myapp",
                file_path="/tmp/sum2.jsonl",
                summary="Setting up PostgreSQL database migration scripts",
                first_prompt="Create db migration",
                first_reply="Creating Alembic migration.",
                messages_text="database migration postgresql alembic",
                message_count=8,
                file_size=2048,
                created="2025-02-01T08:00:00Z",
                modified="2025-02-01T10:00:00Z",
                mtime=1738396800.0,
                is_subagent=False,
            ),
        ]

        for s in sessions:
            upsert_session(conn, s)
        conn.commit()
        conn.close()
        return db_path

    def test_summary_boosts_relevance(self, search_db_with_summary):
        """Sessions with matching summaries should rank high."""
        results = search("JWT authentication", db_path=search_db_with_summary, semantic=False)
        assert len(results) >= 1
        assert results[0].session.session_id == "sum1"

    def test_summary_keyword_in_results(self, search_db_with_summary):
        """Searching for terms only in summary should still find results."""
        results = search("PostgreSQL", db_path=search_db_with_summary, semantic=False)
        assert len(results) >= 1
        ids = [r.session.session_id for r in results]
        assert "sum2" in ids


# ===========================================================================
# Structured search: file:, cmd:, branch:
# ===========================================================================

class TestStructuredSearch:
    @pytest.fixture
    def graph_db(self, db_path: Path):
        """Create a DB with sessions, files, commands for structured search."""
        conn = get_connection(db_path)

        s1 = Session(
            session_id="gs1",
            project_path="/Users/test/myapp",
            project_dir="-Users-test-myapp",
            file_path="/tmp/gs1.jsonl",
            summary="Auth middleware work",
            first_prompt="Debug auth",
            messages_text="debug auth middleware",
            git_branch="fix/auth",
            git_branch_detected="fix/auth",
            message_count=5,
            file_size=1024,
            created="2025-01-15T10:00:00Z",
            modified="2025-01-15T11:00:00Z",
            mtime=1736935800.0,
            is_subagent=False,
        )
        s2 = Session(
            session_id="gs2",
            project_path="/Users/test/webapp",
            project_dir="-Users-test-webapp",
            file_path="/tmp/gs2.jsonl",
            summary="React routing setup",
            first_prompt="Set up routes",
            messages_text="react router setup",
            git_branch="feature/routing",
            git_branch_detected="feature/routing",
            message_count=8,
            file_size=2048,
            created="2025-02-01T08:00:00Z",
            modified="2025-02-01T10:00:00Z",
            mtime=1738396800.0,
            is_subagent=False,
        )
        upsert_session(conn, s1)
        upsert_session(conn, s2)

        # Files
        upsert_session_files(conn, "gs1", [
            {"path": "src/auth.py", "name": "auth.py", "action": "edit"},
            {"path": "tests/test_auth.py", "name": "test_auth.py", "action": "edit"},
        ])
        upsert_session_files(conn, "gs2", [
            {"path": "src/router.tsx", "name": "router.tsx", "action": "edit"},
        ])

        # Commands
        upsert_session_commands(conn, "gs1", [
            {"command": "pytest tests/", "command_name": "pytest"},
            {"command": "git diff", "command_name": "git"},
        ])
        upsert_session_commands(conn, "gs2", [
            {"command": "npm test", "command_name": "npm"},
        ])

        # Graph edges
        upsert_graph_edges(conn, "gs1", [
            {"src_type": "session", "src_name": "gs1",
             "dst_type": "file", "dst_name": "auth.py", "rel": "edited"},
            {"src_type": "session", "src_name": "gs1",
             "dst_type": "file", "dst_name": "test_auth.py", "rel": "edited"},
        ])
        upsert_graph_edges(conn, "gs2", [
            {"src_type": "session", "src_name": "gs2",
             "dst_type": "file", "dst_name": "router.tsx", "rel": "edited"},
        ])

        conn.commit()
        conn.close()
        return db_path

    def test_file_search_by_name(self, graph_db):
        """file: prefix should search by file name."""
        results = search("file:auth.py", db_path=graph_db, semantic=False)
        assert len(results) >= 1
        assert results[0].session.session_id == "gs1"

    def test_file_search_by_path(self, graph_db):
        """file: prefix should also match partial paths."""
        results = search("file:src/auth", db_path=graph_db, semantic=False)
        assert len(results) >= 1
        ids = [r.session.session_id for r in results]
        assert "gs1" in ids

    def test_file_search_no_results(self, graph_db):
        """file: prefix with no matching file should return empty."""
        results = search("file:nonexistent.py", db_path=graph_db, semantic=False)
        assert results == []

    def test_file_search_snippets(self, graph_db):
        """file: results should have file path in snippets."""
        results = search("file:auth.py", db_path=graph_db, semantic=False)
        assert len(results) >= 1
        assert any("auth.py" in s for s in results[0].snippets)

    def test_command_search(self, graph_db):
        """cmd: prefix should search by command name."""
        results = search("cmd:pytest", db_path=graph_db, semantic=False)
        assert len(results) >= 1
        assert results[0].session.session_id == "gs1"

    def test_command_search_full(self, graph_db):
        """cmd: prefix should also match the full command string."""
        results = search("cmd:npm test", db_path=graph_db, semantic=False)
        assert len(results) >= 1
        assert results[0].session.session_id == "gs2"

    def test_branch_search(self, graph_db):
        """branch: prefix should search by git branch."""
        results = search("branch:fix/auth", db_path=graph_db, semantic=False)
        assert len(results) >= 1
        assert results[0].session.session_id == "gs1"

    def test_branch_search_partial(self, graph_db):
        """branch: prefix should match partial branch names."""
        results = search("branch:routing", db_path=graph_db, semantic=False)
        assert len(results) >= 1
        assert results[0].session.session_id == "gs2"

    def test_branch_search_no_results(self, graph_db):
        """branch: prefix with no matching branch should return empty."""
        results = search("branch:nonexistent", db_path=graph_db, semantic=False)
        assert results == []

    def test_file_search_with_project_filter(self, graph_db):
        """file: search should respect project filter."""
        results = search(
            "file:auth.py",
            db_path=graph_db,
            semantic=False,
            project_filter="webapp",
        )
        # auth.py only exists in myapp, not webapp
        assert results == []

    def test_file_search_direct(self, graph_db):
        """_file_search function should work directly."""
        conn = get_connection(graph_db)
        results = _file_search(conn, "router.tsx", 10, None)
        conn.close()
        assert len(results) >= 1
        assert results[0].session.session_id == "gs2"

    def test_command_search_direct(self, graph_db):
        """_command_search function should work directly."""
        conn = get_connection(graph_db)
        results = _command_search(conn, "git", 10, None)
        conn.close()
        assert len(results) >= 1
        assert results[0].session.session_id == "gs1"

    def test_branch_search_direct(self, graph_db):
        """_branch_search function should work directly."""
        conn = get_connection(graph_db)
        results = _branch_search(conn, "feature", 10, None)
        conn.close()
        assert len(results) >= 1
        assert results[0].session.session_id == "gs2"
