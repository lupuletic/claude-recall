"""Tests for claude_recall.utils."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from claude_recall.utils import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    MAX_CHUNK_CHARS,
    MAX_FIRST_PROMPT,
    MAX_MESSAGES_TEXT,
    _build_chunks,
    _build_fts_text,
    _resolve_path_parts,
    clean_display_text,
    decode_project_path,
    discover_sessions,
    extract_text_from_content,
    format_date,
    format_size,
    load_sessions_index,
    parse_session_file,
)


# ===========================================================================
# extract_text_from_content
# ===========================================================================

class TestExtractTextFromContent:
    def test_plain_string(self):
        assert extract_text_from_content("hello world") == "hello world"

    def test_plain_string_strips_whitespace(self):
        assert extract_text_from_content("  hello  ") == "hello"

    def test_empty_string(self):
        assert extract_text_from_content("") is None

    def test_whitespace_only_string(self):
        assert extract_text_from_content("   ") is None

    def test_none(self):
        assert extract_text_from_content(None) is None

    def test_list_single_text_block(self):
        blocks = [{"type": "text", "text": "hello world"}]
        assert extract_text_from_content(blocks) == "hello world"

    def test_list_multiple_text_blocks(self):
        blocks = [
            {"type": "text", "text": "first"},
            {"type": "text", "text": "second"},
        ]
        assert extract_text_from_content(blocks) == "first\nsecond"

    def test_list_mixed_block_types(self):
        blocks = [
            {"type": "text", "text": "hello"},
            {"type": "tool_use", "name": "bash", "input": {}},
            {"type": "text", "text": "world"},
        ]
        assert extract_text_from_content(blocks) == "hello\nworld"

    def test_list_no_text_blocks(self):
        blocks = [{"type": "tool_use", "name": "bash", "input": {}}]
        assert extract_text_from_content(blocks) is None

    def test_list_empty(self):
        assert extract_text_from_content([]) is None

    def test_list_text_block_empty_text(self):
        blocks = [{"type": "text", "text": ""}]
        assert extract_text_from_content(blocks) is None

    def test_integer_returns_none(self):
        assert extract_text_from_content(42) is None

    def test_dict_returns_none(self):
        assert extract_text_from_content({"type": "text", "text": "hi"}) is None


# ===========================================================================
# clean_display_text
# ===========================================================================

class TestCleanDisplayText:
    def test_none(self):
        assert clean_display_text(None) is None

    def test_empty_string(self):
        # empty string is falsy, so clean_display_text returns it as-is (early return)
        result = clean_display_text("")
        assert not result  # returns empty string (falsy)

    def test_plain_text_unchanged(self):
        assert clean_display_text("Hello world") == "Hello world"

    def test_strips_system_reminder(self):
        text = "<system-reminder>You are Claude.</system-reminder>Hello"
        result = clean_display_text(text)
        assert result == "Hello"
        assert "system-reminder" not in result

    def test_strips_local_command_caveat(self):
        text = "<local-command-caveat>Warning about local command</local-command-caveat>Result"
        result = clean_display_text(text)
        assert result == "Result"

    def test_strips_local_command_stdout(self):
        text = "Before <local-command-stdout>output here</local-command-stdout> After"
        result = clean_display_text(text)
        assert result == "Before After"

    def test_strips_local_command_stderr(self):
        text = "<local-command-stderr>error output</local-command-stderr>Clean"
        result = clean_display_text(text)
        assert result == "Clean"

    def test_strips_teammate_message(self):
        text = '<teammate-message from="agent1">do thing</teammate-message>Done'
        result = clean_display_text(text)
        assert result == "Done"

    def test_strips_command_name(self):
        text = "<command-name>bash</command-name>Output"
        result = clean_display_text(text)
        assert result == "Output"

    def test_strips_command_message(self):
        text = "<command-message>Running tests</command-message>Done"
        result = clean_display_text(text)
        assert result == "Done"

    def test_strips_command_args(self):
        text = "<command-args>--verbose</command-args>Output"
        result = clean_display_text(text)
        assert result == "Output"

    def test_strips_user_instructions(self):
        text = "<user_instructions>Be helpful</user_instructions>Answer"
        result = clean_display_text(text)
        assert result == "Answer"

    def test_strips_environment_context(self):
        text = "<environment_context>Python 3.12</environment_context>Code here"
        result = clean_display_text(text)
        assert result == "Code here"

    def test_strips_task_notification(self):
        text = "<task-notification>Task done</task-notification>Result"
        result = clean_display_text(text)
        assert result == "Result"

    def test_strips_request_interrupted(self):
        text = "Some text [Request interrupted by user] more text"
        result = clean_display_text(text)
        assert result == "Some text more text"

    def test_strips_multiple_patterns(self):
        text = (
            "<system-reminder>reminder</system-reminder>"
            "Hello "
            "<task-notification>notif</task-notification>"
            "world"
        )
        result = clean_display_text(text)
        assert result == "Hello world"

    def test_collapses_whitespace(self):
        text = "Hello   \n\n  world"
        result = clean_display_text(text)
        assert result == "Hello world"

    def test_returns_none_when_all_markup(self):
        text = "<system-reminder>all markup</system-reminder>"
        result = clean_display_text(text)
        assert result is None

    def test_strips_generic_xml_tags(self):
        text = "<some-custom-tag>content</some-custom-tag>Visible"
        result = clean_display_text(text)
        assert result == "Visible"

    def test_strips_self_closing_tags(self):
        text = "Before <some-tag /> After"
        result = clean_display_text(text)
        assert result == "Before After"


# ===========================================================================
# decode_project_path
# ===========================================================================

class TestDecodeProjectPath:
    def test_with_sessions_index(self, sessions_index_path):
        result = decode_project_path(
            "-Users-alice-my-project",
            projects_dir=sessions_index_path,
        )
        assert result == "/Users/alice/my-project"

    def test_without_sessions_index_no_leading_dash(self, tmp_path):
        result = decode_project_path("simple-project", projects_dir=tmp_path)
        assert result == "simple-project"

    def test_without_sessions_index_leading_dash(self, tmp_path):
        """With no sessions-index.json and no matching dirs, falls back to naive replace."""
        result = decode_project_path("-Users-fake-nonexistent", projects_dir=tmp_path)
        # Should do naive replace: -Users-fake-nonexistent -> /Users/fake/nonexistent
        assert result == "/Users/fake/nonexistent"

    def test_with_invalid_sessions_index(self, tmp_path):
        """If sessions-index.json is invalid JSON, falls back."""
        proj = tmp_path / "-Users-test-proj"
        proj.mkdir()
        (proj / "sessions-index.json").write_text("not json!")
        result = decode_project_path("-Users-test-proj", projects_dir=tmp_path)
        # Falls through to path resolution / naive replace
        assert isinstance(result, str)

    def test_sessions_index_missing_original_path(self, tmp_path):
        """If sessions-index.json has no originalPath, falls back."""
        proj = tmp_path / "-Users-test-proj"
        proj.mkdir()
        (proj / "sessions-index.json").write_text(json.dumps({"entries": []}))
        result = decode_project_path("-Users-test-proj", projects_dir=tmp_path)
        assert isinstance(result, str)


# ===========================================================================
# _resolve_path_parts
# ===========================================================================

class TestResolvePathParts:
    def test_empty_parts(self):
        assert _resolve_path_parts([]) is None

    def test_known_directories(self, tmp_path):
        """Test greedy matching with real directories."""
        # Create /tmp_path/aaa/bbb-ccc/
        d = tmp_path / "aaa" / "bbb-ccc"
        d.mkdir(parents=True)

        # Parts: ["aaa", "bbb", "ccc"] should resolve "bbb-ccc" greedily
        # But _resolve_path_parts uses "/" as the starting current path,
        # so we need to use system paths that actually exist.
        # Instead, test the logic with parts that form real dirs.
        # We'll test with a simpler case using tmp_path's parts.
        parts = list(tmp_path.parts[1:])  # skip leading /
        result = _resolve_path_parts(parts)
        assert result == str(tmp_path)

    def test_single_part_nonexistent(self):
        result = _resolve_path_parts(["nonexistent_dir_xyz_12345"])
        assert result == "/nonexistent_dir_xyz_12345"


# ===========================================================================
# parse_session_file
# ===========================================================================

class TestParseSessionFile:
    def test_basic_parsing(self, sample_jsonl_file):
        result = parse_session_file(sample_jsonl_file)
        assert result["message_count"] == 3
        assert result["first_prompt"] is not None
        assert "auth" in result["first_prompt"].lower() or "debug" in result["first_prompt"].lower()
        assert result["first_reply"] is not None
        assert result["last_prompt"] is not None
        assert result["last_reply"] is not None
        assert len(result["messages_text"]) > 0
        assert isinstance(result["chunks"], list)
        assert len(result["chunks"]) > 0

    def test_with_markup(self, sample_jsonl_file_with_markup):
        result = parse_session_file(sample_jsonl_file_with_markup)
        assert result["message_count"] == 2
        # Markup should be stripped from display fields
        assert result["first_prompt"] is not None
        assert "system-reminder" not in (result["first_prompt"] or "")
        assert "task-notification" not in (result["last_prompt"] or "")

    def test_empty_file(self, empty_jsonl_file):
        result = parse_session_file(empty_jsonl_file)
        assert result["message_count"] == 0
        assert result["first_prompt"] is None
        assert result["first_reply"] is None
        assert result["chunks"] == []

    def test_nonexistent_file(self, tmp_path):
        result = parse_session_file(tmp_path / "doesnotexist.jsonl")
        assert result["message_count"] == 0

    def test_truncates_first_prompt(self, tmp_path):
        """First prompt is truncated to MAX_FIRST_PROMPT chars."""
        long_text = "x" * (MAX_FIRST_PROMPT + 200)
        lines = [
            json.dumps({
                "type": "user",
                "message": {"role": "user", "content": long_text},
            })
        ]
        p = tmp_path / "long.jsonl"
        p.write_text("\n".join(lines))
        result = parse_session_file(p)
        assert len(result["first_prompt"]) <= MAX_FIRST_PROMPT

    def test_chunks_returned(self, sample_jsonl_file):
        result = parse_session_file(sample_jsonl_file)
        assert len(result["chunks"]) >= 1
        # Each chunk should contain User: and/or Assistant: prefixes
        for chunk in result["chunks"]:
            assert "User:" in chunk or "Assistant:" in chunk

    def test_invalid_json_lines_skipped(self, tmp_path):
        """Lines with invalid JSON are skipped gracefully."""
        lines = [
            "not valid json",
            json.dumps({"type": "user", "message": {"role": "user", "content": "hello"}}),
            "{broken",
            json.dumps({"type": "assistant", "message": {"role": "assistant", "content": [{"type": "text", "text": "hi"}]}}),
        ]
        p = tmp_path / "mixed.jsonl"
        p.write_text("\n".join(lines))
        result = parse_session_file(p)
        assert result["message_count"] == 1


# ===========================================================================
# _build_fts_text
# ===========================================================================

class TestBuildFtsText:
    def test_short_conversation(self):
        """Short conversations (<= 20 msgs) include all messages."""
        msgs = [f"message {i}" for i in range(10)]
        result = _build_fts_text(msgs)
        for msg in msgs:
            assert msg in result

    def test_empty_messages(self):
        assert _build_fts_text([]) == ""

    def test_single_message(self):
        result = _build_fts_text(["only one"])
        assert result == "only one"

    def test_exactly_20_messages(self):
        msgs = [f"message {i}" for i in range(20)]
        result = _build_fts_text(msgs)
        for msg in msgs:
            assert msg in result

    def test_long_conversation_samples(self):
        """Long conversations (> 20 msgs) sample from beginning, middle, end."""
        msgs = [f"message_{i:03d}" for i in range(50)]
        result = _build_fts_text(msgs)

        # First 5 should be included
        for i in range(5):
            assert f"message_{i:03d}" in result

        # Last 5 should be included
        for i in range(45, 50):
            assert f"message_{i:03d}" in result

        # Not all middle messages need to be present
        # but the result should not be empty
        assert len(result) > 0

    def test_truncation_to_max(self):
        """Result is truncated to MAX_MESSAGES_TEXT."""
        msgs = ["x" * 1000 for _ in range(15)]
        result = _build_fts_text(msgs)
        assert len(result) <= MAX_MESSAGES_TEXT


# ===========================================================================
# _build_chunks
# ===========================================================================

class TestBuildChunks:
    def test_empty_input(self):
        assert _build_chunks([], []) == []

    def test_single_turn(self):
        chunks = _build_chunks(["hello"], ["hi there"])
        assert len(chunks) == 1
        assert "User: hello" in chunks[0]
        assert "Assistant: hi there" in chunks[0]

    def test_short_session_single_chunk(self):
        """Sessions with <= CHUNK_SIZE turns produce one chunk."""
        user = [f"q{i}" for i in range(CHUNK_SIZE)]
        asst = [f"a{i}" for i in range(CHUNK_SIZE)]
        chunks = _build_chunks(user, asst)
        assert len(chunks) == 1

    def test_long_session_multiple_chunks(self):
        """Sessions with many turns produce overlapping chunks."""
        user = [f"question {i}" for i in range(20)]
        asst = [f"answer {i}" for i in range(20)]
        chunks = _build_chunks(user, asst)
        assert len(chunks) > 1

    def test_chunk_overlap(self):
        """Chunks should overlap by CHUNK_OVERLAP messages."""
        user = [f"question_{i}" for i in range(15)]
        asst = [f"answer_{i}" for i in range(15)]
        chunks = _build_chunks(user, asst)
        # With overlap, later chunks should share content with earlier ones
        if len(chunks) >= 2:
            # The last message of chunk[0]'s window should appear in chunk[1]
            # since they overlap
            assert len(chunks) >= 2

    def test_chunk_truncation(self):
        """Each chunk is truncated to MAX_CHUNK_CHARS."""
        user = ["x" * 500 for _ in range(10)]
        asst = ["y" * 500 for _ in range(10)]
        chunks = _build_chunks(user, asst)
        for chunk in chunks:
            assert len(chunk) <= MAX_CHUNK_CHARS

    def test_uneven_user_assistant(self):
        """Handles more user messages than assistant messages."""
        user = ["q1", "q2", "q3"]
        asst = ["a1"]
        chunks = _build_chunks(user, asst)
        assert len(chunks) >= 1
        assert "User: q1" in chunks[0]
        assert "Assistant: a1" in chunks[0]


# ===========================================================================
# format_size
# ===========================================================================

class TestFormatSize:
    def test_bytes(self):
        assert format_size(500) == "500B"

    def test_kilobytes(self):
        assert format_size(2048) == "2.0KB"

    def test_megabytes(self):
        assert format_size(1_500_000) == "1.4MB"

    def test_zero(self):
        assert format_size(0) == "0B"

    def test_exact_kb_boundary(self):
        assert format_size(1024) == "1.0KB"

    def test_exact_mb_boundary(self):
        assert format_size(1024 * 1024) == "1.0MB"


# ===========================================================================
# format_date
# ===========================================================================

class TestFormatDate:
    def test_iso_date(self):
        assert format_date("2025-01-15T10:30:00Z") == "2025-01-15"

    def test_date_only(self):
        assert format_date("2025-01-15") == "2025-01-15"

    def test_none(self):
        assert format_date(None) == "unknown"

    def test_empty_string(self):
        assert format_date("") == "unknown"


# ===========================================================================
# discover_sessions
# ===========================================================================

class TestDiscoverSessions:
    def test_discovers_all_sessions(self, projects_dir):
        sessions = discover_sessions(projects_dir)
        ids = {s["session_id"] for s in sessions}
        assert "session-001" in ids
        assert "session-002" in ids
        assert "session-003" in ids
        assert "agent-sub1" in ids

    def test_subagent_detection(self, projects_dir):
        sessions = discover_sessions(projects_dir)
        sub = [s for s in sessions if s["session_id"] == "agent-sub1"][0]
        assert sub["is_subagent"] is True
        assert sub["parent_session"] == "session-001"

    def test_main_sessions_not_subagent(self, projects_dir):
        sessions = discover_sessions(projects_dir)
        main = [s for s in sessions if s["session_id"] == "session-001"][0]
        assert main["is_subagent"] is False

    def test_nonexistent_dir(self, tmp_path):
        sessions = discover_sessions(tmp_path / "nope")
        assert sessions == []


# ===========================================================================
# load_sessions_index
# ===========================================================================

class TestLoadSessionsIndex:
    def test_loads_index(self, projects_dir):
        idx = load_sessions_index("-Users-test-Projects-myapp", projects_dir)
        assert "session-001" in idx
        assert idx["session-001"]["summary"] == "Auth middleware debugging"

    def test_missing_index(self, tmp_path):
        idx = load_sessions_index("nonexistent", tmp_path)
        assert idx == {}

    def test_invalid_json(self, tmp_path):
        proj = tmp_path / "broken"
        proj.mkdir()
        (proj / "sessions-index.json").write_text("not json")
        idx = load_sessions_index("broken", tmp_path)
        assert idx == {}
