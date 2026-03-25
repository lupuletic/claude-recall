"""Utilities for parsing Claude Code session files."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

CLAUDE_DIR = Path.home() / ".claude"
PROJECTS_DIR = CLAUDE_DIR / "projects"

# Max chars to keep for indexed text fields
MAX_FIRST_PROMPT = 500
MAX_FIRST_REPLY = 500
MAX_MESSAGES_TEXT = 50_000


def decode_project_path(project_dir: str, projects_dir: Path = PROJECTS_DIR) -> str:
    """Decode an encoded project directory name back to a filesystem path.

    Checks sessions-index.json for originalPath first (most reliable),
    then tries to reconstruct the path by testing which combination of
    dashes-as-slashes vs dashes-as-literal-dashes actually exists on disk.
    """
    # Best source: sessions-index.json has the original path
    idx_path = projects_dir / project_dir / "sessions-index.json"
    if idx_path.exists():
        try:
            with open(idx_path) as f:
                data = json.load(f)
                if data.get("originalPath"):
                    return data["originalPath"]
        except (json.JSONDecodeError, OSError):
            pass

    if not project_dir.startswith("-"):
        return project_dir

    # Smart decode: try to find which path actually exists on disk
    # The encoding replaces / with - but folder names can also contain -
    # So "-Users-foo-my-project" could be /Users/foo/my-project or /Users/foo/my/project
    # We try the most likely paths by splitting on - and testing existence
    parts = project_dir[1:].split("-")  # strip leading dash, split
    resolved = _resolve_path_parts(parts)
    if resolved:
        return resolved

    # Last resort: naive replace (may be wrong for paths with dashes)
    decoded = project_dir.replace("-", "/")
    if sys.platform == "win32" and len(decoded) > 2 and decoded[2] == "/":
        decoded = decoded[1] + ":" + decoded[2:]
    return decoded


def _resolve_path_parts(parts: list[str]) -> str | None:
    """Try to reconstruct a filesystem path from encoded parts.

    Greedily matches the longest existing directory at each level.
    e.g. ["Users", "foo", "claude", "hackathon"] tries:
      /Users → exists, consume
      /Users/foo → exists, consume
      /Users/foo/claude-hackathon → exists! consume both
    """
    if not parts:
        return None

    current = "/"
    i = 0

    while i < len(parts):
        # Try joining progressively more parts with dashes (longest match first)
        matched = False
        for end in range(len(parts), i, -1):
            candidate = "-".join(parts[i:end])
            test_path = os.path.join(current, candidate)
            if os.path.isdir(test_path):
                current = test_path
                i = end
                matched = True
                break

        if not matched:
            # No match found — use single part and continue
            current = os.path.join(current, parts[i])
            i += 1

    return current


import re

# Patterns to strip from displayed text (internal Claude Code markup)
_MARKUP_PATTERNS = [
    # Strip any XML/HTML-like tags that are internal Claude Code markup
    re.compile(r"<local-command-caveat>.*?</local-command-caveat>", re.DOTALL),
    re.compile(r"<local-command-stdout>.*?</local-command-stdout>", re.DOTALL),
    re.compile(r"<local-command-stderr>.*?</local-command-stderr>", re.DOTALL),
    re.compile(r"<teammate-message[^>]*>.*?</teammate-message>", re.DOTALL),
    re.compile(r"<command-name>.*?</command-name>", re.DOTALL),
    re.compile(r"<command-message>.*?</command-message>", re.DOTALL),
    re.compile(r"<command-args>.*?</command-args>", re.DOTALL),
    re.compile(r"<system-reminder>.*?</system-reminder>", re.DOTALL),
    re.compile(r"<user_instructions>.*?</user_instructions>", re.DOTALL),
    re.compile(r"<environment_context>.*?</environment_context>", re.DOTALL),
    re.compile(r"<task-notification>.*?</task-notification>", re.DOTALL),
    re.compile(r"\[Request interrupted by user\]"),
    # Catch any remaining XML-style tags
    re.compile(r"<[a-z_-]+(?:\s[^>]*)?>.*?</[a-z_-]+>", re.DOTALL),
    re.compile(r"<[a-z_-]+(?:\s[^>]*)?\s*/?>"),
]


def clean_display_text(text: str | None) -> str | None:
    """Strip internal Claude Code markup from text for display."""
    if not text:
        return text
    for pattern in _MARKUP_PATTERNS:
        text = pattern.sub("", text)
    # Collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text if text else None


def extract_text_from_content(content) -> str | None:
    """Extract readable text from a message content field.

    Content can be a plain string or a list of content blocks.
    """
    if isinstance(content, str):
        return content.strip() if content.strip() else None

    if isinstance(content, list):
        texts = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                text = block.get("text", "").strip()
                if text:
                    texts.append(text)
        return "\n".join(texts) if texts else None

    return None


def parse_session_file(file_path: str | Path) -> dict:
    """Parse a session .jsonl file and extract searchable content.

    Returns dict with:
        first_prompt, first_reply, last_prompt, last_reply: str | None
        messages_text: str  (sampled user+assistant messages for FTS)
        message_count: int
        chunks: list[str]  (conversation chunks for embedding)
        files_modified: list[str]  (file paths edited/written)
        commands_run: list[str]  (key bash commands)
        git_branch_detected: str | None
    """
    first_prompt = None
    first_reply = None
    last_prompt = None
    last_reply = None
    user_messages: list[str] = []
    assistant_texts: list[str] = []
    files_modified: set[str] = set()
    commands_run: list[str] = []
    git_branch_detected: str | None = None

    # Commands to skip (low signal)
    _SKIP_CMDS = {"cd", "ls", "cat", "echo", "pwd", "head", "tail", "wc", "true", "false"}

    try:
        with open(file_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue

                msg_type = obj.get("type")

                if msg_type == "user":
                    msg = obj.get("message", {})
                    text = extract_text_from_content(msg.get("content", ""))
                    if text:
                        user_messages.append(text)
                        # Use cleaned text for display fields
                        cleaned = clean_display_text(text)
                        if cleaned:
                            if first_prompt is None:
                                first_prompt = cleaned[:MAX_FIRST_PROMPT]
                            last_prompt = cleaned[:MAX_FIRST_PROMPT]

                elif msg_type == "assistant":
                    msg = obj.get("message", {})
                    content = msg.get("content", [])
                    text = extract_text_from_content(content)
                    if text:
                        assistant_texts.append(text)
                        cleaned = clean_display_text(text)
                        if cleaned:
                            if first_reply is None:
                                first_reply = cleaned[:MAX_FIRST_REPLY]
                            last_reply = cleaned[:MAX_FIRST_REPLY]

                    # Extract tool calls (files modified, commands run)
                    if isinstance(content, list):
                        for block in content:
                            if not isinstance(block, dict) or block.get("type") != "tool_use":
                                continue
                            tool = block.get("name", "")
                            inp = block.get("input", {})

                            if tool in ("Edit", "Write", "NotebookEdit"):
                                fp = inp.get("file_path", "")
                                if fp:
                                    # Store full file path for accurate graph edges
                                    files_modified.add(fp)

                            elif tool == "Bash":
                                cmd = inp.get("command", "").strip()
                                if cmd:
                                    # Extract first word (the actual command)
                                    first_word = cmd.split()[0] if cmd.split() else ""
                                    if first_word and first_word not in _SKIP_CMDS:
                                        commands_run.append(cmd[:80])
                                    # Detect git branch
                                    if not git_branch_detected:
                                        for pattern in ["git checkout ", "git switch "]:
                                            if pattern in cmd:
                                                branch = cmd.split(pattern)[-1].split()[0]
                                                git_branch_detected = branch

                # Detect git branch from session metadata
                if not git_branch_detected and obj.get("gitBranch"):
                    git_branch_detected = obj["gitBranch"]

    except OSError:
        pass

    # Build FTS text: smart sampling of user + assistant messages
    messages_text = _build_fts_text(user_messages, assistant_texts)

    # Build conversation chunks for embedding
    chunks = _build_chunks(user_messages, assistant_texts)

    # Auto-generate summary from first prompt + reply
    summary = generate_summary(first_prompt, first_reply)

    # Deduplicate commands by full command string and limit
    seen_cmds: set[str] = set()
    unique_cmds: list[str] = []
    for cmd in commands_run:
        if cmd not in seen_cmds:
            seen_cmds.add(cmd)
            unique_cmds.append(cmd)
    commands_run = unique_cmds[:30]

    return {
        "first_prompt": first_prompt,
        "first_reply": first_reply,
        "last_prompt": last_prompt,
        "last_reply": last_reply,
        "messages_text": messages_text,
        "message_count": len(user_messages),
        "chunks": chunks,
        "summary": summary,
        "files_modified": sorted(files_modified)[:50],
        "commands_run": commands_run,
        "git_branch_detected": git_branch_detected,
    }


# Chunk configuration
CHUNK_SIZE = 5  # messages per chunk
CHUNK_OVERLAP = 1  # overlapping messages between chunks
MAX_CHUNK_CHARS = 2000  # max chars per chunk text


def _build_fts_text(user_messages: list[str], assistant_texts: list[str] | None = None) -> str:
    """Build FTS-indexed text by sampling messages throughout the conversation.

    Interleaves user and assistant messages for richer keyword coverage.
    For short conversations (< 20 msgs): include everything.
    For longer ones: take first 5, every Nth from middle, and last 5.
    This ensures keywords from any part of the conversation are searchable.
    """
    # Interleave user and assistant messages for better coverage
    all_messages: list[str] = []
    assistant = assistant_texts or []
    for i in range(max(len(user_messages), len(assistant))):
        if i < len(user_messages) and user_messages[i].strip():
            all_messages.append(user_messages[i])
        if i < len(assistant) and assistant[i].strip():
            all_messages.append(assistant[i])

    n = len(all_messages)
    if n <= 20:
        text = "\n".join(all_messages)
    else:
        sampled = []
        sampled.extend(all_messages[:5])
        middle = all_messages[5:-5]
        step = max(1, len(middle) // 15)
        sampled.extend(middle[::step])
        sampled.extend(all_messages[-5:])
        text = "\n".join(sampled)

    if len(text) > MAX_MESSAGES_TEXT:
        text = text[:MAX_MESSAGES_TEXT]
    return text


def _build_chunks(
    user_messages: list[str],
    assistant_texts: list[str],
) -> list[str]:
    """Build conversation chunks for embedding.

    Interleaves user + assistant messages into turn pairs, then creates
    overlapping sliding windows. Research shows including assistant
    responses improves semantic retrieval (anchors what was discussed).

    Each chunk is a window of ~5 turn pairs (~512 tokens).
    """
    # Build interleaved turn pairs
    turns: list[str] = []
    for i in range(max(len(user_messages), len(assistant_texts))):
        parts = []
        if i < len(user_messages) and user_messages[i].strip():
            # Truncate individual messages to keep chunks balanced
            parts.append(f"User: {user_messages[i][:300]}")
        if i < len(assistant_texts) and assistant_texts[i].strip():
            parts.append(f"Assistant: {assistant_texts[i][:300]}")
        if parts:
            turns.append("\n".join(parts))

    if not turns:
        return []

    # For very short sessions, one chunk is enough
    if len(turns) <= CHUNK_SIZE:
        text = "\n\n".join(turns)
        return [text[:MAX_CHUNK_CHARS]]

    chunks = []
    step = max(1, CHUNK_SIZE - CHUNK_OVERLAP)

    for start in range(0, len(turns), step):
        end = min(start + CHUNK_SIZE, len(turns))
        window = turns[start:end]
        chunk_text = "\n\n".join(window)
        if len(chunk_text) > MAX_CHUNK_CHARS:
            chunk_text = chunk_text[:MAX_CHUNK_CHARS]
        if chunk_text.strip():
            chunks.append(chunk_text)

        if end >= len(turns):
            break

    return chunks


def load_sessions_index(project_dir: str, projects_dir: Path = PROJECTS_DIR) -> dict[str, dict]:
    """Load sessions-index.json for a project directory.

    Returns dict mapping session_id -> index entry.
    """
    idx_path = projects_dir / project_dir / "sessions-index.json"
    if not idx_path.exists():
        return {}

    try:
        with open(idx_path) as f:
            data = json.load(f)
            return {
                entry["sessionId"]: entry
                for entry in data.get("entries", [])
                if "sessionId" in entry
            }
    except (json.JSONDecodeError, OSError):
        return {}


def discover_sessions(projects_dir: Path = PROJECTS_DIR) -> list[dict]:
    """Walk the projects directory and discover all session .jsonl files.

    Returns list of dicts with:
        session_id, file_path, project_dir, is_subagent, parent_session,
        file_size, mtime
    """
    sessions = []

    if not projects_dir.exists():
        return sessions

    for root, dirs, files in os.walk(projects_dir):
        for filename in files:
            if not filename.endswith(".jsonl"):
                continue

            file_path = os.path.join(root, filename)
            session_id = filename[:-6]  # strip .jsonl

            # Determine project dir and subagent status
            rel = os.path.relpath(root, projects_dir)
            parts = rel.split(os.sep)
            project_dir = parts[0]

            is_subagent = "subagents" in rel
            parent_session = None
            if is_subagent and len(parts) >= 2:
                # Structure: project_dir/session_id/subagents/agent-xxx.jsonl
                parent_session = parts[1]

            try:
                stat = os.stat(file_path)
            except OSError:
                continue

            sessions.append({
                "session_id": session_id,
                "file_path": file_path,
                "project_dir": project_dir,
                "is_subagent": is_subagent,
                "parent_session": parent_session,
                "file_size": stat.st_size,
                "mtime": stat.st_mtime,
            })

    return sessions


def generate_summary(first_prompt: str | None, first_reply: str | None) -> str | None:
    """Generate a short summary from the first prompt and reply.

    Extracts the core intent by stripping boilerplate prefixes
    and combining with key context from the assistant's reply.
    Returns ~150 chars of dense, keyword-rich text for FTS ranking.
    """
    if not first_prompt:
        return None

    text = first_prompt.strip()

    # Strip common boilerplate prefixes from automated sessions
    _PREFIXES = [
        re.compile(r"^Fix issue #\d+:\s*\[[\w]+\]:\s*", re.IGNORECASE),
        re.compile(r"^Review this code change for issue #\d+:\s*\[[\w]+\]:\s*", re.IGNORECASE),
        re.compile(r"^Analyze this GitHub issue[^.]*\.\s*", re.IGNORECASE),
        re.compile(r"^## Merge target:.*?\n", re.IGNORECASE),
        re.compile(r"^## Code Review.*?\n", re.IGNORECASE),
        re.compile(r"^Given the query and candidates in the input,.*", re.IGNORECASE),
    ]
    for pat in _PREFIXES:
        text = pat.sub("", text).strip()

    if not text:
        text = first_prompt.strip()

    # Take first meaningful sentence/line (up to 120 chars)
    # Split on sentence boundaries or newlines
    for sep in ["\n", ". ", "! ", "? "]:
        if sep in text[:150]:
            text = text[:text.index(sep, 0, 150)]
            break
    text = text[:120].strip()

    # Append context from first_reply if it adds new keywords
    if first_reply:
        reply_text = first_reply.strip()
        # Take first sentence of reply
        for sep in ["\n", ". ", "! ", "? "]:
            if sep in reply_text[:120]:
                reply_text = reply_text[:reply_text.index(sep, 0, 120)]
                break
        reply_text = reply_text[:80].strip()
        if reply_text and reply_text.lower() != text.lower():
            text = f"{text} — {reply_text}"

    return text[:200] if text else None


def format_size(size_bytes: int) -> str:
    """Format bytes as human-readable size."""
    if size_bytes < 1024:
        return f"{size_bytes}B"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.1f}KB"
    else:
        return f"{size_bytes / (1024 * 1024):.1f}MB"


def format_date(iso_date: str | None) -> str:
    """Format an ISO date string for display."""
    if not iso_date:
        return "unknown"
    return iso_date[:10]  # YYYY-MM-DD
