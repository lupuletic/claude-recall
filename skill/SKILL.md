---
name: find-session
description: Search past Claude Code sessions by topic, intent, or keywords using semantic search. Use when users say "find session", "search sessions", "which session", "find where we", "recall when we", or reference past Claude Code work they want to locate.
allowed-tools: Bash(claude-recall *)
---

# Find Session — Semantic Session Search

Search across all Claude Code sessions using natural language. Unlike keyword search, this understands intent — "sessions where we optimized the API layer" will find sessions about performance tuning, caching, query optimization, etc.

## Usage

```
/find-session <natural language query>
```

## Step 1: Run the Search

```bash
claude-recall search "<user's query>" --no-tui --limit 10
```

If the user wants JSON output for further processing:
```bash
claude-recall search "<user's query>" --json --limit 10
```

Optional filters:
- `--project <name>`: Filter by project path substring
- `--after YYYY-MM-DD`: Only sessions after this date
- `--before YYYY-MM-DD`: Only sessions before this date
- `--semantic`: Force semantic search (requires `claude-recall[semantic]`)

## Step 2: Present Results

Format the output for the user. For each result, show:
- Session title/summary
- Project path, date, git branch, message count
- Relevant snippet
- Resume command: `claude --resume <session_id>`

If no results found, suggest:
- Broadening the search terms
- Checking if the index is up to date: `claude-recall index`
- Using `--semantic` if not already (requires `pip install claude-recall[semantic]`)

## Step 3: Resume (if requested)

If the user wants to resume a session, tell them to run:
```
claude --resume <session_id>
```

## Installation

If `claude-recall` is not installed, tell the user:
```bash
pip install claude-recall              # Core (keyword search)
pip install claude-recall[semantic]    # + semantic search
pip install claude-recall[all]         # + TUI
```
