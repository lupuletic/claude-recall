# claude-recall

Find any Claude Code session instantly. Semantic search across all your past conversations.

```bash
pip install claude-recall[all]
claude-recall "debugging the auth middleware"
```

## The Problem

You use Claude Code heavily and accumulate hundreds of sessions across dozens of projects. Finding that one session where you debugged a tricky auth issue, optimized database queries, or set up CI/CD is painful — built-in `/resume` only shows 10 recent sessions with basic name matching.

## How It Works

claude-recall builds a local search index of all your Claude Code sessions and provides a **4-stage search pipeline**:

1. **FTS5 keyword search** (BM25) — fast, zero dependencies
2. **Semantic embedding search** — local ONNX model, no API keys
3. **Reciprocal Rank Fusion** — merges keyword + semantic signals
4. **Cross-encoder reranking** — 18ms, full query-document cross-attention for precise relevance

```
"restaurant ordering app"
  → Stage 1: FTS5 finds sessions with "restaurant", "ordering", "app"
  → Stage 2: Embeddings find semantically similar sessions
  → Stage 3: RRF merges both result sets
  → Stage 4: Cross-encoder reranks top candidates by true relevance
  → Result: The exact session you were looking for
```

## Install

```bash
# Everything (recommended)
pip install claude-recall[all]

# Or pick what you need
pip install claude-recall                  # keyword search only (zero deps)
pip install claude-recall[semantic]        # + embeddings + reranking
pip install claude-recall[tui]             # + interactive terminal UI
```

**With uv (recommended):**
```bash
uv tool install claude-recall --with textual --with fastembed --with sqlite-vec
```

## Usage

```bash
# Interactive TUI — type to search, arrows to navigate, Enter to resume
claude-recall

# Direct search
claude-recall "debugging auth middleware"
claude-recall "polymarket trading bot"
claude-recall "setting up 2 git accounts"

# Filter by project
claude-recall "optimization" --project storefront

# Filter by date
claude-recall "database migration" --after 2026-01-01

# Plain text output (no TUI)
claude-recall "query" --no-tui

# JSON output (for scripting)
claude-recall "query" --json

# Force search mode
claude-recall "query" --semantic       # semantic only
claude-recall "query" --no-semantic    # keyword only
```

### TUI Keyboard Shortcuts

| Key | Action |
|-----|--------|
| Type | Search as you type (500ms debounce) |
| ↓ | Move from search bar to results |
| ↑ | Move from results back to search bar |
| Enter | Focus results list / Resume selected session |
| Tab | Toggle preview panel |
| Ctrl+W | Delete last word in search |
| Esc | Quit |

### Settings

```bash
# View all settings
claude-recall config

# Set search mode
claude-recall config search_mode reranked    # most accurate (default: hybrid)
claude-recall config search_mode keyword     # fastest, no dependencies
claude-recall config search_mode semantic    # embedding-based only
claude-recall config search_mode hybrid      # keyword + semantic

# Other settings
claude-recall config limit 20               # max results (default: 10)
claude-recall config relevance_cutoff 0.3   # show more borderline results
claude-recall config show_subagents true    # include subagent sessions
```

### Index Management

```bash
claude-recall index              # rebuild index
claude-recall index --force      # force full reindex
claude-recall info               # show index stats
claude-recall gc                 # clean orphaned entries
claude-recall install-hooks      # install SessionEnd hook for auto-updates
```

## First Run

On first run, claude-recall:
1. Builds a keyword index of all sessions (~8 seconds)
2. Shows search results immediately
3. Generates semantic embeddings in the background (~90 seconds)
4. Auto-installs a Claude Code SessionEnd hook for live index updates

Subsequent searches are instant (< 100ms).

## Architecture

```
┌─────────────────────────────────────────────────────┐
│  Search Pipeline                                     │
│                                                      │
│  Query → FTS5 (BM25)  ──┐                           │
│                          ├── RRF Fusion → Reranker   │
│  Query → Embeddings   ──┘    (merge)     (18ms)      │
│          (sqlite-vec)                                │
├─────────────────────────────────────────────────────┤
│  Index (SQLite)                                      │
│  • sessions table + FTS5 virtual table               │
│  • chunks table (5-turn sliding windows)             │
│  • chunks_vec (384d embeddings via sqlite-vec)       │
├─────────────────────────────────────────────────────┤
│  Data Source                                         │
│  ~/.claude/projects/*/*.jsonl                        │
│  Incremental updates via file mtime tracking         │
│  Auto-updated via SessionEnd hook                    │
└─────────────────────────────────────────────────────┘
```

### Conversation Chunking

Sessions are split into sliding windows of 5 user+assistant turn pairs with 1-turn overlap. Both user and assistant messages are included — assistant responses anchor what was actually discussed. Each chunk is embedded separately, and search returns the parent session (parent-child retrieval pattern).

### Cross-Encoder Reranking

The cross-encoder (`Xenova/ms-marco-MiniLM-L-6-v2`, 80MB) takes each (query, document) pair through a transformer with full cross-attention. This is far more accurate than bi-encoder cosine similarity because the query and document tokens attend to each other directly. Applied to the top ~20 candidates from the hybrid search, it takes ~18ms on CPU.

## Comparison

| Feature | claude-recall | [recall](https://github.com/arjunkmrm/recall) | [search-sessions](https://github.com/sinzin91/search-sessions) | [claude-history](https://github.com/raine/claude-history) |
|---------|--------------|--------|-----------------|----------------|
| Keyword search | FTS5 + BM25 | FTS5 + BM25 | ripgrep | Fuzzy word |
| Semantic search | FastEmbed (local) | No | No | No |
| Cross-encoder reranking | Yes (18ms) | No | No | No |
| Hybrid ranking | RRF fusion | No | No | No |
| Conversation chunking | 5-turn windows | Full text | N/A | N/A |
| Interactive TUI | Yes | No | No | Yes |
| Auto-index | Yes + SessionEnd hook | No | N/A | N/A |
| cd to project dir | Yes | No | No | No |
| Settings/config | Yes | No | No | No |
| API keys needed | No | No | No | No |

## How Sessions Are Indexed

- **FTS5 text**: Smart sampling — first 5 messages, every Nth from middle, last 5 messages. Covers keywords from the entire conversation arc.
- **Embeddings**: 5-turn sliding windows with user+assistant messages. Captures semantic meaning throughout the conversation.
- **Metadata**: first prompt, last prompt, summary (from sessions-index.json), git branch, project path, message count, dates.
- **Incremental**: File mtime tracking. Only re-indexes changed/new sessions.

## Requirements

- Python 3.10+
- Claude Code (sessions stored at `~/.claude/projects/`)

## License

MIT
