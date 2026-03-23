# claude-recall

Find any Claude Code session instantly. Semantic search across all your past conversations.

```bash
pip install claude-recall[all]
claude-recall "debugging the auth middleware"
```

## The Problem

You use Claude Code heavily and accumulate hundreds of sessions across dozens of projects. Finding that one session where you debugged a tricky auth issue, optimized database queries, or set up CI/CD is painful — built-in `/resume` only shows 10 recent sessions with basic name matching.

**claude-recall** solves this with a 6-stage search pipeline that understands what your sessions were about, not just what keywords they contain.

## Quick Start

```bash
# Install everything
pip install claude-recall[all]

# Search — that's it
claude-recall "polymarket trading bot"
claude-recall "restaurant ordering app"
claude-recall "the iOS app that captures screenshots"
```

First run indexes all sessions (~8s) and generates embeddings in the background. Subsequent searches are instant.

## How the Search Pipeline Works

When you search for "an app that takes and analyses screenshots", here's what happens:

```
┌──────────────────────────────────────────────────────────────────┐
│ Stage 1: FTS5 Keyword Search (BM25)                              │
│                                                                    │
│ Query → stop word removal → "app" AND "takes" AND "analyses"      │
│         AND "screenshots"                                          │
│ Searches: summary (5x weight), first_prompt (3x),                 │
│           last_prompt (2x), messages_text (2x), project_path (4x) │
│                                                                    │
│ If < 3 results: falls back to OR matching with penalty             │
├──────────────────────────────────────────────────────────────────┤
│ Stage 2: Semantic Embedding Search                                │
│                                                                    │
│ Query → bge-small-en-v1.5 embedding (384d, ONNX, local)          │
│ Searches 5,900+ conversation chunks via sqlite-vec cosine sim     │
│                                                                    │
│ Key: subagent chunks map to parent sessions                       │
│ A chunk mentioning "ScreenshotMonitor" in a subagent              │
│ → resolves to the parent hackathon session                        │
├──────────────────────────────────────────────────────────────────┤
│ Stage 3: Reciprocal Rank Fusion                                   │
│                                                                    │
│ Merges keyword + semantic results                                  │
│ score = α/(k+rank_fts) + (1-α)/(k+rank_vec)                      │
│ α adapts: more semantic weight when FTS finds few results          │
├──────────────────────────────────────────────────────────────────┤
│ Stage 4: Cross-Encoder Reranking                                  │
│                                                                    │
│ ms-marco-MiniLM-L-6-v2 (80MB, 18ms for 20 docs)                  │
│ Takes each (query, document) pair with full cross-attention       │
│ Input includes: summary + matched chunk + first_prompt + reply    │
│ Much more accurate than bi-encoder similarity                     │
├──────────────────────────────────────────────────────────────────┤
│ Stage 5: LLM Reranking (optional, "llm" mode)                    │
│                                                                    │
│ Pipes top candidates through claude -p --model haiku              │
│ Claude understands "screenshot app" = "Reshot iOS project"        │
│ ~8-10s latency, but highest accuracy                              │
├──────────────────────────────────────────────────────────────────┤
│ Stage 6: Depth Boost + Relevance Cutoff                           │
│                                                                    │
│ Mild boost for sessions with more messages (log2 scale)           │
│ Drop results below 40% of top score (removes noise)               │
└──────────────────────────────────────────────────────────────────┘
```

### The Key Insight: Subagent Content Bubbling

Claude Code spawns subagents (background workers) for complex tasks. A session about building an iOS app might have 15 subagents doing the actual work — reading files, writing code, running tests. The parent session's first message might just be "let's build this."

**The subagent content is where the real keywords live.** When you search for "Reshot" or "screenshot analysis," those terms only appear in subagent sessions.

We solve this two ways:
1. **At index time**: Parent sessions are enriched with their subagents' first prompts, so FTS keyword search finds them
2. **At search time**: When semantic search matches a subagent chunk, we map it back to the parent session (or the best main session in the same project if the parent isn't indexed)

### Conversation Chunking

Sessions are split into sliding windows of 5 user+assistant turn pairs with 1-turn overlap. Both sides of the conversation are included — assistant responses anchor what was actually discussed.

Each chunk is embedded separately (5,900+ vectors across 1,800+ sessions), and search returns the parent session — the **parent-child retrieval** pattern from RAG literature.

### Why Not Just Use Embeddings?

Embeddings alone miss exact matches. Searching "biwiz" should find the Biwiz project instantly — that's a keyword match, not a semantic one. Our hybrid approach uses:
- **Keywords** for exact terms, project names, error messages, file paths
- **Embeddings** for conceptual queries ("an app that analyses screenshots")
- **Cross-encoder** for precise relevance scoring
- **RRF fusion** to merge both signals without normalizing incompatible score scales

## Install

```bash
# Everything (recommended)
pip install claude-recall[all]

# Or pick what you need
pip install claude-recall                  # keyword search only (zero deps)
pip install claude-recall[semantic]        # + embeddings + reranking
pip install claude-recall[tui]             # + interactive terminal UI

# With uv
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

# Output formats
claude-recall "query" --no-tui      # plain text
claude-recall "query" --json        # JSON for scripting
```

### TUI Keyboard Shortcuts

| Key | Action |
|-----|--------|
| Type | Search as you type (500ms debounce) |
| ↓ / ↑ | Navigate between search bar and results |
| Enter | Focus results / Resume selected session |
| Tab | Toggle preview panel |
| Ctrl+S | Open settings |
| Ctrl+W | Delete last word |
| Esc | Quit |

When you select a session, claude-recall `cd`s to the project directory and runs `claude --resume <id>` — you land right back where you left off.

### Settings

```bash
claude-recall config                              # view settings
claude-recall config search_mode reranked         # best local accuracy
claude-recall config search_mode llm              # best accuracy (uses Claude)
claude-recall config search_mode keyword          # fastest, zero deps
claude-recall config limit 20                     # more results
claude-recall config relevance_cutoff 0.3         # show more borderline results
```

Or press **Ctrl+S** in the TUI for a visual settings panel.

### Index Management

```bash
claude-recall index              # rebuild index
claude-recall index --force      # force full reindex with embeddings
claude-recall info               # show index stats
claude-recall gc                 # clean orphaned entries
claude-recall install-hooks      # install SessionEnd hook for auto-updates
```

## Search Quality

Tested against 20 realistic "vague memory" queries — the kind of thing a developer would type when they can't remember exactly which session they need:

| Query | Expected | Found | Status |
|-------|----------|-------|--------|
| "reshot" | Reshot iOS app | claude-hackathon | ✓ |
| "biwiz website" | Biwiz marketing site | biwiz-marketing-website | ✓ |
| "polymarket bot" | Trading bot | polymarket-copy-trading-bot-v1 | ✓ |
| "fixing bugs on the marketing site" | Biwiz | biwiz-marketing-website | ✓ |
| "market maker profitability" | Polymarket | polymarket-copy-trading-bot-v1 | ✓ |
| "restaurant ordering app for demo" | Restaurant | restaurant-agent-claude | ✓ |
| "grey residence real estate website" | Grey Residence | grey-residence | ✓ |
| "filling in a job application" | CV session | Projects | ✓ |
| "red team testing AI shopping assistant" | Promptfoo | promptfoo | ✓ |
| "iOS app that captures screenshots" | Reshot (semantic) | claude-hackathon | ✓ |
| "email setup for biwizz domain" | Email config | biwizz-ledger | ✓ |
| "ESP32 wifi bluetooth integration" | Telescope | skywatcher | ✓ |
| "saas kit turbo nextjs" | SaaS kit | next-supabase-saas-kit-turbo | ✓ |
| ... | ... | ... | ... |

**Result: 18/20 (90%) accuracy** — including semantic queries like "iOS app that captures screenshots" finding the Reshot project without using its name.

## Architecture

```
┌─────────────────────────────────────────────────────┐
│  Search Pipeline                                     │
│                                                      │
│  1. FTS5 (BM25) with weighted columns               │
│  2. Bi-encoder semantic (bge-small, sqlite-vec)      │
│  3. Reciprocal Rank Fusion                           │
│  4. Cross-encoder reranking (ms-marco-MiniLM, 18ms)  │
│  5. Optional LLM reranking (claude -p)               │
│  6. Depth boost + relevance cutoff                   │
├─────────────────────────────────────────────────────┤
│  Index (SQLite)                                      │
│  • sessions table + FTS5 (5 columns, weighted BM25)  │
│  • chunks table (5-turn sliding windows)             │
│  • chunks_vec (384d cosine, via sqlite-vec)          │
│  • Subagent content enrichment on parent sessions    │
├─────────────────────────────────────────────────────┤
│  Data Source                                         │
│  ~/.claude/projects/*/*.jsonl                        │
│  Incremental updates via file mtime tracking         │
│  Auto-updated via SessionEnd hook                    │
└─────────────────────────────────────────────────────┘
```

## Comparison

| Feature | claude-recall | [recall](https://github.com/arjunkmrm/recall) | [search-sessions](https://github.com/sinzin91/search-sessions) | [claude-history](https://github.com/raine/claude-history) |
|---------|--------------|--------|-----------------|----------------|
| Keyword search | FTS5 + weighted BM25 | FTS5 + BM25 | ripgrep | Fuzzy word |
| Semantic search | Local embeddings | No | No | No |
| Cross-encoder reranking | Yes (18ms) | No | No | No |
| LLM reranking | Optional (claude -p) | No | No | No |
| Subagent content bubbling | Yes | No | No | No |
| Conversation chunking | 5-turn windows | Full text | N/A | N/A |
| Interactive TUI | Yes (Textual) | No | No | Yes (ratatui) |
| Settings UI | Yes (Ctrl+S) | No | No | No |
| Auto-index | Yes + SessionEnd hook | No | N/A | N/A |
| cd to project dir | Yes | No | No | No |
| Search quality benchmark | 90% (18/20) | Not tested | Not tested | Not tested |
| API keys needed | No (LLM mode optional) | No | No | No |

## How It Works Under the Hood

### What Gets Indexed

For each session:
- **Summary**: Auto-generated from first prompt + first reply, or from sessions-index.json
- **First prompt**: The user's first message (cleaned of internal markup)
- **Last prompt**: Where the user left off
- **Messages text**: Smart sample of all user + assistant messages (first 5, every Nth, last 5)
- **Project path**: Full filesystem path (searchable as keywords)
- **Chunks**: 5-turn sliding windows of interleaved user+assistant messages
- **Subagent enrichment**: Parent sessions include subagent first prompts

### Internal Markup Cleaning

Claude Code sessions contain XML markup (`<local-command-caveat>`, `<task-notification>`, `<system-reminder>`, etc.) that we strip for display and indexing. This prevents false matches on internal protocol text.

### Incremental Indexing

File mtimes are tracked in SQLite. Only new or modified sessions are re-indexed. Typical incremental update: < 0.5s for 1,800+ sessions.

### Background Embedding Generation

On first run, FTS indexing completes in ~8s and results are available immediately. Embedding generation (5,900+ chunks) happens in a background process (~2-3 minutes) so you're never blocked.

## Requirements

- Python 3.10+
- Claude Code (sessions stored at `~/.claude/projects/`)

## Development

```bash
git clone https://github.com/lupuletic/claude-recall
cd claude-recall
uv venv && source .venv/bin/activate
uv pip install -e ".[all,dev]"
python -m pytest tests/ -q     # 243 tests
```

## License

MIT
