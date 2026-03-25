"""CLI interface for claude-recall."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from claude_recall import __version__
from claude_recall.db import DB_PATH, get_connection, get_stats
from claude_recall.indexer import build_index, ensure_index
from claude_recall.searcher import search
from claude_recall.utils import PROJECTS_DIR, clean_display_text, format_date, format_size

HOOKS_MARKER = Path.home() / ".claude-recall" / ".hooks-installed"


def main(argv: list[str] | None = None) -> None:
    try:
        _run(argv)
    except KeyboardInterrupt:
        print("", file=sys.stderr)  # clean newline
        sys.exit(0)


def _run(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="claude-recall",
        description="Semantic search across Claude Code sessions.",
        usage="%(prog)s [query ...] [options]\n       %(prog)s <command> [options]",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    parser.add_argument("--db", type=Path, default=DB_PATH, help=argparse.SUPPRESS)
    parser.add_argument(
        "--claude-dir", type=Path, default=PROJECTS_DIR, help=argparse.SUPPRESS
    )

    # Search options (work both with and without 'search' subcommand)
    parser.add_argument("query", nargs="*", help="Search query (or subcommand)")
    parser.add_argument("-n", "--limit", type=int, default=10, help="Max results")
    parser.add_argument("-p", "--project", help="Filter by project path substring")
    parser.add_argument("--after", help="Only sessions after this date (YYYY-MM-DD)")
    parser.add_argument("--before", help="Only sessions before this date (YYYY-MM-DD)")
    parser.add_argument(
        "--semantic", action="store_true", default=None, help="Force semantic search"
    )
    parser.add_argument("--no-semantic", action="store_true", help="Disable semantic search")
    parser.add_argument("--no-tui", action="store_true", help="Plain text output (no TUI)")
    parser.add_argument("--json", action="store_true", dest="json_output", help="JSON output")
    parser.add_argument("--min-messages", type=int, default=1, help=argparse.SUPPRESS)
    parser.add_argument("-v", "--verbose", action="store_true", help="Show debug logs")
    parser.add_argument("--force", action="store_true", help="Force full reindex (with 'index')")
    parser.add_argument("--quiet", action="store_true", help="Suppress progress output")

    args = parser.parse_args(argv)

    # Setup logging
    from claude_recall.logger import enable_verbose, get_logger

    log = get_logger()
    if getattr(args, "verbose", False):
        enable_verbose(log)
    log.debug(f"claude-recall started, args={vars(args)}")

    # Route to subcommands if the first positional arg is a known command
    command = args.query[0] if args.query else None

    if command in ("index", "i"):
        _cmd_index(args)
    elif command == "info":
        _cmd_info(args)
    elif command == "gc":
        _cmd_gc(args)
    elif command == "config":
        _cmd_config(args)
    elif command in ("install-hooks", "setup"):
        _cmd_install_hooks()
    elif command == "search" or command == "s":
        # Explicit 'search' subcommand — strip it from query
        args.query = args.query[1:]
        _cmd_search(args)
    else:
        # Default: treat all positional args as search query
        _cmd_search(args)

    # Non-blocking update check (once per day)
    from claude_recall.updater import check_for_update

    check_for_update(quiet=args.quiet or args.json_output)


def _first_run_setup(args: argparse.Namespace) -> None:
    """Run on first use: index + install hooks."""
    is_first_run = not args.db.exists()
    show_output = not args.quiet and not args.json_output

    if is_first_run and show_output:
        print("Welcome to claude-recall! Setting up...\n", file=sys.stderr)

    # Auto-index (silent on subsequent runs — takes <1s)
    ensure_index(args.claude_dir, args.db, verbose=is_first_run and show_output)

    # Auto-install hooks on first run
    if is_first_run and not HOOKS_MARKER.exists():
        _auto_install_hooks()


def _auto_install_hooks() -> None:
    """Silently install SessionEnd hooks on first run."""
    from claude_recall.config import load_config

    if not load_config().get("auto_index_hook", True):
        return

    import shutil

    settings_path = Path.home() / ".claude" / "settings.json"
    claude_recall_bin = shutil.which("claude-recall")
    if not claude_recall_bin:
        return

    hook_command = f"{claude_recall_bin} index --quiet"

    settings = {}
    if settings_path.exists():
        try:
            with open(settings_path) as f:
                settings = json.load(f)
        except (json.JSONDecodeError, OSError):
            pass

    hooks = settings.get("hooks", {})
    session_end_hooks = hooks.get("SessionEnd", [])

    # Don't install if already present
    for rule in session_end_hooks:
        for hook in rule.get("hooks", []):
            if "claude-recall" in hook.get("command", ""):
                HOOKS_MARKER.parent.mkdir(parents=True, exist_ok=True)
                HOOKS_MARKER.touch()
                return

    new_hook = {
        "hooks": [
            {"type": "command", "command": hook_command, "timeout": 30000}
        ]
    }
    session_end_hooks.append(new_hook)
    hooks["SessionEnd"] = session_end_hooks
    settings["hooks"] = hooks

    try:
        with open(settings_path, "w") as f:
            json.dump(settings, f, indent=2)
        HOOKS_MARKER.parent.mkdir(parents=True, exist_ok=True)
        HOOKS_MARKER.touch()
        print(
            "  Auto-installed SessionEnd hook for live index updates.\n",
            file=sys.stderr,
        )
    except OSError:
        pass


def _cmd_config(args: argparse.Namespace) -> None:
    """View or set config values."""
    from claude_recall.config import print_config, set_value

    config_args = args.query[1:]  # strip "config"
    if len(config_args) >= 2:
        key, value = config_args[0], " ".join(config_args[1:])
        err = set_value(key, value)
        if err:
            print(f"Error: {err}", file=sys.stderr)
            sys.exit(1)
        print(f"Set {key} = {value}")
    else:
        print_config()


def _cmd_search(args: argparse.Namespace) -> None:
    from claude_recall.config import load_config

    config = load_config()
    query = " ".join(args.query)

    # Use config limit if user didn't explicitly set -n (default is 10)
    if args.limit == 10:
        args.limit = config.get("limit", 10)

    # Determine search mode from config + CLI flags
    semantic = None  # None = auto-detect (use if available)
    if args.semantic:
        semantic = True
    elif args.no_semantic:
        semantic = False
    elif config["search_mode"] == "keyword":
        semantic = False

    # First-run setup (auto-index + hooks)
    _first_run_setup(args)

    # No query + interactive terminal → open TUI for browsing
    if not query and sys.stdout.isatty():
        try:
            from claude_recall.tui import run_tui

            run_tui("", [])
            return
        except ImportError:
            pass

    if not query:
        print("Usage: claude-recall <query>", file=sys.stderr)
        print('  Example: claude-recall "debugging auth middleware"', file=sys.stderr)
        sys.exit(1)

    results = search(
        query=query,
        db_path=args.db,
        limit=args.limit,
        project_filter=args.project,
        after=args.after,
        before=args.before,
        semantic=semantic,
        min_messages=args.min_messages,
    )

    if args.json_output:
        _print_json(results)
        return

    # TUI if available and not disabled
    if not args.no_tui and sys.stdout.isatty():
        try:
            from claude_recall.tui import run_tui

            run_tui(query, results)
            return
        except ImportError:
            pass

    _print_plain(query, results)


def _print_plain(query: str, results: list) -> None:
    """Print results as formatted plain text."""
    if not results:
        print(f'No sessions found for "{query}"')
        return

    print(f'\nFound {len(results)} sessions for "{query}"\n')

    for i, r in enumerate(results, 1):
        s = r.session
        score_str = f"score: {r.score:.2f}"

        title = clean_display_text(s.summary) or clean_display_text(s.first_prompt) or "Untitled"
        if len(title) > 60:
            title = title[:60] + "..."
        print(f" {i:>2}. {title:<52} {score_str}")

        meta_parts = [r.display_project]
        if s.modified:
            meta_parts.append(format_date(s.modified))
        if s.git_branch:
            meta_parts.append(s.git_branch)
        meta_parts.append(f"{s.message_count} msgs")
        if s.file_size:
            meta_parts.append(format_size(s.file_size))
        print(f"     {' · '.join(meta_parts)}")

        # Show last activity if different from title
        if s.last_prompt and s.last_prompt != s.first_prompt:
            last = clean_display_text(s.last_prompt)
            if last:
                print(f"     Last: {last[:120]}")
        elif r.snippets:
            snippet = clean_display_text(r.snippets[0])
            if snippet:
                print(f"     > {snippet[:120]}")

        print(f"     Resume: cd {r.display_project} && claude --resume {s.session_id}")
        print()


def _print_json(results: list) -> None:
    """Print results as JSON."""
    output = []
    for r in results:
        s = r.session
        output.append({
            "session_id": s.session_id,
            "project_path": s.project_path,
            "summary": s.summary,
            "first_prompt": s.first_prompt,
            "last_prompt": s.last_prompt,
            "git_branch": s.git_branch,
            "message_count": s.message_count,
            "file_size": s.file_size,
            "modified": s.modified,
            "score": round(r.score, 4),
            "fts_rank": r.fts_rank,
            "vec_score": round(r.vec_score, 4) if r.vec_score is not None else None,
            "snippets": r.snippets,
            "resume_command": r.resume_command,
        })
    json.dump(output, sys.stdout, indent=2)
    print()


def _cmd_index(args: argparse.Namespace) -> None:
    if not args.quiet:
        print("Building search index...", file=sys.stderr)

    # Pre-download models during explicit index (not during search)
    try:
        from claude_recall.embedder import ensure_models_downloaded
        ensure_models_downloaded()
    except Exception:
        pass

    stats = build_index(
        projects_dir=args.claude_dir,
        db_path=args.db,
        force=args.force,
        verbose=not args.quiet,
    )
    if not args.quiet:
        print(
            f"\nIndex complete: {stats['indexed']} indexed, "
            f"{stats['skipped']} unchanged, "
            f"{stats['removed']} removed",
            file=sys.stderr,
        )


def _cmd_info(args: argparse.Namespace) -> None:
    if not args.db.exists():
        print("No index found. Run 'claude-recall' to build it.")
        return

    conn = get_connection(args.db)
    stats = get_stats(conn)

    chunk_count = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]

    # Graph stats
    try:
        edges = conn.execute("SELECT COUNT(*) FROM graph_edges").fetchone()[0]
        files = conn.execute("SELECT COUNT(DISTINCT file_name) FROM session_files").fetchone()[0]
        chains = conn.execute("SELECT COUNT(DISTINCT chain_id) FROM session_chains").fetchone()[0]
    except Exception:
        edges = files = chains = 0

    conn.close()

    print(f"claude-recall v{__version__}")
    print(f"  Database: {args.db}")
    print(f"  Size: {format_size(args.db.stat().st_size)}")
    print(
        f"  Sessions: {stats.get('total', 0)} "
        f"({stats.get('main_sessions', 0)} main, "
        f"{stats.get('subagent_sessions', 0)} subagent)"
    )
    print(f"  Chunks: {chunk_count}")
    print(f"  Projects: {stats.get('projects', 0)}")
    print(f"  Messages: {stats.get('total_messages', 0)}")
    print(f"  Graph: {edges} edges, {files} unique files, {chains} session chains")
    print(f"  Source size: {format_size(stats.get('total_size', 0) or 0)}")
    print(
        f"  Date range: {format_date(stats.get('earliest'))} to "
        f"{format_date(stats.get('latest'))}"
    )

    from claude_recall import has_semantic, has_tui

    sem = "enabled" if has_semantic() else "not installed (pip install claude-recall[semantic])"
    tui = "enabled" if has_tui() else "not installed (pip install claude-recall[tui])"
    print(f"  Semantic: {sem}")
    print(f"  TUI: {tui}")


def _cmd_gc(args: argparse.Namespace) -> None:
    """Remove orphaned index entries for deleted session files."""
    if not args.db.exists():
        print("No index found.")
        return

    conn = get_connection(args.db)
    rows = conn.execute("SELECT session_id, file_path FROM sessions").fetchall()

    removed = 0
    for row in rows:
        if not Path(row["file_path"]).exists():
            conn.execute(
                "DELETE FROM sessions WHERE session_id = ?", (row["session_id"],)
            )
            removed += 1

    conn.commit()
    conn.close()
    print(f"Removed {removed} orphaned entries")


def _cmd_install_hooks() -> None:
    """Install Claude Code SessionEnd hook for automatic index updates."""
    import shutil

    settings_path = Path.home() / ".claude" / "settings.json"
    claude_recall_bin = shutil.which("claude-recall")

    if not claude_recall_bin:
        print("Warning: 'claude-recall' not found in PATH.")
        claude_recall_bin = "claude-recall"

    hook_command = f"{claude_recall_bin} index --quiet"

    settings = {}
    if settings_path.exists():
        try:
            with open(settings_path) as f:
                settings = json.load(f)
        except (json.JSONDecodeError, OSError):
            pass

    hooks = settings.get("hooks", {})
    session_end_hooks = hooks.get("SessionEnd", [])

    for rule in session_end_hooks:
        for hook in rule.get("hooks", []):
            if "claude-recall" in hook.get("command", ""):
                print("Hook already installed!")
                print(f"  Command: {hook['command']}")
                return

    new_hook = {
        "hooks": [
            {"type": "command", "command": hook_command, "timeout": 30000}
        ]
    }
    session_end_hooks.append(new_hook)
    hooks["SessionEnd"] = session_end_hooks
    settings["hooks"] = hooks

    with open(settings_path, "w") as f:
        json.dump(settings, f, indent=2)

    HOOKS_MARKER.parent.mkdir(parents=True, exist_ok=True)
    HOOKS_MARKER.touch()

    print("Installed Claude Code SessionEnd hook!")
    print(f"  Settings: {settings_path}")
    print(f"  Command: {hook_command}")
    print()
    print("Index will auto-update when you exit Claude Code sessions.")


if __name__ == "__main__":
    main()
