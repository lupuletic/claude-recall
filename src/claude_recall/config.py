"""Configuration for claude-recall."""

from __future__ import annotations

import json
from pathlib import Path

CONFIG_PATH = Path.home() / ".claude-recall" / "config.json"

DEFAULTS = {
    "search_mode": "hybrid",  # keyword, semantic, hybrid, reranked
    "limit": 10,
    "show_subagents": False,
    "relevance_cutoff": 0.4,  # drop results below this % of top score
    "auto_index_hook": True,
}

SEARCH_MODES = {
    "keyword": "FTS5 keyword search only (fastest, no dependencies)",
    "semantic": "Embedding-based semantic search only",
    "hybrid": "Keyword + semantic with RRF fusion",
    "reranked": "Hybrid + cross-encoder reranking (most accurate)",
}


def load_config() -> dict:
    """Load config from disk, merged with defaults."""
    config = dict(DEFAULTS)
    if CONFIG_PATH.exists():
        try:
            with open(CONFIG_PATH) as f:
                saved = json.load(f)
                config.update(saved)
        except (json.JSONDecodeError, OSError):
            pass
    return config


def save_config(config: dict) -> None:
    """Save config to disk."""
    CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(CONFIG_PATH, "w") as f:
        json.dump(config, f, indent=2)


def set_value(key: str, value: str) -> str | None:
    """Set a config value. Returns error message or None on success."""
    config = load_config()

    if key not in DEFAULTS:
        return f"Unknown setting: {key}. Available: {', '.join(DEFAULTS)}"

    # Type conversion
    default_type = type(DEFAULTS[key])
    try:
        if default_type is bool:
            if value.lower() in ("true", "1", "yes", "on"):
                config[key] = True
            elif value.lower() in ("false", "0", "no", "off"):
                config[key] = False
            else:
                return f"Invalid boolean: {value}. Use true/false."
        elif default_type is int:
            config[key] = int(value)
        elif default_type is float:
            config[key] = float(value)
        elif key == "search_mode":
            if value not in SEARCH_MODES:
                return f"Invalid mode: {value}. Options: {', '.join(SEARCH_MODES)}"
            config[key] = value
        else:
            config[key] = value
    except ValueError:
        return f"Invalid value for {key}: {value}"

    save_config(config)
    return None


def print_config() -> None:
    """Print current config with descriptions."""
    config = load_config()

    from claude_recall import has_semantic

    print("claude-recall settings")
    print(f"  Config: {CONFIG_PATH}\n")

    for key, value in config.items():
        default = DEFAULTS.get(key)
        marker = "" if value == default else " (modified)"
        if key == "search_mode":
            desc = SEARCH_MODES.get(value, "")
            print(f"  {key}: {value}{marker}")
            print(f"    {desc}")
            if value in ("semantic", "hybrid", "reranked") and not has_semantic():
                print(f"    ⚠ Requires: pip install claude-recall[semantic]")
        else:
            print(f"  {key}: {value}{marker}")

    print(f"\nSet with: claude-recall config <key> <value>")
    print(f"Modes: {', '.join(SEARCH_MODES)}")
