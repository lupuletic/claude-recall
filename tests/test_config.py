"""Tests for claude_recall.config."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from claude_recall.config import DEFAULTS, SEARCH_MODES, load_config, save_config, set_value


@pytest.fixture(autouse=True)
def config_path(tmp_path, monkeypatch):
    """Redirect CONFIG_PATH to a temp directory for all tests."""
    cfg_path = tmp_path / "config.json"
    monkeypatch.setattr("claude_recall.config.CONFIG_PATH", cfg_path)
    return cfg_path


# ===========================================================================
# load_config
# ===========================================================================

class TestLoadConfig:
    def test_returns_defaults_when_no_file(self):
        config = load_config()
        assert config == DEFAULTS

    def test_returns_defaults_keys(self):
        config = load_config()
        assert "search_mode" in config
        assert "limit" in config
        assert "show_subagents" in config
        assert "relevance_cutoff" in config
        assert "auto_index_hook" in config

    def test_default_values(self):
        config = load_config()
        assert config["search_mode"] == "hybrid"
        assert config["limit"] == 10
        assert config["show_subagents"] is False
        assert config["relevance_cutoff"] == 0.4
        assert config["auto_index_hook"] is True


# ===========================================================================
# save_config / load_config roundtrip
# ===========================================================================

class TestSaveLoadRoundtrip:
    def test_roundtrip(self, config_path):
        original = {"search_mode": "keyword", "limit": 20, "show_subagents": True,
                     "relevance_cutoff": 0.5, "auto_index_hook": False}
        save_config(original)
        loaded = load_config()
        assert loaded["search_mode"] == "keyword"
        assert loaded["limit"] == 20
        assert loaded["show_subagents"] is True
        assert loaded["relevance_cutoff"] == 0.5
        assert loaded["auto_index_hook"] is False

    def test_saved_file_is_valid_json(self, config_path):
        save_config({"search_mode": "hybrid"})
        data = json.loads(config_path.read_text())
        assert data["search_mode"] == "hybrid"

    def test_creates_parent_dirs(self, tmp_path, monkeypatch):
        nested = tmp_path / "a" / "b" / "config.json"
        monkeypatch.setattr("claude_recall.config.CONFIG_PATH", nested)
        save_config({"search_mode": "hybrid"})
        assert nested.exists()

    def test_merge_with_defaults(self, config_path):
        """Saved config merges with defaults, so new keys get defaults."""
        save_config({"search_mode": "keyword"})
        config = load_config()
        # Should have the saved value
        assert config["search_mode"] == "keyword"
        # Should have default values for keys not in saved file
        assert config["limit"] == DEFAULTS["limit"]

    def test_invalid_json_returns_defaults(self, config_path):
        config_path.write_text("not json!!!")
        config = load_config()
        assert config == DEFAULTS


# ===========================================================================
# set_value
# ===========================================================================

class TestSetValue:
    def test_set_search_mode_keyword(self):
        err = set_value("search_mode", "keyword")
        assert err is None
        assert load_config()["search_mode"] == "keyword"

    def test_set_search_mode_auto(self):
        err = set_value("search_mode", "hybrid")
        assert err is None
        assert load_config()["search_mode"] == "hybrid"

    def test_set_search_mode_auto_2(self):
        err = set_value("search_mode", "hybrid")
        assert err is None
        assert load_config()["search_mode"] == "hybrid"

    def test_set_search_mode_reranked(self):
        err = set_value("search_mode", "hybrid")
        assert err is None
        assert load_config()["search_mode"] == "hybrid"

    def test_set_invalid_search_mode(self):
        err = set_value("search_mode", "invalid_mode")
        assert err is not None
        assert "Invalid mode" in err

    def test_set_bool_true_variants(self):
        for val in ("true", "1", "yes", "on", "True", "YES"):
            err = set_value("show_subagents", val)
            assert err is None
            assert load_config()["show_subagents"] is True

    def test_set_bool_false_variants(self):
        for val in ("false", "0", "no", "off", "False", "NO"):
            err = set_value("show_subagents", val)
            assert err is None
            assert load_config()["show_subagents"] is False

    def test_set_invalid_bool(self):
        err = set_value("show_subagents", "maybe")
        assert err is not None
        assert "Invalid boolean" in err

    def test_set_int(self):
        err = set_value("limit", "25")
        assert err is None
        assert load_config()["limit"] == 25

    def test_set_invalid_int(self):
        err = set_value("limit", "not_a_number")
        assert err is not None

    def test_set_float(self):
        err = set_value("relevance_cutoff", "0.75")
        assert err is None
        assert load_config()["relevance_cutoff"] == pytest.approx(0.75)

    def test_set_invalid_float(self):
        err = set_value("relevance_cutoff", "abc")
        assert err is not None

    def test_set_unknown_key(self):
        err = set_value("nonexistent_key", "value")
        assert err is not None
        assert "Unknown setting" in err

    def test_set_persists(self, config_path):
        set_value("limit", "50")
        # Reload from disk
        config = json.loads(config_path.read_text())
        assert config["limit"] == 50
