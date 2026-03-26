"""Search quality benchmark.

Run with: python -m pytest tests/test_search_quality.py -v --tb=short

These tests require a populated index from real Claude Code sessions.
They are skipped when no index exists (CI environments).
"""

from __future__ import annotations

import pytest

from claude_recall.db import DB_PATH


pytestmark = pytest.mark.skipif(
    not DB_PATH.exists(),
    reason="No index found — run 'claude-recall index' first",
)


def _search_finds(query: str, expected_project: str, limit: int = 3) -> bool:
    """Check if the top search result matches the expected project."""
    from claude_recall.searcher import search

    results = search(query, limit=limit)
    if not results:
        return False
    return expected_project.lower() in results[0].session.project_path.lower()


def _search_finds_in_top(query: str, expected_project: str, top_n: int = 3) -> bool:
    """Check if the expected project appears anywhere in the top N results."""
    from claude_recall.searcher import search

    results = search(query, limit=top_n)
    return any(
        expected_project.lower() in r.session.project_path.lower()
        for r in results
    )


class TestExactProjectNames:
    """Searching by project name should always find the right session."""

    def test_reshot(self):
        assert _search_finds("reshot", "claude-hackathon")

    def test_skywatcher(self):
        assert _search_finds("skywatcher", "skywatcher")

    def test_mirofish(self):
        assert _search_finds("mirofish", "MiroFish")

    def test_grey_residence(self):
        assert _search_finds("grey residence", "grey-residence")

    def test_clawdbot(self):
        assert _search_finds("clawdbot", "clawdbot")

    def test_saas_kit(self):
        assert _search_finds("saas kit turbo", "next-supabase")

    def test_sefu_la_bani(self):
        assert _search_finds("sefu la bani", "polymarket")


class TestDescribingWork:
    """Searching by describing what was done should find the right session."""

    def test_git_accounts(self):
        assert _search_finds("setting up git with multiple accounts", "Projects")

    def test_market_maker(self):
        assert _search_finds("market maker bot profitability", "polymarket")

    def test_restaurant(self):
        assert _search_finds("restaurant ordering app for demo", "restaurant")

    def test_red_team(self):
        assert _search_finds("red team testing a shopping assistant", "promptfoo")

    def test_bug_fixes(self):
        assert _search_finds("fixing bugs on the marketing site", "biwiz")

    def test_job_application(self):
        assert _search_finds("filling in a job application form", "Projects")

    def test_email_setup(self):
        assert _search_finds("email setup with microsoft 365", "biwizz-ledger")

    def test_esp32(self):
        assert _search_finds("ESP32 wifi and bluetooth integration", "skywatcher")

    def test_local_llm(self):
        assert _search_finds("running a local LLM Qwen model", "Projects")

    @pytest.mark.skip(reason="grey-residence parent JSONL not on disk — only subagent dir exists")
    def test_private_repo(self):
        assert _search_finds("creating a private github repo for a property site", "grey")


class TestSemanticQueries:
    """Conceptual queries that don't use exact terms from the session."""

    def test_screenshot_app(self):
        # Both second-look-ios and claude-hackathon/Reshot are iOS screenshot apps
        assert (
            _search_finds("iOS app that captures screenshots", "claude-hackathon")
            or _search_finds("iOS app that captures screenshots", "second-look")
            or _search_finds("iOS app that captures screenshots", "Reshot")
        )

    def test_trading_bot(self):
        assert _search_finds("session where we worked on the trading bot", "polymarket")

    def test_telescope(self):
        assert _search_finds("that project with the telescope and electronics", "skywatcher")

    def test_food_ordering(self):
        assert _search_finds("when we built the food ordering demo", "restaurant")

    def test_awards_application(self):
        assert _search_finds("help filling in competition awards", "Projects")

    def test_email_saas(self):
        assert _search_finds("setting up email sender for a SaaS", "biwizz-ledger")


class TestTechnologyQueries:
    """Searching by technology stack."""

    def test_chrome_extension(self):
        assert _search_finds("chrome browser extension development", "clawdbot")

    def test_docker(self):
        assert _search_finds("docker container gateway issues", "clawdbot")

    def test_telegram(self):
        assert _search_finds("telegram bot webhook setup", "clawdbot")

    def test_polymarket_api(self):
        assert _search_finds("polymarket API order placement", "polymarket")

    def test_whatsapp(self):
        assert _search_finds("whatsapp channel integration", "clawdbot")

    def test_promptfoo(self):
        assert _search_finds("promptfoo red team security testing", "promptfoo")


class TestLastMessageQueries:
    """Finding sessions by what was last discussed."""

    def test_push_latest(self):
        assert _search_finds("push the latest to remote", "Projects")

    def test_godaddy(self):
        assert _search_finds("use godaddy now", "biwiz")

    @pytest.mark.skip(reason="grey-residence parent JSONL not on disk — prompt was in unsaved parent session")
    def test_final_version(self):
        assert _search_finds("give me the final version", "grey")
