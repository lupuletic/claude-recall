"""Auto-update check for claude-recall."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

from claude_recall import __version__

UPDATE_CHECK_FILE = Path.home() / ".claude-recall" / ".last-update-check"
CHECK_INTERVAL = 86400  # 24 hours


def check_for_update(quiet: bool = False) -> None:
    """Check PyPI for a newer version. Runs at most once per day."""
    if quiet:
        return

    from claude_recall.config import load_config

    if not load_config().get("update_check", True):
        return

    # Skip if checked recently
    if UPDATE_CHECK_FILE.exists():
        try:
            last_check = float(UPDATE_CHECK_FILE.read_text().strip())
            if time.time() - last_check < CHECK_INTERVAL:
                return
        except (ValueError, OSError):
            pass

    # Record this check
    try:
        UPDATE_CHECK_FILE.parent.mkdir(parents=True, exist_ok=True)
        UPDATE_CHECK_FILE.write_text(str(time.time()))
    except OSError:
        return

    # Check PyPI in a non-blocking way
    try:
        from urllib.request import urlopen

        resp = urlopen(
            "https://pypi.org/pypi/claude-recall/json",
            timeout=2,
        )
        data = json.loads(resp.read())
        latest = data["info"]["version"]

        if latest != __version__ and _is_newer(latest, __version__):
            print(
                f"\n  Update available: {__version__} → {latest}"
                f"\n  Run: pip install --upgrade claude-recall\n",
                file=sys.stderr,
            )
    except Exception:
        pass  # Network error, not on PyPI yet, etc.


def _is_newer(latest: str, current: str) -> bool:
    """Compare version strings (simple tuple comparison)."""
    try:
        lat = tuple(int(x) for x in latest.split("."))
        cur = tuple(int(x) for x in current.split("."))
        return lat > cur
    except ValueError:
        return False
