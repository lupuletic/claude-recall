"""Logging for claude-recall."""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path

LOG_DIR = Path.home() / ".claude-recall"
LOG_FILE = LOG_DIR / "debug.log"

_logger: logging.Logger | None = None


def get_logger() -> logging.Logger:
    """Get the claude-recall logger."""
    global _logger
    if _logger is not None:
        return _logger

    _logger = logging.getLogger("claude-recall")
    _logger.setLevel(logging.WARNING)  # quiet by default
    # No handlers until enable_verbose() is called
    return _logger


def enable_verbose(logger: logging.Logger) -> None:
    """Enable verbose logging to both file and stderr."""
    logger.setLevel(logging.DEBUG)
    # Add file handler
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(LOG_FILE, mode="a")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
    )
    logger.addHandler(file_handler)
    # Add stderr handler
    stderr_handler = logging.StreamHandler(sys.stderr)
    stderr_handler.setLevel(logging.DEBUG)
    stderr_handler.setFormatter(
        logging.Formatter("  [%(levelname)s] %(message)s")
    )
    logger.addHandler(stderr_handler)


class Timer:
    """Context manager for timing operations."""

    def __init__(self, label: str, logger: logging.Logger | None = None):
        self.label = label
        self.logger = logger or get_logger()
        self.elapsed_ms = 0.0

    def __enter__(self):
        self.start = time.monotonic()
        return self

    def __exit__(self, *args):
        self.elapsed_ms = (time.monotonic() - self.start) * 1000
        self.logger.debug(f"{self.label}: {self.elapsed_ms:.0f}ms")
