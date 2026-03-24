"""Allow running as `python -m claude_recall`."""

import sys

from claude_recall.cli import main

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(0)
