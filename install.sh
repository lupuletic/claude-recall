#!/usr/bin/env bash
set -euo pipefail

# claude-recall installer
# Usage: curl -sSL https://raw.githubusercontent.com/anthropics/claude-recall/main/install.sh | bash

PACKAGE="claude-recall"
EXTRAS="textual fastembed sqlite-vec"

# Use local path if running from the repo directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ -f "$SCRIPT_DIR/pyproject.toml" ]; then
    PACKAGE="$SCRIPT_DIR"
fi

echo "Installing claude-recall..."
echo ""

# Detect best installer
if command -v uv &>/dev/null; then
    echo "Found uv — installing with uv tool..."
    WITH_ARGS=""
    for extra in $EXTRAS; do
        WITH_ARGS="$WITH_ARGS --with $extra"
    done
    uv tool install "$PACKAGE" $WITH_ARGS
    uv tool update-shell 2>/dev/null || true

elif command -v pipx &>/dev/null; then
    echo "Found pipx — installing..."
    pipx install "${PACKAGE}[all]"

elif command -v pip3 &>/dev/null; then
    echo "Found pip3 — installing..."
    pip3 install --user "${PACKAGE}[all]"

elif command -v pip &>/dev/null; then
    echo "Found pip — installing..."
    pip install --user "${PACKAGE}[all]"

else
    echo "Error: No Python package manager found."
    echo "Install one of: uv, pipx, or pip"
    echo ""
    echo "  curl -LsSf https://astral.sh/uv/install.sh | sh"
    exit 1
fi

echo ""
echo "Building search index (first run)..."
claude-recall index --quiet 2>/dev/null || true

echo ""
echo "Done! Try it:"
echo ""
echo "  claude-recall \"debugging auth middleware\""
echo "  claude-recall                              # interactive TUI"
echo ""
