#!/bin/bash
# Double-click this file in Finder to start the nnInteractive server on the Apple GPU.
# (This is the Mac equivalent of the lab's desktop launcher script.)
# A Terminal window opens and stays open while the server runs; close it or press
# Ctrl-C to stop the server.
#
# The first run solves + installs the environment (a few minutes); later runs are fast.

DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$DIR"

# Make sure pixi is on PATH even when launched from Finder (which uses a minimal env).
export PATH="$HOME/.pixi/bin:$PATH"

echo "================================================================"
echo "  nnInteractive (MPS) server"
echo "  Keep this window open. In 3D Slicer's nnInteractive module,"
echo "  choose Remote mode and set the server URL to:"
echo "      http://localhost:1527"
echo "================================================================"
echo ""

if ! command -v pixi >/dev/null 2>&1; then
  echo "ERROR: pixi is not installed. Install it once with:" >&2
  echo "    curl -fsSL https://pixi.sh/install.sh | bash" >&2
  echo "then re-open this launcher." >&2
  echo ""
  echo "You can close this window."
  exit 1
fi

pixi run start

echo ""
echo "Server stopped. You can close this window."
