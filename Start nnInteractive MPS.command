#!/bin/bash
# Double-click this file in Finder to start the nnInteractive server on the Apple GPU.
# (This is the Mac equivalent of the lab's desktop launcher script.)
# A Terminal window opens and stays open while the server runs; close it or press
# Ctrl-C to stop the server.

DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$DIR"

echo "================================================================"
echo "  nnInteractive (MPS) server"
echo "  Keep this window open. In 3D Slicer, set the extension's"
echo "  Server URL to:   http://localhost:1527"
echo "================================================================"
echo ""

bash "$DIR/start.sh"

echo ""
echo "Server stopped. You can close this window."
