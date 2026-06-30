#!/bin/bash
# Start the nnInteractive MPS server. Used by the double-click .command launcher,
# and can also be run directly from a terminal.
set -euo pipefail

cd "$(dirname "$0")"

# Lets PyTorch run any op that lacks an MPS kernel on the CPU instead of crashing.
export PYTORCH_ENABLE_MPS_FALLBACK=1

if [ ! -d .venv ]; then
  echo "No .venv found. Run ./setup.sh first." >&2
  exit 1
fi

# shellcheck disable=SC1091
source .venv/bin/activate

# Apply the MPS source patches to the installed nnInteractive package (idempotent).
python server/apply_mps_patches.py

# First launch downloads the model weights (~hundreds of MB) into
# server/.nninteractive_weights; subsequent launches are fast.
cd server
exec python server_mps.py --host 127.0.0.1 --port 1527
