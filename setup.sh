#!/bin/bash
# One-time setup for the nnInteractive MPS server.
# Creates a Python virtual environment and installs the pinned dependency stack.
#
# IMPORTANT: the pinned stack (torch==2.8.0, numpy==2.2.3) only has wheels for
# Python 3.10-3.13. A newer interpreter (3.14+) has no matching torch wheel and the
# install fails. This script therefore searches for a compatible Python and uses it.
set -euo pipefail

cd "$(dirname "$0")"

# ---- Find a Python in the 3.10-3.13 range (prefer 3.12) --------------------
PYBIN=""
for cand in python3.12 python3.11 python3.13 python3.10; do
  if command -v "$cand" >/dev/null 2>&1; then
    PYBIN="$cand"
    break
  fi
done

# Fall back to a generic python3 only if its version is in range.
if [ -z "$PYBIN" ] && command -v python3 >/dev/null 2>&1; then
  if python3 -c 'import sys; raise SystemExit(0 if (3,10) <= sys.version_info[:2] <= (3,13) else 1)'; then
    PYBIN="python3"
  fi
fi

if [ -z "$PYBIN" ]; then
  echo "ERROR: No compatible Python found." >&2
  echo "The pinned stack (torch 2.8.0) needs Python 3.10-3.13, but only a newer/older" >&2
  echo "interpreter is installed. Install 3.12 and re-run this script:" >&2
  echo "" >&2
  echo "    brew install python@3.12" >&2
  echo "    ./setup.sh" >&2
  echo "" >&2
  echo "Currently on PATH: $(python3 --version 2>&1 || echo 'no python3')" >&2
  exit 1
fi

echo "==> Using interpreter: $PYBIN ($($PYBIN --version 2>&1))"

echo "==> Creating virtual environment (.venv)..."
rm -rf .venv
"$PYBIN" -m venv .venv
# shellcheck disable=SC1091
source .venv/bin/activate

echo "==> Upgrading pip..."
pip install --upgrade pip

echo "==> Installing dependencies (downloads PyTorch + nnInteractive; a few minutes)..."
pip install -r server/requirements.txt

echo "==> Applying MPS compatibility patches to nnInteractive..."
python server/apply_mps_patches.py

echo "==> Verifying the MPS (Apple GPU) backend is available..."
python - <<'PY'
import torch
print("torch:", torch.__version__)
mps_built = getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_built()
mps_avail = mps_built and torch.backends.mps.is_available()
print("MPS built:", mps_built, "| MPS available:", mps_avail)
if mps_avail:
    print("OK: the server will run on the Apple GPU (mps).")
else:
    print("NOTE: MPS not available - the server will fall back to CPU (much slower).")
PY

echo ""
echo "Setup complete. Start the server by double-clicking"
echo "  'Start nnInteractive MPS.command'  (or run ./start.sh)"
