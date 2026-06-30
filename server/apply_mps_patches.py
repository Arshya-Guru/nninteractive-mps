"""
Idempotent source patches that make nnInteractive 1.0.1 run on the Apple MPS backend.

These are surgical fixes for spots where the CUDA/CPU code paths rely on behavior the
MPS backend doesn't share. Each patch is a plain string replacement applied to the
installed package source; re-running is a no-op once applied. Run automatically by
setup.sh and start.sh.

Patch 1 — scalar index in autozoom border check (inference_session._predict)
    The autozoom logic calls `tensor.index_select(dim, torch.tensor(idx, ...))`, where
    `torch.tensor(idx)` is a 0-dim scalar. CPU/CUDA tolerate a scalar index here, but the
    MPS index_select rejects it ("Dimension specified as -1 but tensor has no dimensions").
    Making the index 1-D (`[idx]`) is mathematically identical for the subsequent
    `sum` / `!=` reductions and works on every backend.
"""

import importlib.util
import pathlib
import sys


PATCHES = {
    "nnInteractive.inference.inference_session": [
        (
            "torch.tensor(idx, device=self.device)",
            "torch.tensor([idx], device=self.device)",
        ),
    ],
}


def _patch_module(module_name, replacements):
    spec = importlib.util.find_spec(module_name)
    if spec is None or not spec.origin:
        print(f"[mps-patch] WARNING: could not locate {module_name}", file=sys.stderr)
        return False
    path = pathlib.Path(spec.origin)
    text = path.read_text()
    original = text
    for old, new in replacements:
        if old in text:
            text = text.replace(old, new)
    if text != original:
        path.write_text(text)
        print(f"[mps-patch] patched {path}")
        return True
    print(f"[mps-patch] {path} already up to date")
    return False


def main():
    for module_name, replacements in PATCHES.items():
        _patch_module(module_name, replacements)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
