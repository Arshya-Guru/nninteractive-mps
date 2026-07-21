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

Patch 2 — float64 interpolation in autozoom (inference_session._predict)
    Two spots in the zoom code upsample a prediction with `interpolate(x.to(float), ...)`.
    Python's `float` maps to torch float64, which MPS does not support ("Cannot convert a
    MPS Tensor to float64 dtype"). Both cases hold integer class labels and use
    nearest/trilinear interpolation, so float32 is exact enough:
      - the border-change check (`mode='nearest'`)
      - the final resize back to patch size (`mode='trilinear'`)
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
        (
            "interpolate(previous_zoom_prediction[None, None].to(float), pred.shape, mode='nearest')",
            "interpolate(previous_zoom_prediction[None, None].to(torch.float32), pred.shape, mode='nearest')",
        ),
        (
            "interpolate(pred[None, None].to(float), scaled_patch_size, mode='trilinear')",
            "interpolate(pred[None, None].to(torch.float32), scaled_patch_size, mode='trilinear')",
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
