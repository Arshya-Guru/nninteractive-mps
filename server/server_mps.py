"""
nnInteractive server — Apple Silicon (MPS) launcher.

Starts the official nnInteractive v2 inference server (`nninteractive-server`, shipped with
the `nnInteractive` package, https://github.com/MIC-DKFZ/nnInteractive) on the best available
device: Apple GPU (MPS) on an Apple Silicon Mac, CUDA on an NVIDIA box, CPU otherwise.

The current SlicerNNInteractive extension talks to exactly this server (Remote mode), so no
custom HTTP API is needed anymore — just point the extension's Server URL at
http://localhost:1527.

Why this works with no patches to the model code: nnInteractive v2 already keeps every
CUDA-only path (fp16 autocast, pinned staging buffers, cuDNN benchmark, torch.compile, the
GPU-resident refinement cache) behind `device.type == 'cuda'`, and nnU-Net's `empty_cache`
dispatches per backend. The upstream server simply defaults to `--device cuda`; this
launcher picks the device for you and turns on PyTorch's MPS-to-CPU op fallback.

Any extra command-line arguments are passed straight through to `nninteractive-server`
(e.g. `pixi run start --port 1530`, or `--device cpu` to override the device choice).
"""

import os
import sys

# Must be set BEFORE torch is imported. The nnInteractive network is a 3D-conv U-Net;
# the vast majority of ops have native MPS kernels, but if any single op lacks one,
# this lets PyTorch transparently run just that op on the CPU instead of crashing.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import torch  # noqa: E402

from nnInteractive.inference.server.main import main as server_main  # noqa: E402


def pick_device() -> str:
    """Prefer CUDA, then Apple MPS, then CPU."""
    if torch.cuda.is_available():
        return "cuda:0"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def flag_value(argv, flag):
    """Value of `--flag value` / `--flag=value` in argv, True for a bare flag, else None."""
    for i, a in enumerate(argv):
        if a.startswith(flag + "="):
            return a.split("=", 1)[1]
        if a == flag:
            return argv[i + 1] if i + 1 < len(argv) and not argv[i + 1].startswith("--") else True
    return None


def main() -> int:
    argv = sys.argv[1:]

    device = flag_value(argv, "--device")
    if device is None:
        device = pick_device()
        argv += ["--device", device]
    if flag_value(argv, "--torch-n-threads") is None:
        # Upstream defaults to 8; use every core (CPU-side preprocessing and any ops
        # that fall back from MPS to CPU).
        argv += ["--torch-n-threads", str(os.cpu_count() or 8)]
    if not str(device).startswith("cuda") and flag_value(argv, "--no-torch-compile") is None:
        # torch.compile only pays off on CUDA; upstream disables it elsewhere anyway but
        # logs a warning. Pass the flag so the log stays clean.
        argv.append("--no-torch-compile")

    print(f"[nninteractive-mps] torch {torch.__version__}; nninteractive-server {' '.join(argv)}", flush=True)
    return server_main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
