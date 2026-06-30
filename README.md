# nnInteractive on Apple Silicon (MPS)

Run the **nnInteractive** interactive-segmentation server on a Mac's GPU (Metal / MPS),
so you get the same "double-click a launcher, then drive it from 3D Slicer" workflow as a
lab machine — but without an NVIDIA GPU.

## How it works

nnInteractive in Slicer is a **client–server** system:

- The **nnInteractive extension inside Slicer** is just the client. It sends your
  clicks / scribbles / boxes to a server over HTTP and draws back the returned mask.
- The **server** is what loads the AI model onto the GPU and runs inference.

The official server assumes Linux/Windows + an NVIDIA GPU (CUDA). This project swaps in
**device auto-selection** so the same server runs the model on Apple Silicon via PyTorch's
MPS backend. Everything else — the HTTP API, the port (`1527`), the Slicer extension — is
unchanged.

Why so little had to change: the nnInteractive inference engine (v1.0.1) already guards all
its CUDA-only optimizations (pinned memory, fp16 autocast, async copies) behind
`device.type == 'cuda'`, and cache clearing is dispatched per-backend. The one thing that
forced CUDA was the device the server handed to the model — which is what `server/server_mps.py`
changes. (See `NOTICE.md` for attribution.)

## Requirements

- Apple Silicon Mac (M1/M2/M3/M4). MPS gives the GPU speedup; on an Intel Mac it still runs, on CPU.
- macOS with **Python 3.10–3.13** (`python3 --version`). The pinned stack (torch 2.8.0)
  has no wheels for Python 3.14+, so if your default `python3` is 3.14, install 3.12:
  `brew install python@3.12`. `setup.sh` auto-detects a compatible interpreter.
- ~16 GB RAM recommended. The model runs in float32 on MPS (no fp16 autocast), so it uses
  more memory than the CUDA build.
- 3D Slicer (already installed on this machine).

## 1. One-time setup

From this folder, in Terminal:

```bash
cd "nninteractive mps"
chmod +x setup.sh start.sh "Start nnInteractive MPS.command"
./setup.sh
```

This creates a `.venv` and installs the pinned stack (PyTorch 2.8 + nnInteractive 1.0.1,
which on arm64 macOS includes the MPS backend). It prints whether MPS is available at the end.

## 2. Start the server

Double-click **`Start nnInteractive MPS.command`** in Finder (the desktop-launcher equivalent),
or run `./start.sh`. Leave the window open while you work.

- The **first** launch downloads the model weights (a few hundred MB) into
  `server/.nninteractive_weights/`. Later launches skip that.
- When it's ready you'll see uvicorn listening on `http://127.0.0.1:1527`.

Tip: to put the launcher on your Desktop like the lab machines, drag
`Start nnInteractive MPS.command` to the Desktop while holding ⌥⌘ (makes an alias).

## 3. Install the Slicer extension

In 3D Slicer: **Extensions Manager → search "nnInteractive" → Install → restart Slicer.**
(Extension repo: https://github.com/coendevente/SlicerNNInteractive)

## 4. Point Slicer at your local server

1. Load a volume in Slicer.
2. Open the **nnInteractive** module.
3. In its **Configuration** tab, set **Server URL** to:

   ```
   http://localhost:1527
   ```

   (It must start with `http://`.) Click to verify it's reachable — the server window will
   show the request.
4. Use points / bounding box / scribble / lasso to segment. Each interaction is sent to the
   local server, run on the Apple GPU, and the mask comes back.

## Performance notes

- MPS runs the network in **float32** (the CUDA build uses fp16 autocast), so expect a single
  interaction to be slower than an NVIDIA workstation and to use more memory. Still interactive
  for typical volumes.
- `PYTORCH_ENABLE_MPS_FALLBACK=1` is set by the launcher: if any single op lacks an MPS kernel,
  PyTorch runs just that op on the CPU instead of erroring. If you see a hard "not implemented
  for MPS" crash, that's the thing to investigate first.

## Troubleshooting

- **"No .venv found"** → run `./setup.sh`.
- **Slicer can't reach the server** → confirm the launcher window is still open and the URL is
  exactly `http://localhost:1527`.
- **Port already in use** → start with a different port (`./start.sh` calls
  `server_mps.py --port 1530`) and set the same port in Slicer.
- **`MPS available: False`** → you're likely on an Intel Mac or an old PyTorch; it'll run on CPU.

## Layout

```
nninteractive mps/
├─ Start nnInteractive MPS.command   # double-click launcher (keeps Terminal open)
├─ setup.sh                          # one-time: create venv + install deps
├─ start.sh                          # start the server (used by the launcher)
├─ server/
│  ├─ server_mps.py                  # MPS-aware server (device auto-select)
│  └─ requirements.txt
├─ NOTICE.md                         # attribution / licenses
└─ README.md
```
