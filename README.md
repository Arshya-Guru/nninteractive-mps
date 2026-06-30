# nnInteractive on Apple Silicon (MPS)

Run the [nnInteractive](https://github.com/MIC-DKFZ/nnInteractive) interactive-segmentation
server on a Mac's GPU (Metal / MPS) and drive it from 3D Slicer — no NVIDIA card required.

The official server is CUDA-only. This is a thin port that runs the same model on Apple
Silicon via PyTorch's MPS backend. The HTTP API, the port (`1527`), and the Slicer
extension are all unchanged.

---

## Quick start

**You need:**
- An Apple Silicon Mac (M1/M2/M3/M4) running macOS.
- [3D Slicer](https://download.slicer.org/) installed.
- Python **3.10–3.13** (check with `python3 --version`). If you only have 3.14+,
  install 3.12 first: `brew install python@3.12`.
- [Git](https://git-scm.com/download/mac) (macOS prompts to install it the first
  time you run `git`).

**Open Terminal** (⌘+Space → "Terminal"), then run:

```bash
git clone https://github.com/Arshya-Guru/nninteractive-mps.git
cd nninteractive-mps
chmod +x setup.sh start.sh "Start nnInteractive MPS.command"
./setup.sh
```

`setup.sh` creates a `.venv` and installs the pinned stack (PyTorch 2.8 +
nnInteractive 1.0.1). It takes a few minutes and prints whether MPS is available at
the end.

## Start the server

Either **double-click `Start nnInteractive MPS.command`** in Finder, or from the same
folder in Terminal:

```bash
./start.sh
```

Leave the window open while you work. When you see uvicorn listening on
`http://127.0.0.1:1527`, the server is ready. The **first** launch downloads
~hundreds of MB of model weights into `server/.nninteractive_weights/`; later
launches skip that.

## Connect 3D Slicer to it

1. In Slicer: **Extensions Manager → search "nnInteractive" → Install → restart Slicer.**
   (Extension repo: <https://github.com/coendevente/SlicerNNInteractive>.)
2. Load a volume.
3. Open the **nnInteractive** module → **Configuration** tab.
4. Set **Server URL** to `http://localhost:1527` (the `http://` prefix is required) and
   verify it's reachable. The terminal window running the server will log the request.
5. Use points / bounding box / scribble / lasso to segment. Each interaction is sent to
   the local server, run on the Apple GPU, and the mask comes back.

> **Tip:** drag `Start nnInteractive MPS.command` to your Desktop while holding ⌥⌘ to
> make a launcher alias, matching the workflow on the lab's Linux/NVIDIA boxes.

---

## Troubleshooting

| Symptom | Fix |
|---|---|
| `No .venv found` | Run `./setup.sh` first. |
| Slicer can't reach the server | Check the launcher window is still open and the URL is exactly `http://localhost:1527`. |
| Port already in use | Start with a different port (`./start.sh` calls `server_mps.py --port 1530`) and set the same port in Slicer. |
| `MPS available: False` at end of setup | You're on an Intel Mac or an old PyTorch — the server still runs, just on CPU. |
| Hard `"not implemented for MPS"` crash | An op is missing an MPS kernel. The launcher already sets `PYTORCH_ENABLE_MPS_FALLBACK=1` so this is rare; if it still happens, file an issue with the op name. |

## Performance notes

- The model runs in **float32** on MPS (the CUDA build uses fp16 autocast), so expect
  more memory use and slower per-interaction latency than an NVIDIA workstation. Still
  interactive for typical volumes; **~16 GB RAM recommended**.
- `PYTORCH_ENABLE_MPS_FALLBACK=1` is set by `start.sh` so unsupported ops fall back to
  CPU rather than crashing.

## How it works

nnInteractive in Slicer is a **client–server** system: the Slicer extension is just a
client that POSTs your clicks/scribbles/boxes to a server and renders the returned mask.
The server is what loads the model onto the GPU and runs inference.

The nnInteractive inference engine (v1.0.1) already guards its CUDA-only optimizations
(pinned memory, fp16 autocast, async copies) behind `device.type == 'cuda'`, and cache
clearing is dispatched per-backend. The only thing forcing CUDA was the device the
server handed to the model — which is what `server/server_mps.py` changes via
device auto-selection. A small set of source patches to the installed nnInteractive
package (applied idempotently by `server/apply_mps_patches.py` on every start) covers
the remaining MPS edge cases.

See [`NOTICE.md`](NOTICE.md) for attribution and licenses.

## Layout

```
nninteractive-mps/
├─ Start nnInteractive MPS.command   # double-click launcher (keeps Terminal open)
├─ setup.sh                          # one-time: create venv + install deps
├─ start.sh                          # start the server (used by the launcher)
├─ server/
│  ├─ server_mps.py                  # MPS-aware server (device auto-select)
│  ├─ apply_mps_patches.py           # idempotent source patches for MPS
│  └─ requirements.txt
├─ NOTICE.md                         # attribution / licenses
├─ LICENSE
└─ README.md
```
