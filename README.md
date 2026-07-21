# nnInteractive on Apple Silicon (MPS)

Run the [nnInteractive](https://github.com/MIC-DKFZ/nnInteractive) interactive-segmentation
server on a Mac's GPU (Metal / MPS) and drive it from 3D Slicer — no NVIDIA card required.

The official server is CUDA-only. This is a thin port that runs the same model on Apple
Silicon via PyTorch's MPS backend. The HTTP API, the port (`1527`), and the Slicer
extension are all unchanged.

> **Now managed by [pixi](https://pixi.sh).** Earlier versions used hand-rolled
> `setup.sh` / `start.sh` scripts that assumed you'd already installed the right Python
> and pip-installed everything by hand. Those are gone. pixi now owns the whole stack —
> the exact Python interpreter, PyTorch, nnInteractive, and every other dependency —
> pinned in `pixi.toml` and locked in `pixi.lock`. There is nothing to `pip install` and
> no Python version to match: one command solves, installs, patches, and launches. If you
> cloned an older copy, just `git pull` and run `pixi run start`.

---

## Quick start

**You need:**
- An Apple Silicon Mac (M1/M2/M3/M4) running macOS.
- [3D Slicer](https://download.slicer.org/) installed.
- [Git](https://git-scm.com/download/mac) (macOS prompts to install it the first
  time you run `git`).
- **pixi** — the one tool this project needs. It manages Python and every dependency for
  you, so you do **not** install Python, PyTorch, or anything else by hand. See step 1.

**Step 1 — install pixi (once per machine).** If you've never used pixi, open Terminal
(⌘+Space → "Terminal") and run:

```bash
curl -fsSL https://pixi.sh/install.sh | bash
```

Then **close and re-open Terminal** so `pixi` is on your `PATH`. Confirm it worked:

```bash
pixi --version
```

If that prints a version number, you're set. (Already have pixi? Skip to step 2.)

**Step 2 — clone and launch.** In Terminal:

```bash
git clone https://github.com/Arshya-Guru/nninteractive-mps.git
cd nninteractive-mps
pixi run start
```

That single command does everything: on the first run it solves and installs the pinned
stack (Python 3.12 + PyTorch 2.8 + nnInteractive 1.0.1) into a local `.pixi/`
environment, applies the MPS source patches, and starts the server. It takes a few
minutes the first time; later runs skip straight to launch.

Prefer clicking? Once step 1 is done, **double-click `Start nnInteractive MPS.command`**
in Finder instead — it runs the same `pixi run start`. (If pixi isn't installed yet, the
launcher window tells you exactly how to install it and then exits, so it can't fail
silently.)

Leave the window open while you work. When you see uvicorn listening on
`http://127.0.0.1:1527`, the server is ready. The **first** launch also downloads
~hundreds of MB of model weights into `server/.nninteractive_weights/`; later
launches skip that.

> Want to confirm the Apple GPU is being used? Run `pixi run verify`.

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
| `pixi: command not found` | Install pixi: `curl -fsSL https://pixi.sh/install.sh \| bash`, then re-open Terminal. |
| Slicer can't reach the server | Check the launcher window is still open and the URL is exactly `http://localhost:1527`. |
| Port already in use | Edit the `start` task's `--port` in `pixi.toml` and set the same port in Slicer. |
| `MPS available: False` (`pixi run verify`) | You're on an Intel Mac or an old PyTorch — the server still runs, just on CPU. |
| Hard `"not implemented for MPS"` crash | An op is missing an MPS kernel. The env already sets `PYTORCH_ENABLE_MPS_FALLBACK=1` so this is rare; if it still happens, file an issue with the op name. |
| Want a clean reinstall | Delete the `.pixi/` folder and run `pixi run start` again. |

## Performance notes

- The model runs in **float32** on MPS (the CUDA build uses fp16 autocast), so expect
  more memory use and slower per-interaction latency than an NVIDIA workstation. Still
  interactive for typical volumes; **~16 GB RAM recommended**.
- `PYTORCH_ENABLE_MPS_FALLBACK=1` is set by the pixi environment so unsupported ops fall
  back to CPU rather than crashing.

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
├─ Start nnInteractive MPS.command   # double-click launcher (runs `pixi run start`)
├─ pixi.toml                         # env + deps + tasks (install/patch/start in one)
├─ pixi.lock                          # exact resolved versions (reproducible installs)
├─ server/
│  ├─ server_mps.py                  # MPS-aware server (device auto-select)
│  ├─ apply_mps_patches.py           # idempotent source patches for MPS
│  └─ requirements.txt               # dependency notes (source of truth: pixi.toml)
├─ NOTICE.md                         # attribution / licenses
├─ LICENSE
└─ README.md
```
