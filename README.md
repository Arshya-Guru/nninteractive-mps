# nnInteractive on Apple Silicon (MPS)

Run the [nnInteractive](https://github.com/MIC-DKFZ/nnInteractive) interactive-segmentation
server on a Mac's GPU (Metal / MPS) and drive it from 3D Slicer — no NVIDIA card required.

The official server defaults to an NVIDIA GPU (CUDA). This repo is a thin wrapper that
installs the official **nnInteractive v2** server (`nninteractive-server`) and starts it on
Apple Silicon via PyTorch's MPS backend. The current Slicer extension connects to it
unchanged (Remote mode, port `1527`).

> **Updated for nnInteractive v2 (October 2026).** In July 2026 the SlicerNNInteractive
> extension was reworked to target nnInteractive v2 and no longer talks to the old
> v1-era server this repo used to ship — which is why the previous version stopped
> working with the latest Slicer. This repo now runs the official v2 server instead
> (nnInteractive 2.6, nnU-Net 2.8, PyTorch 2.11). The old v1 MPS source patches are
> gone: nnInteractive v2 fixed both problems upstream. If you cloned an older copy,
> `git pull` and run `pixi run start`; in Slicer, update the nnInteractive extension
> and switch it to **Remote** mode (see below).

> **Now managed by [pixi](https://pixi.sh).** Earlier versions used hand-rolled
> `setup.sh` / `start.sh` scripts that assumed you'd already installed the right Python
> and pip-installed everything by hand. Those are gone. pixi now owns the whole stack —
> the exact Python interpreter, PyTorch, nnInteractive, and every other dependency —
> pinned in `pixi.toml` and locked in `pixi.lock`. There is nothing to `pip install` and
> no Python version to match: one command solves, installs, and launches. If you
> cloned an older copy, just `git pull` and run `pixi run start`.

---

## Quick start

**You need:**
- An Apple Silicon Mac (M1/M2/M3/M4/M5) running macOS.
- [3D Slicer](https://download.slicer.org/) installed (a current release), with an
  up-to-date **nnInteractive** extension.
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
stack (Python 3.12 + PyTorch 2.11 + nnInteractive 2.6) into a local `.pixi/`
environment and starts the server on the Apple GPU. It takes a few
minutes the first time; later runs skip straight to launch.

Prefer clicking? Once step 1 is done, **double-click `Start nnInteractive MPS.command`**
in Finder instead — it runs the same `pixi run start`. (If pixi isn't installed yet, the
launcher window tells you exactly how to install it and then exits, so it can't fail
silently.)

Leave the window open while you work. When you see uvicorn listening on
`http://127.0.0.1:1527`, the server is ready. The **first** launch also downloads
the model weights (~400 MB, from Hugging Face) into `~/.nninteractive/`; later
launches skip that. (Older versions of this repo kept weights in
`server/.nninteractive_weights/` — you can delete that folder.)

> Want to confirm the Apple GPU is being used? Run `pixi run verify`.

## Connect 3D Slicer to it

1. In Slicer: **Extensions Manager → search "nnInteractive" → Install** (or **Update** if
   you already have it) **→ restart Slicer.**
   (Extension repo: <https://github.com/coendevente/SlicerNNInteractive>.)
2. Open the **nnInteractive** module. The first time, it asks what to install: pick
   **Client only (remote)**. It's small and doesn't need PyTorch inside Slicer — the
   model runs in this server instead. (If you already did a **Full** install, that's fine
   too; just switch the mode to **Remote** in step 4.)
3. Load a volume (e.g. drag your scan file into Slicer).
4. In the **Configuration** tab, choose **Remote**, set the server URL to
   `http://localhost:1527` (the `http://` prefix is required), leave the API key empty,
   and click **Test connection**.
5. In the **nnInteractive Prompts** tab, click **Initialize** (this uploads the image to
   the server), then use points / bounding box / scribble / lasso to segment. Each
   interaction runs on the Apple GPU and the mask comes back. `Ctrl+Z` undoes the last one.

> **Tip:** drag `Start nnInteractive MPS.command` to your Desktop while holding ⌥⌘ to
> make a launcher alias, matching the workflow on the lab's Linux/NVIDIA boxes.

---

## Troubleshooting

| Symptom | Fix |
|---|---|
| `pixi: command not found` | Install pixi: `curl -fsSL https://pixi.sh/install.sh \| bash`, then re-open Terminal. |
| Slicer can't reach the server | Check the launcher window is still open and the URL is exactly `http://localhost:1527`. |
| Port already in use | Start on another port with `pixi run start --port 1530` and set the same port in Slicer. |
| Slicer says the server/session expired | The server drops sessions idle for 10 minutes. Click **Initialize** again — your segmentation is kept. |
| Slicer won't connect / "outdated" warning | Update the nnInteractive extension in Slicer's Extensions Manager and use **Remote** mode. Slicer extensions from before July 2026 only spoke the old server protocol. |
| `MPS available: False` (`pixi run verify`) | You're on an Intel Mac or an old PyTorch — the server still runs, just on CPU. |
| Hard `"not implemented for MPS"` crash | An op is missing an MPS kernel. The env already sets `PYTORCH_ENABLE_MPS_FALLBACK=1` so this is rare; if it still happens, file an issue with the op name. |
| Want a clean reinstall | Delete the `.pixi/` folder and run `pixi run start` again. |

## Performance notes

- The model runs in **float32** on MPS (the CUDA path uses fp16 autocast), so expect
  more memory use and slower per-interaction latency than an NVIDIA workstation. Still
  interactive for typical volumes; **~16 GB RAM recommended**.
- `PYTORCH_ENABLE_MPS_FALLBACK=1` is set by the pixi environment so unsupported ops fall
  back to CPU rather than crashing.

## How it works

In Remote mode the Slicer extension is a lightweight client (`nninteractive-client`) that
sends your image and clicks/scribbles/boxes to an `nninteractive-server` and gets the
mask back. The server is what loads the model onto the GPU and runs inference.

The official nnInteractive v2 server is device-agnostic: its inference session keeps
every CUDA-only optimization (fp16 autocast, pinned staging buffers, cuDNN benchmark,
`torch.compile`, the GPU-resident refinement cache) behind `device.type == 'cuda'`, and
nnU-Net clears caches per backend. It just defaults to `--device cuda`.
`server/server_mps.py` picks the device for you (MPS on Apple Silicon, else CUDA, else
CPU), turns on PyTorch's MPS-to-CPU op fallback, and launches the official server. Any
extra arguments are passed through, e.g. `pixi run start --port 1530` or
`pixi run start --device cpu`; see `nninteractive-server --help` for the full list.

The two MPS bugs that the old version patched in nnInteractive 1.0.1 (a 0-dim
`index_select` index and a float64 `interpolate` in autozoom) are both fixed upstream in
v2, so no source patching is needed anymore.

See [`NOTICE.md`](NOTICE.md) for attribution and licenses.

## Layout

```
nninteractive-mps/
├─ Start nnInteractive MPS.command   # double-click launcher (runs `pixi run start`)
├─ pixi.toml                         # env + deps + tasks (install/start in one)
├─ pixi.lock                         # exact resolved versions (reproducible installs)
├─ server/
│  └─ server_mps.py                  # launches the official server on MPS (device auto-select)
├─ NOTICE.md                         # attribution / licenses
├─ LICENSE
└─ README.md
```
