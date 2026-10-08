# Attribution

This project runs the official **nnInteractive** server on the Apple Silicon (MPS) backend
so it can be used from the **SlicerNNInteractive** 3D Slicer extension (by Coen de Vente
and contributors, Apache License 2.0). Earlier versions of this repo adapted that
extension's original server.

- SlicerNNInteractive: https://github.com/coendevente/SlicerNNInteractive
- nnInteractive (model + inference engine), MIC-DKFZ: https://github.com/MIC-DKFZ/nnInteractive
- Paper: "SlicerNNInteractive: A 3D Slicer extension for nnInteractive" — https://arxiv.org/abs/2504.07991

`server/server_mps.py` is a thin launcher for the official `nninteractive-server` from the
`nnInteractive` package (MIC-DKFZ, Apache-2.0). It only picks the inference device
(CUDA → MPS → CPU) and enables PyTorch's MPS op fallback; the server, its HTTP API and the
model code are upstream's, unmodified.

The nnInteractive model weights downloaded on first launch are licensed separately under
Creative Commons Attribution-NonCommercial-ShareAlike 4.0 (CC BY-NC-SA 4.0); see
https://github.com/MIC-DKFZ/nnInteractive#license.
