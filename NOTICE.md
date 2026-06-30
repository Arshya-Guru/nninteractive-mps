# Attribution

This project adapts the **SlicerNNInteractive** server for the Apple Silicon (MPS)
backend. The original server and 3D Slicer extension are by Coen de Vente and
contributors, licensed under the Apache License 2.0.

- SlicerNNInteractive: https://github.com/coendevente/SlicerNNInteractive
- nnInteractive (model + inference engine), MIC-DKFZ: https://github.com/MIC-DKFZ/nnInteractive
- Paper: "SlicerNNInteractive: A 3D Slicer extension for nnInteractive" — https://arxiv.org/abs/2504.07991

`server/server_mps.py` is derived from the upstream server's `main.py`. The only
functional change is device selection (CUDA → MPS → CPU) plus enabling PyTorch's
MPS op fallback; the HTTP API is unchanged so the upstream Slicer extension works
without modification.
