"""
nnInteractive Slicer server — Apple Silicon (MPS) build.

This is a port of the upstream SlicerNNInteractive server
(https://github.com/coendevente/SlicerNNInteractive, Apache-2.0) that auto-selects
the inference device instead of hardcoding CUDA. On an Apple Silicon Mac it runs
the nnInteractive model on the GPU via PyTorch's MPS (Metal) backend; it still
prefers CUDA on an NVIDIA box and falls back to CPU otherwise.

The HTTP API (endpoints, request/response shapes, default port 1527) is byte-for-byte
compatible with the upstream server, so the unmodified SlicerNNInteractive extension
talks to it without any changes — just point its Server URL at http://localhost:1527.

Why this works with no changes to the model code: the nnInteractive inference session
(v1.0.1) already guards every CUDA-only path (pinned memory, autocast, non-blocking
host->device copies) behind `device.type == 'cuda'`, and `empty_cache` is dispatched
per-backend by nnunetv2. The only thing that forced CUDA was the device argument the
upstream server passed into the session — which is exactly what we change here.
"""

import os

# Must be set BEFORE torch is imported. The nnInteractive network is a 3D-conv U-Net;
# the vast majority of ops have native MPS kernels, but if any single op lacks one,
# this lets PyTorch transparently run just that op on the CPU instead of crashing.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import time
import warnings
import io
import gzip
import hashlib
import argparse

import numpy as np
import torch
import uvicorn

import xxhash

from pydantic import BaseModel
from huggingface_hub import snapshot_download

from nnInteractive.inference.inference_session import nnInteractiveInferenceSession

from fastapi import FastAPI, Response, UploadFile, File, Form


###############################################################################
# Device selection
###############################################################################
def pick_device() -> torch.device:
    """Prefer CUDA, then Apple MPS, then CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


###############################################################################
# Global constants & FastAPI app
###############################################################################
REPO_ID = "nnInteractive/nnInteractive"
MODEL_NAME = "nnInteractive_v1.0"  # Updated models may be available in the future
DOWNLOAD_DIR = ".nninteractive_weights"  # Specify the download directory

app = FastAPI()

###############################################################################
# Utility / helper functions
###############################################################################


def calculate_md5_array(image_data, xx=False):
    """Calculate either an xxHash (if xx=True) or MD5 hash of a NumPy array's bytes."""
    if xx:
        xh = xxhash.xxh64()
        xh.update(image_data.tobytes())
        out_hash = xh.hexdigest()
    else:
        md5_hash = hashlib.md5()
        md5_hash.update(image_data.tobytes())
        out_hash = md5_hash.hexdigest()
    return out_hash


def unpack_binary_segmentation(binary_data, vol_shape):
    """Unpacks binary data (1 bit per voxel) into a full 3D numpy array (bool type)."""
    total_voxels = np.prod(vol_shape)
    unpacked_bits = np.unpackbits(np.frombuffer(binary_data, dtype=np.uint8))
    unpacked_bits = unpacked_bits[:total_voxels]
    segmentation_mask = (
        unpacked_bits.reshape(vol_shape).astype(np.bool_).astype(np.uint8)
    )
    return segmentation_mask


def segmentation_binary(seg_in, compress=False):
    """Convert a (boolean) segmentation array into packed bits and optionally compress."""
    seg_result = seg_in.astype(bool)
    packed_segmentation = np.packbits(seg_result, axis=None)
    packed_segmentation = packed_segmentation.tobytes()
    if compress:
        packed_segmentation = gzip.compress(packed_segmentation)
    return packed_segmentation


def process_mask_and_click_input(file_bytes, positive_click):
    """Decompress file_bytes, load the numpy mask, interpret positive_click as bool."""
    positive_click_bool = positive_click.lower() in ["true", "1", "yes"]

    error = get_error_if_img_not_set()
    if error is not None:
        return error

    try:
        decompressed = gzip.decompress(file_bytes)
    except Exception as e:
        return {"status": "error", "message": f"Decompression failed: {e}"}

    mask = np.load(io.BytesIO(decompressed))
    return mask, positive_click_bool


def get_error_if_img_not_set():
    if PROMPT_MANAGER.img is None:
        warnings.warn("There is no image in the server. Be sure to send it before")
        return {"status": "error", "message": "No image uploaded"}
    return


###############################################################################
# PromptManager class
###############################################################################
class PromptManager:
    """Manages the image, target tensor, and runs inference sessions."""

    def __init__(self):
        self.img = None
        self.target_tensor = None

        self.device = pick_device()
        print(f"[nninteractive-mps] torch {torch.__version__}, inference device: {self.device}")

        self.download_weights()
        self.session = self.make_session()

    def download_weights(self):
        """Downloads only the files matching 'MODEL_NAME/*' into DOWNLOAD_DIR."""
        snapshot_download(
            repo_id=REPO_ID, allow_patterns=[f"{MODEL_NAME}/*"], local_dir=DOWNLOAD_DIR
        )

    def make_session(self):
        """Creates an nnInteractiveInferenceSession on the selected device."""
        # Pinned memory only ever helps CUDA host->device transfers; the session
        # already no-ops it off-CUDA, but we pass the honest flag anyway.
        use_pinned = self.device.type == "cuda"
        session = nnInteractiveInferenceSession(
            device=self.device,
            use_torch_compile=False,
            verbose=True,
            torch_n_threads=os.cpu_count(),
            do_autozoom=True,
            use_pinned_memory=use_pinned,
        )

        model_path = os.path.join(DOWNLOAD_DIR, MODEL_NAME)
        session.initialize_from_trained_model_folder(model_path)
        return session

    def set_image(self, input_image):
        """Loads the user-provided 3D image into the session, resets interactions."""
        self.session.reset_interactions()

        self.img = input_image[None]  # Ensure shape (1, x, y, z)
        self.session.set_image(self.img)

        print("self.img.shape:", self.img.shape)

        if self.img.ndim != 4:
            raise ValueError("Input image must be 4D with shape (1, x, y, z)")

        self.target_tensor = torch.zeros(self.img.shape[1:], dtype=torch.uint8)
        self.session.set_target_buffer(self.target_tensor)

    def set_segment(self, mask):
        """Sets or resets a segmentation (mask) on the server side."""
        if np.sum(mask) == 0:
            self.session.reset_interactions()
            self.target_tensor = torch.zeros(self.img.shape[1:], dtype=torch.uint8)
            self.session.set_target_buffer(self.target_tensor)
        else:
            self.session.add_initial_seg_interaction(mask)

    def add_point_interaction(self, point_coordinates, include_interaction):
        self.session.add_point_interaction(
            point_coordinates, include_interaction=include_interaction
        )
        return self.target_tensor.clone().cpu().detach().numpy()

    def add_bbox_interaction(self, outer_point_one, outer_point_two, include_interaction):
        print("outer_point_one, outer_point_two:", outer_point_one, outer_point_two)

        data = np.array([outer_point_one, outer_point_two])
        _min = np.min(data, axis=0)
        _max = np.max(data, axis=0)

        bbox = [
            [int(_min[0]), int(_max[0])],
            [int(_min[1]), int(_max[1])],
            [int(_min[2]), int(_max[2])],
        ]

        self.session.add_bbox_interaction(bbox, include_interaction=include_interaction)
        return self.target_tensor.clone().cpu().detach().numpy()

    def add_lasso_interaction(self, mask, include_interaction):
        print("Lasso mask received with shape:", mask.shape)
        self.session.add_lasso_interaction(mask, include_interaction=include_interaction)
        return self.target_tensor.clone().cpu().detach().numpy()

    def add_scribble_interaction(self, mask, include_interaction):
        print("Scribble mask received with shape:", mask.shape)
        self.session.add_scribble_interaction(mask, include_interaction=include_interaction)
        return self.target_tensor.clone().cpu().detach().numpy()


###############################################################################
# Global prompt manager instance
###############################################################################
PROMPT_MANAGER = PromptManager()


###############################################################################
# FastAPI endpoints
###############################################################################
@app.post("/upload_image")
async def upload_image(file: UploadFile = File(None)):
    file_bytes = await file.read()
    arr = np.load(io.BytesIO(file_bytes))
    PROMPT_MANAGER.set_image(arr)
    return {"status": "ok"}


@app.post("/upload_segment")
async def upload_segment(file: UploadFile = File(None)):
    error = get_error_if_img_not_set()
    if error is not None:
        return error

    file_bytes = await file.read()
    decompressed = gzip.decompress(file_bytes)
    arr = np.load(io.BytesIO(decompressed))

    PROMPT_MANAGER.set_segment(arr)
    return {"status": "ok"}


class PointParams(BaseModel):
    voxel_coord: list[int]
    positive_click: bool


@app.post("/add_point_interaction")
async def add_point_interaction(params: PointParams):
    error = get_error_if_img_not_set()
    if error is not None:
        return error

    t = time.time()
    seg_result = PROMPT_MANAGER.add_point_interaction(
        point_coordinates=params.voxel_coord, include_interaction=params.positive_click
    )
    compressed_bin = segmentation_binary(seg_result, compress=True)
    print(f"Server whole infer function time: {time.time() - t}")

    return Response(
        content=compressed_bin,
        media_type="application/octet-stream",
        headers={"Content-Encoding": "gzip"},
    )


class BBoxParams(BaseModel):
    outer_point_one: list[int]
    outer_point_two: list[int]
    positive_click: bool


@app.post("/add_bbox_interaction")
async def add_bbox_interaction(params: BBoxParams):
    error = get_error_if_img_not_set()
    if error is not None:
        return error

    t = time.time()
    seg_result = PROMPT_MANAGER.add_bbox_interaction(
        params.outer_point_one,
        params.outer_point_two,
        include_interaction=params.positive_click,
    )
    segmentation_binary_data = segmentation_binary(seg_result, compress=True)
    print(f"Server whole infer function time: {time.time() - t}")

    return Response(
        content=segmentation_binary_data,
        media_type="application/octet-stream",
        headers={"Content-Encoding": "gzip"},
    )


@app.post("/add_lasso_interaction")
async def add_lasso_interaction(file: UploadFile = File(...), positive_click: str = Form(...)):
    error = get_error_if_img_not_set()
    if error is not None:
        return error

    file_bytes = await file.read()
    mask, positive_click_bool = process_mask_and_click_input(file_bytes, positive_click)

    seg_result = PROMPT_MANAGER.add_lasso_interaction(
        mask, include_interaction=positive_click_bool
    )
    segmentation_binary_data = segmentation_binary(seg_result, compress=True)

    return Response(
        content=segmentation_binary_data,
        media_type="application/octet-stream",
        headers={"Content-Encoding": "gzip"},
    )


@app.post("/add_scribble_interaction")
async def add_scribble_interaction(file: UploadFile = File(...), positive_click: str = Form(...)):
    error = get_error_if_img_not_set()
    if error is not None:
        return error

    file_bytes = await file.read()
    mask, positive_click_bool = process_mask_and_click_input(file_bytes, positive_click)

    seg_result = PROMPT_MANAGER.add_scribble_interaction(
        mask, include_interaction=positive_click_bool
    )
    segmentation_binary_data = segmentation_binary(seg_result, compress=True)

    return Response(
        content=segmentation_binary_data,
        media_type="application/octet-stream",
        headers={"Content-Encoding": "gzip"},
    )


def main():
    parser = argparse.ArgumentParser(description="Run the nnInteractive Slicer server (MPS build).")
    parser.add_argument("--host", default="127.0.0.1", help="Host interface to bind to.")
    parser.add_argument("--port", type=int, default=1527, help="Port to listen on.")
    args = parser.parse_args()

    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    print(f"torch.__version__: {torch.__version__}")
    main()
