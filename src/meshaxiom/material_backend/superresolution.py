"""Optional FP32 RealESRGAN x2plus atlas inference, isolated from BasicSR."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

WEIGHTS_SHA256 = "49fafd45f8fd7aa8d31ab2a22d14d91b536c34494a5cfe31eb5d89c2fa266abb"
WEIGHTS_URL = "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth"
DEFAULT_TILE = 512
HALO = 192


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_checkpoint(binding):
    if not isinstance(binding, dict) or not binding.get("path"):
        raise ValueError("material.superresolution_checkpoint must contain path and sha256")
    if binding.get("sha256") != WEIGHTS_SHA256:
        raise ValueError("This profile requires the pinned RealESRGAN_x2plus SHA256")
    path = Path(binding["path"]).expanduser().resolve(strict=True)
    if not path.is_file() or sha256(path) != WEIGHTS_SHA256:
        raise ValueError("RealESRGAN_x2plus weight SHA256 mismatch")
    return path


def validate_vendor():
    root = Path(__file__).resolve().parents[1] / "vendor/superresolution"
    manifest = json.loads((root / "SOURCE_MANIFEST.json").read_text())
    for entry in manifest["files"]:
        if sha256(root / entry["file"]) != entry["vendored_sha256"]:
            raise ValueError("Vendored RRDB source or license changed: " + entry["file"])
    return {"manifest_sha256": sha256(root / "SOURCE_MANIFEST.json"), "architecture": manifest}


def load_model(checkpoint, device="cuda:0"):
    """Load the exact trained EMA model without importing the basicsr package."""
    path = validate_checkpoint(checkpoint)
    validate_vendor()
    import torch
    from meshaxiom.vendor.superresolution.rrdbnet_arch import RRDBNet

    model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=2)
    state = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(state, dict) or "params_ema" not in state:
        raise ValueError("Pinned checkpoint does not contain EMA weights")
    loaded = model.load_state_dict(state["params_ema"], strict=True)
    if loaded.missing_keys or loaded.unexpected_keys:
        raise ValueError("RealESRGAN EMA state failed strict loading")
    del state
    return model.eval().requires_grad_(False).to(device=device, dtype=torch.float32)


def validate_tile(tile):
    if isinstance(tile, bool) or not isinstance(tile, int) or not 32 <= tile <= DEFAULT_TILE or tile % 2:
        raise ValueError("tile must be an even integer between 32 and 512; halo stays fixed at 192")
    return tile


def upscale(model, rgb, tile=DEFAULT_TILE):
    """Apply tiled x2 RGB inference and prove exact output texel coverage.

    A right/bottom replicated pixel handles odd source dimensions, followed by
    cropping back to exactly 2H x 2W. Tile starts and halo use even positions,
    preserving the RRDB pixel-unshuffle lattice. No blend/feather changes UVs.
    """
    import torch

    tile = validate_tile(tile)
    rgb = np.asarray(rgb)
    if rgb.ndim != 3 or rgb.shape[2] != 3 or not rgb.shape[0] or not rgb.shape[1] or rgb.dtype != np.uint8:
        raise ValueError("Superresolution input must be a nonempty H x W x 3 uint8 RGB image")
    device = next(model.parameters()).device
    if next(model.parameters()).dtype != torch.float32:
        raise ValueError("The accepted superresolution profile uses FP32 weights")
    original_h, original_w = rgb.shape[:2]
    pad_h, pad_w = original_h % 2, original_w % 2
    padded = np.pad(rgb, ((0, pad_h), (0, pad_w), (0, 0)), mode="edge") if pad_h or pad_w else rgb
    h, w = padded.shape[:2]
    result = np.empty((h * 2, w * 2, 3), dtype=np.uint8)
    coverage = np.zeros((h * 2, w * 2), dtype=np.uint8)
    tensor = torch.from_numpy(np.ascontiguousarray(padded.transpose(2, 0, 1))).to(device=device, dtype=torch.float32).unsqueeze(0) / 255
    count = 0
    with torch.inference_mode():
        for y in range(0, h, tile):
            for x in range(0, w, tile):
                y_end, x_end = min(h, y + tile), min(w, x + tile)
                y_pad, x_pad = max(0, y - HALO), max(0, x - HALO)
                y_pad_end, x_pad_end = min(h, y_end + HALO), min(w, x_end + HALO)
                prediction = model(tensor[:, :, y_pad:y_pad_end, x_pad:x_pad_end])
                expected_shape = (1, 3, (y_pad_end - y_pad) * 2, (x_pad_end - x_pad) * 2)
                if tuple(prediction.shape) != expected_shape or not torch.isfinite(prediction).all():
                    raise RuntimeError(f"Invalid or nonfinite x2 model output at tile ({x}, {y})")
                core = prediction[0, :, (y - y_pad) * 2:(y_end - y_pad) * 2, (x - x_pad) * 2:(x_end - x_pad) * 2]
                pixels = np.rint(core.clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
                result[y * 2:y_end * 2, x * 2:x_end * 2] = pixels
                coverage[y * 2:y_end * 2, x * 2:x_end * 2] += 1
                count += 1
    if not np.all(coverage == 1):
        raise RuntimeError("Missing or multiply written texture texels")
    result = result[:original_h * 2, :original_w * 2].copy()
    visible_coverage = coverage[:original_h * 2, :original_w * 2]
    audit = {
        "scale": 2, "input_dimensions": [original_w, original_h], "output_dimensions": [original_w * 2, original_h * 2],
        "precision": "float32", "core_tile": tile, "context_halo": HALO, "tiles": count,
        "tile_coverage_exact": True, "coverage_min": int(visible_coverage.min()), "coverage_max": int(visible_coverage.max()),
        "output_texels_written_once": int(visible_coverage.size), "tile_lattice_even": True,
        "input_padding_right": pad_w, "input_padding_bottom": pad_h,
        "padding_policy": "replicate right/bottom edge pixel to even input, then crop to exact x2 dimensions",
        "blend": "none; crop each context tile to its original core",
    }
    return result, audit


def upscale_alpha(alpha):
    """Preserve alpha as exact repeated coverage; no AI-generated transparency."""
    alpha = np.asarray(alpha)
    if alpha.ndim != 2 or alpha.dtype != np.uint8:
        raise ValueError("Alpha must be an H x W uint8 channel")
    return np.repeat(np.repeat(alpha, 2, axis=0), 2, axis=1)
