"""Small runtime contracts shared by the two isolated geometry workers."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import sys
import time

DEFAULT_SEED = 2026082301


def sha256(path: str | Path) -> str:
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def binding(path: str | Path) -> dict:
    path = Path(path).resolve()
    return {"path": str(path), "size": path.stat().st_size, "sha256": sha256(path)}


def write_json(path: str | Path, value: dict) -> None:
    path = Path(path)
    tmp = path.with_name(path.name + f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    os.replace(tmp, path)


def write_glb(path: str | Path, vertices, faces) -> None:
    """Keep Z-up vertex storage, with an explicit scene transform into glTF Y-up."""
    import struct
    import numpy as np
    positions = np.asarray(vertices, dtype="<f4").reshape(-1, 3)
    indices = np.asarray(faces, dtype="<u4").reshape(-1, 3)
    if not len(positions) or not len(indices) or not np.isfinite(positions).all() or indices.max() >= len(positions):
        raise ValueError("GLB requires finite vertices and valid nonempty triangle indices")
    normals = np.zeros_like(positions)
    points = positions[indices]
    face_normals = np.cross(points[:, 1] - points[:, 0], points[:, 2] - points[:, 0])
    for corner in range(3):
        np.add.at(normals, indices[:, corner], face_normals)
    normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-20)
    pieces = [positions.tobytes(), normals.astype("<f4", copy=False).tobytes(), indices.tobytes()]
    offsets = [0, len(pieces[0]), len(pieces[0]) + len(pieces[1])]
    binary = b"".join(pieces)
    document = {
        "asset": {"version": "2.0", "generator": "MeshAxiom"}, "scene": 0,
        "scenes": [{"nodes": [0]}], "nodes": [{
            "mesh": 0, "rotation": [-(0.5 ** 0.5), 0.0, 0.0, 0.5 ** 0.5],
            "extras": {"vertex_storage_up_axis": "z", "world_up_axis": "y"},
        }],
        "meshes": [{"primitives": [{"attributes": {"POSITION": 0, "NORMAL": 1}, "indices": 2, "material": 0, "mode": 4}]}],
        "materials": [{"name": "Geometry", "doubleSided": True, "pbrMetallicRoughness": {
            "baseColorFactor": [0.65, 0.65, 0.65, 1.0], "metallicFactor": 0.0, "roughnessFactor": 0.75,
        }}],
        "buffers": [{"byteLength": len(binary)}],
        "bufferViews": [{"buffer": 0, "byteOffset": offset, "byteLength": len(piece), "target": 34963 if i == 2 else 34962}
                        for i, (offset, piece) in enumerate(zip(offsets, pieces))],
        "accessors": [
            {"bufferView": 0, "componentType": 5126, "count": len(positions), "type": "VEC3",
             "min": positions.min(0).tolist(), "max": positions.max(0).tolist()},
            {"bufferView": 1, "componentType": 5126, "count": len(positions), "type": "VEC3"},
            {"bufferView": 2, "componentType": 5125, "count": int(indices.size), "type": "SCALAR"},
        ],
    }
    description = json.dumps(document, separators=(",", ":"), allow_nan=False).encode()
    description += b" " * ((-len(description)) % 4)
    binary += b"\0" * ((-len(binary)) % 4)
    total = 12 + 8 + len(description) + 8 + len(binary)
    Path(path).write_bytes(struct.pack("<III", 0x46546C67, 2, total)
                          + struct.pack("<II", len(description), 0x4E4F534A) + description
                          + struct.pack("<II", len(binary), 0x004E4942) + binary)


def read_request(path: str | Path) -> dict:
    request = json.loads(Path(path).read_text())
    if not isinstance(request, dict) or not isinstance(request.get("config"), dict):
        raise ValueError("request must contain the complete config object")
    geometry = request["config"].get("geometry")
    if not isinstance(geometry, dict):
        raise ValueError("config.geometry is required")
    for name in ("vertex_checkpoint", "topology_checkpoint", "vae_checkpoint"):
        spec = geometry.get(name, {})
        if not isinstance(spec, dict) or not spec.get("path") or len(spec.get("sha256", "")) != 64:
            raise ValueError(f"config.geometry.{name} needs path and full sha256")
        if spec.get("weights") != "ema":
            raise ValueError(f"the released geometry profile uses {name} EMA weights")
        if isinstance(spec.get("step"), bool) or not isinstance(spec.get("step"), int):
            raise ValueError(f"config.geometry.{name}.step must be an integer")
    for name in ("image", "output_dir"):
        if not isinstance(request.get(name), str) or not Path(request[name]).is_absolute():
            raise ValueError(f"request.{name} must be an absolute path")
    if not Path(request["image"]).is_file():
        raise FileNotFoundError(request["image"])
    for key, default, maximum in (("face_budget", 8000, 52000), ("steps", 20, 100)):
        value = request.get(key, default)
        if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= maximum:
            raise ValueError(f"request.{key} must be an integer in [1, {maximum}]")
        request[key] = value
    seed = request.get("seed", DEFAULT_SEED)
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < (1 << 63):
        raise ValueError("seed must be an integer in [0, 2**63)")
    request["seed"] = seed
    Path(request["output_dir"]).mkdir(parents=True, exist_ok=True)
    return request


def activate_namespace(stage: str) -> None:
    """The two frozen MeshAxiom versions are intentionally never imported in one process."""
    if any(name == "meshaxiom_core" or name.startswith("meshaxiom_core.") for name in sys.modules):
        raise RuntimeError("geometry stages require separate, clean worker processes")
    for key in list(os.environ):
        if key.startswith(("MESHAXIOM_", "NCCL_", "TORCHELASTIC_")) or key in (
            "RANK", "WORLD_SIZE", "LOCAL_RANK", "MASTER_ADDR", "MASTER_PORT"
        ):
            os.environ.pop(key, None)
    root = Path(__file__).resolve().parents[1] / stage
    if stage not in ("vertex", "topology") or not (root / "meshaxiom_core").is_dir():
        raise ValueError("unknown geometry stage")
    sys.path.insert(0, str(root))


def checkpoint(spec: dict, kind: str) -> tuple[dict, dict]:
    import torch
    path = Path(spec["path"]).resolve()
    artifact = binding(path)
    if artifact["sha256"] != spec["sha256"].lower():
        raise ValueError(f"{kind} checkpoint SHA256 does not match the configured profile")
    ck = torch.load(path, map_location="cpu", mmap=True, weights_only=True)
    if not isinstance(ck, dict) or not ck.get("ema") or not isinstance(ck.get("cfg"), dict):
        raise ValueError(f"{kind} needs a MeshAxiom checkpoint with EMA and cfg")
    if ck.get("step") != spec["step"]:
        raise ValueError(f"{kind} actual step {ck.get('step')} != expected {spec['step']}")
    artifact.update(step=int(ck["step"]), weights="ema", source_tree_sha256=ck.get("source_tree_sha256"))
    return ck, artifact


def image_pixels(cfg, request: dict):
    import torch
    from PIL import Image
    from transformers import AutoImageProcessor
    model_id = request["config"]["geometry"].get("image_model")
    if model_id:
        cfg.set_path("condition.model_id", model_id)
    if cfg.get_path("data.image_background") != [0, 0, 0]:
        raise ValueError("released geometry checkpoints require a black image background")
    if cfg.get_path("condition.image_size") != 512 or cfg.get_path("condition.camera_token", False):
        raise ValueError("released geometry profile requires 512 image input and no camera token")
    processor = AutoImageProcessor.from_pretrained(cfg.get_path("condition.model_id"))
    with Image.open(request["image"]) as opened:
        if opened.size != (512, 512):
            raise ValueError("worker expects a preprocessed 512x512 image; use the public preprocessing entrypoint")
        rgba = opened.convert("RGBA")
        rgb = Image.alpha_composite(Image.new("RGBA", rgba.size, (0, 0, 0, 255)), rgba).convert("RGB")
    pixels = processor(images=rgb, do_resize=True, size={"height": 512, "width": 512}, return_tensors="pt")["pixel_values"]
    audit = {
        "shape": list(pixels.shape),
        "tensor_sha256": hashlib.sha256(pixels.float().contiguous().numpy().tobytes()).hexdigest(),
        "processor_model": cfg.get_path("condition.model_id"),
        "input": binding(request["image"]),
        "policy": "same_512_rgb_black_no_recrop",
    }
    return pixels, audit


def cuda_device(request: dict):
    import torch
    torch.set_num_threads(2)
    dev = torch.device(request.get("device", "cuda:0"))
    if dev.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("full geometry inference requires a CUDA device")
    fraction = request["config"].get("runtime", {}).get("gpu_memory_fraction")
    if fraction is not None:
        if not 0 < float(fraction) <= 1:
            raise ValueError("gpu_memory_fraction must be in (0,1]")
        torch.cuda.set_per_process_memory_fraction(float(fraction), dev)
    return dev


def stage_cli(main) -> None:
    import argparse
    parser = argparse.ArgumentParser(description=main.__doc__)
    parser.add_argument("--request", required=True, help="Path to a JSON worker request")
    args = parser.parse_args()
    request = read_request(args.request)
    start = time.monotonic()
    try:
        report = main(request)
        print(json.dumps({"status": "COMPLETE", "seconds": round(time.monotonic() - start, 3), "report": report}), flush=True)
    except (Exception, SystemExit) as exc:
        write_json(Path(request["output_dir"]) / "WORKER_FAILED.json", {
            "status": "FAILED", "worker": main.__module__, "exception": type(exc).__name__, "message": str(exc),
            "seconds": round(time.monotonic() - start, 3)
        })
        raise
