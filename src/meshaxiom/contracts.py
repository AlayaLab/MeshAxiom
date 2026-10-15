"""Serializable requests and results shared by the CLI, Python API and agents."""
from pathlib import Path
import hashlib
import json
import math
import os
from datetime import datetime, timezone

class MeshAxiomError(RuntimeError):
    pass

def utc_now():
    return datetime.now(timezone.utc).isoformat()

def file_binding(path):
    p = Path(path).resolve()
    with p.open("rb") as handle:
        digest = hashlib.file_digest(handle, "sha256").hexdigest() if hasattr(hashlib, "file_digest") else _stream_sha(handle)
    return {"path": str(p), "size": p.stat().st_size, "sha256": digest}

def _stream_sha(handle):
    digest = hashlib.sha256()
    for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
        digest.update(chunk)
    return digest.hexdigest()

def write_json(path, value):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    temporary = p.with_name(p.name + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    os.replace(temporary, p)

def integer(value, name, minimum, maximum):
    if isinstance(value, bool) or not isinstance(value, int) or not minimum <= value <= maximum:
        raise MeshAxiomError(f"{name} must be an integer between {minimum} and {maximum}")
    return value

def validate_options(operation, *, seed, face_budget=None, texture_size=None, background="auto", camera=None):
    if not isinstance(operation, str) or operation not in {"geometry", "material", "superresolution"}:
        raise MeshAxiomError("operation must be geometry, material or superresolution")
    integer(seed, "seed", 0, 2**63 - 1)
    if not isinstance(background, str) or background not in {"auto", "alpha", "keep", "remove"}:
        raise MeshAxiomError("background must be auto, alpha, keep or remove")
    if operation == "geometry":
        integer(face_budget, "face_budget", 100, 30000)
    elif operation == "material":
        integer(texture_size, "texture_size", 256, 4096)
        if texture_size & (texture_size - 1):
            raise MeshAxiomError("texture_size must be a power of two")
        if camera is not None:
            if not isinstance(camera, dict):
                raise MeshAxiomError("camera must be an object")
            allowed = {"camera_angle_x", "distance", "azimuth", "elevation", "offset_x", "offset_y", "up_axis"}
            if set(camera) - allowed:
                raise MeshAxiomError("Unknown camera field: " + ", ".join(sorted(set(camera) - allowed)))
            for name, value in camera.items():
                if name == "up_axis":
                    if not isinstance(value, str) or value not in {"y", "z", "Y", "Z", "y_up", "z_up"}:
                        raise MeshAxiomError("camera.up_axis must be y or z")
                elif isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                    raise MeshAxiomError("camera values must be finite numbers")
            if "distance" in camera and camera["distance"] <= 0:
                raise MeshAxiomError("camera.distance must be positive")

