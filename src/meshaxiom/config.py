"""Explicit checkpoint paths; no workstation paths are built into the package."""
import copy
import json
import math
import os
import re
import shutil
import sys
import subprocess
import tempfile
from pathlib import Path
from .contracts import MeshAxiomError, file_binding

_VERIFIED = {}
_GEOMETRY_STEPS = {"vertex_checkpoint": 180000, "topology_checkpoint": 10000, "vae_checkpoint": 30000}

def load_config(value):
    if isinstance(value, dict):
        config = copy.deepcopy(value)
        root = Path.cwd()
    elif isinstance(value, (str, os.PathLike)):
        p = Path(value).expanduser().resolve()
        try:
            config = json.loads(p.read_text())
        except (OSError, ValueError) as exc:
            raise MeshAxiomError("Cannot read configuration: " + str(p)) from exc
        root = p.parent
    else:
        raise MeshAxiomError("Configuration must be a JSON object or a path")
    if not isinstance(config, dict):
        raise MeshAxiomError("Configuration must be a JSON object")
    if config.get("schema_version") != 1:
        raise MeshAxiomError("Configuration schema_version must be 1")
    runtime = config.setdefault("runtime", {})
    for group in ["runtime", "geometry", "material"]:
        if group in config and not isinstance(config[group], dict):
            raise MeshAxiomError(group + " configuration must be an object")
    for name in ("vertex_python", "topology_python", "material_python", "superresolution_python"):
        value = runtime.get(name, runtime.get("material_python", sys.executable) if name == "superresolution_python" else sys.executable)
        if not isinstance(value, str):
            raise MeshAxiomError(name + " must be a Python interpreter path")
        # Preserve the executable's symlink: resolving a venv's Python target
        # would discard its environment and load the base interpreter instead.
        resolved = os.path.abspath(os.path.join(root, os.path.expandvars(os.path.expanduser(value)))) if "/" in value or value.startswith(("~", ".")) else shutil.which(value)
        # Keep the invalid path for doctor to explain; execution rejects it.
        runtime[name] = resolved or value
    runtime.setdefault("device", "cuda:0")
    runtime.setdefault("timeout_seconds", 3600)
    runtime.setdefault("gpu_lock_directory", os.environ.get("MESHAXIOM_LOCK_DIRECTORY", str(Path(tempfile.gettempdir()) / "meshaxiom-gpu-locks")))
    runtime.setdefault("gpu_lock_pattern", "cuda-{index}.lock")
    for name in ["gpu_lock_directory", "gpu_lock_pattern", "scratch_directory"]:
        if name in runtime and (not isinstance(runtime[name], str) or not runtime[name]):
            raise MeshAxiomError("runtime." + name + " must be a nonempty path or filename")
    if not isinstance(runtime["device"], str) or not re.fullmatch(r"cuda:\d+", runtime["device"]):
        raise MeshAxiomError("runtime.device must be a CUDA device such as cuda:0")
    for name in ["timeout_seconds", "lock_timeout_seconds", "minimum_free_disk_gib", "minimum_free_workspace_gib"]:
        value = runtime.get(name, 0)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0 or name == "timeout_seconds" and value == 0:
            raise MeshAxiomError("Invalid runtime." + name)
    for group in [config.get("geometry", {}), config.get("material", {}), runtime]:
        for key, item in list(group.items()):
            if isinstance(item, dict) and "path" in item:
                if not isinstance(item["path"], str):
                    raise MeshAxiomError("Checkpoint path must be a string")
                item["path"] = _path(item["path"], root)
            elif key.endswith(("_dir", "_repo", "_checkpoint", "_directory")) and isinstance(item, str):
                group[key] = _path(item, root)
    image_model = config.get("geometry", {}).get("image_model")
    if image_model is not None and (not isinstance(image_model, str) or not image_model):
        raise MeshAxiomError("geometry.image_model must be a nonempty model ID or directory")
    if image_model and (image_model.startswith((".", "/", "~")) or os.path.exists(image_model)):
        config["geometry"]["image_model"] = _path(image_model, root)
    geometry = config.get("geometry", {})
    if geometry:
        geometry.setdefault("implementation", "exact")
        if geometry["implementation"] != "exact":
            raise MeshAxiomError("Only the Exact implementation is supported in this release")
        geometry.setdefault("topology_seed", 1574365204)
    return config

def _path(value, root):
    p = Path(os.path.expandvars(value)).expanduser()
    return str((p if p.is_absolute() else root / p).resolve())

def check_assets(config, operation, *, verify_hashes=True):
    checks = []
    if operation == "geometry":
        group = config.get("geometry", {})
        for name, step in _GEOMETRY_STEPS.items():
            row = group.get(name)
            if not isinstance(row, dict) or row.get("step") != step or row.get("weights") != "ema":
                raise MeshAxiomError(f"{name} must declare step {step} and EMA weights")
            p = Path(row.get("path", ""))
            if not p.is_file():
                raise MeshAxiomError(f"Missing {name}: {p}; configure downloaded weights first")
            expected = row.get("sha256", "")
            if not re.fullmatch(r"[0-9a-f]{64}", expected):
                raise MeshAxiomError(f"{name} requires a SHA256 binding")
            if verify_hashes:
                stat = p.stat()
                identity = (str(p.resolve()), stat.st_size, stat.st_mtime_ns, stat.st_ino, expected)
                if identity not in _VERIFIED:
                    actual = file_binding(p)
                    if actual["sha256"] != expected:
                        raise MeshAxiomError(f"SHA256 mismatch for {name}")
                    _VERIFIED[identity] = True
            checks.append({"asset": name, "path": str(p), "step": step, "weights": "ema", "sha256": expected, "hash_verified": verify_hashes})
        image_model = Path(group.get("image_model", ""))
        for name, expected in {
            "config.json": "135ecd23e34a70b6fbed8b083fdecb319b7e3a54e3d849258bbe4ddcf1783bb5",
            "preprocessor_config.json": "960c41d1f3a7778b936365769a2d90550b318a6c0a53a0296957adacfe5e0dd7",
        }.items():
            path = image_model / name
            if not path.is_file():
                raise MeshAxiomError("Missing local DINOv3 configuration: " + str(path))
            actual = file_binding(path)
            if actual["sha256"] != expected:
                raise MeshAxiomError("DINOv3 configuration SHA256 mismatch: " + name)
            checks.append({"asset": "dinov3/" + name, **actual, "hash_verified": True})
    elif operation == "superresolution":
        row = config.get("material", {}).get("superresolution_checkpoint", {})
        p = Path(row.get("path", ""))
        if not p.is_file() or not re.fullmatch(r"[0-9a-f]{64}", row.get("sha256", "")):
            raise MeshAxiomError("Configure material.superresolution_checkpoint with local RealESRGAN_x2plus weights and SHA256")
        if verify_hashes and file_binding(p)["sha256"] != row["sha256"]:
            raise MeshAxiomError("Super-resolution checkpoint SHA256 mismatch")
        checks.append({"asset": "RealESRGAN_x2plus", "path": str(p), "sha256": row["sha256"], "hash_verified": verify_hashes})
    elif operation == "material":
        group = config.get("material", {})
        for name in ("pixal_model_dir", "trellis2_model_dir", "image_encoder_dir", "naf_repo"):
            p = Path(group.get(name, ""))
            if name not in group or not p.is_dir():
                raise MeshAxiomError(f"Missing material.{name}; configure the external material dependencies")
            checks.append({"asset": name, "path": str(p)})
    else:
        raise MeshAxiomError("Unknown operation")
    return checks

def doctor(config, operation="all", *, verify_hashes=False, check_runtime=False):
    report = {"schema_version": 1, "status": "ready", "checks": [], "errors": []}
    for stage in (["geometry", "material"] if operation == "all" else [operation]):
        try:
            report["checks"].extend(check_assets(config, stage, verify_hashes=verify_hashes))
        except MeshAxiomError as exc:
            report["errors"].append(str(exc))
        for name in (["vertex_python", "topology_python"] if stage == "geometry" else [stage + "_python"]):
            interpreter = config["runtime"][name]
            if not (Path(interpreter).is_file() and os.access(interpreter, os.X_OK)):
                report["errors"].append("Python interpreter unavailable: " + str(interpreter))
        if check_runtime:
            imports = "from transformers import AutoImageProcessor, AutoModel; import numpy, scipy, yaml, torch" if stage == "geometry" else "import numpy, scipy, cv2, trimesh, torch, safetensors" if stage == "material" else "import torch, numpy, PIL"
            names = ["vertex_python", "topology_python"] if stage == "geometry" else [stage + "_python"]
            checked = set()
            for name in names:
                interpreter = config["runtime"][name]
                if interpreter in checked:
                    continue
                checked.add(interpreter)
                try:
                    probe = subprocess.run([interpreter, "-c", imports], capture_output=True, text=True, timeout=60)
                    if probe.returncode:
                        lines = probe.stderr.strip().splitlines()
                        reason = lines[-1] if lines else "process exited with code " + str(probe.returncode)
                        report["errors"].append(name + " runtime imports failed: " + reason)
                    else:
                        report["checks"].append({"runtime": stage, "interpreter": interpreter, "imports": "passed", "model_weights_loaded": False})
                except (OSError, subprocess.TimeoutExpired) as exc:
                    report["errors"].append(name + " runtime probe failed: " + str(exc))
    if report["errors"]:
        report["status"] = "setup_required"
    report["models_loaded"] = False
    report["runtime_imports_checked"] = check_runtime
    report["gpu_inference_executed"] = False
    return report

