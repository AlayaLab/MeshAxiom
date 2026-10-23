"""Read-only Hugging Face model loading with the released EMA hashes enforced."""
from pathlib import Path
import json
import re

from .contracts import MeshAxiomError

EXPECTED_WEIGHTS = {
    "vertex_checkpoint": {"file": "vertex-ema.pt", "step": 100000, "weights": "ema", "size": 9421814821,
                          "sha256": "7743cc03ad05abcadab204b8a00f7d312ad2b26d7de1eb64ba009a75c12054f2"},
    "topology_checkpoint": {"file": "topology-ema.pt", "step": 10000, "weights": "ema", "size": 9423507316,
                            "sha256": "7fb0c7cc2035616f811326d3e64c1393de39d8f652ab7aa122ab84a987d2a19f"},
    "vae_checkpoint": {"file": "vae-ema.pt", "step": 30000, "weights": "ema", "size": 446157647,
                       "sha256": "7d6833425af71e923756d2ebdb2863e704857bffb8b365ab0c2f5ec3bfb6f4bd"},
}

DOWNLOAD_FILES = ["weights/" + row["file"] for row in EXPECTED_WEIGHTS.values()] + [
    "config.json", "meshaxiom.json", "model_index.json", "SHA256SUMS", "README.md", "LICENSE.md", "WEIGHTS.json", "VALIDATION.json",
    "external/dinov3/config.json", "external/dinov3/preprocessor_config.json", "external/dinov3/LICENSE.md",
]


def _read_model_configuration(directory):
    canonical = directory / "config.json"
    legacy = directory / "meshaxiom.json"
    try:
        canonical.lstat()
    except FileNotFoundError:
        configuration = legacy
    except OSError as exc:
        raise MeshAxiomError("Cannot inspect the downloaded model configuration") from exc
    else:
        configuration = canonical
    try:
        declared = json.loads(configuration.read_text())
    except (OSError, ValueError) as exc:
        raise MeshAxiomError("Cannot read the downloaded model configuration") from exc
    return configuration, declared


def model_config(directory, *, device="cuda:0"):
    directory = Path(directory).expanduser().resolve()
    from .config import load_config
    configuration, declared = _read_model_configuration(directory)
    declared_geometry = declared.get("geometry", {}) if isinstance(declared, dict) else {}
    if not isinstance(declared_geometry, dict):
        raise MeshAxiomError("Model geometry configuration must be an object")
    # Validate the declared snapshot entries before following normal Hub cache
    # symlinks. The resulting blob files legitimately live outside snapshots/.
    for key, expected in EXPECTED_WEIGHTS.items():
        row = declared_geometry.get(key, {})
        _require_snapshot_entry(row.get("path") if isinstance(row, dict) else None,
                                Path("weights") / expected["file"])
    _require_snapshot_entry(declared_geometry.get("image_model"), Path("external/dinov3"))
    config = load_config(configuration)
    geometry = config.get("geometry", {})
    if geometry.get("implementation") != "exact":
        raise MeshAxiomError("This release supports only the Exact implementation")
    for key, expected in EXPECTED_WEIGHTS.items():
        row = geometry.get(key, {})
        if any(row.get(name) != expected[name] for name in ["sha256", "step", "weights"]):
            raise MeshAxiomError("Downloaded model is not the released Base EMA recipe: " + key)
        if Path(row.get("path", "")).resolve() != (directory / "weights" / expected["file"]).resolve():
            raise MeshAxiomError("Model checkpoint must stay inside its downloaded snapshot")
    if geometry.get("topology_seed") != 1574365204:
        raise MeshAxiomError("The released topology seed must be 1574365204")
    if Path(geometry.get("image_model", "")).resolve() != (directory / "external/dinov3").resolve():
        raise MeshAxiomError("Image processor must use the local released DINOv3 configuration")
    import os
    import tempfile
    config["runtime"]["gpu_lock_directory"] = str(Path(os.environ.get("MESHAXIOM_LOCK_DIRECTORY", Path(tempfile.gettempdir()) / "meshaxiom-gpu-locks")))
    config["runtime"]["device"] = device
    return config


def _require_snapshot_entry(value, expected):
    if (not isinstance(value, str) or Path(value).is_absolute()
            or ".." in Path(value).parts or Path(value) != expected):
        raise MeshAxiomError("Model data must use its released snapshot entry: " + str(expected))


def download_model(repo_id, *, revision=None, cache_dir=None, local_files_only=False, token=None):
    """Download model data only. No remote Python or trust_remote_code is used."""
    if not isinstance(repo_id, str) or not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", repo_id):
        raise MeshAxiomError("Pass the actual published Hugging Face organization/model ID")
    if revision is None and not local_files_only:
        raise MeshAxiomError("Pass an immutable full Hugging Face commit SHA as revision")
    if revision is not None and not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise MeshAxiomError("revision must be a full 40-character Hugging Face commit SHA")
    from huggingface_hub import snapshot_download
    return Path(snapshot_download(repo_id=repo_id, revision=revision, cache_dir=cache_dir,
                                  local_files_only=local_files_only, token=token, allow_patterns=DOWNLOAD_FILES))
