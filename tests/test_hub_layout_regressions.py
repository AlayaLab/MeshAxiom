"""CPU regressions for installed Hub layouts and clean stage worker startup."""
import hashlib
import json
import os
from pathlib import Path
import sys

import pytest

from meshaxiom import MeshAxiomPipeline
from meshaxiom.config import load_config
from meshaxiom.contracts import MeshAxiomError
from meshaxiom.hub import EXPECTED_WEIGHTS, model_config
from meshaxiom.session import ResidentWorker


def make_snapshot(tmp_path, layout):
    """Model-like data, never usable for inference; no checkpoint load is run."""
    cache = tmp_path / "cache"
    snapshot = cache / "snapshots" / ("a" * 40)
    blobs = cache / "blobs"
    snapshot.mkdir(parents=True)
    blobs.mkdir()
    geometry = {
        key: {"path": "weights/" + spec["file"],
              **{field: spec[field] for field in ("sha256", "step", "weights")}}
        for key, spec in EXPECTED_WEIGHTS.items()
    }
    geometry.update(image_model="external/dinov3", topology_seed=1574365204,
                    implementation="exact")
    config = {"schema_version": 1, "geometry": geometry,
              "runtime": {"vertex_python": "python", "topology_python": "python",
                          "device": "cuda:0", "timeout_seconds": 30}}
    files = {"meshaxiom.json": json.dumps(config).encode(),
             "external/dinov3/config.json": b'{"model_type":"dinov3_vit"}',
             "external/dinov3/preprocessor_config.json": b'{}',
             "external/dinov3/LICENSE.md": b"processor license placeholder"}
    files.update({"weights/" + spec["file"]: ("not real model data:" + key).encode()
                  for key, spec in EXPECTED_WEIGHTS.items()})
    for relative, data in files.items():
        path = snapshot / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        if layout == "all-symlinks" or layout == "weights-symlinks" and relative.startswith("weights/"):
            blob = blobs / hashlib.sha256(data).hexdigest()
            blob.write_bytes(data)
            path.symlink_to(os.path.relpath(blob, path.parent))
        else:
            path.write_bytes(data)
    if layout == "directory-symlink":
        alias = tmp_path / "model-alias"
        alias.symlink_to(snapshot, target_is_directory=True)
        return alias, snapshot, config
    return snapshot, snapshot, config


def inventory(path):
    return {str(p.relative_to(path)): (p.is_symlink(), os.readlink(p) if p.is_symlink() else None,
                                      p.stat().st_size, p.stat().st_mtime_ns)
            for p in path.rglob("*") if p.is_file()}


@pytest.mark.parametrize("layout", ["regular", "weights-symlinks", "all-symlinks", "directory-symlink"])
def test_model_directory_and_from_pretrained_accept_real_cache_layouts(tmp_path, monkeypatch, layout):
    directory, snapshot, _ = make_snapshot(tmp_path, layout)
    unrelated = tmp_path / "unrelated-working-directory"
    unrelated.mkdir()
    monkeypatch.chdir(unrelated)
    before = inventory(snapshot)
    config = model_config(directory)
    for key, spec in EXPECTED_WEIGHTS.items():
        assert Path(config["geometry"][key]["path"]).samefile(snapshot / "weights" / spec["file"])
    assert Path(config["geometry"]["image_model"]).resolve() == (snapshot / "external/dinov3").resolve()
    assert not Path(config["runtime"]["gpu_lock_directory"]).is_relative_to(snapshot)
    with MeshAxiomPipeline.from_pretrained(directory) as pipe:
        assert pipe.config["geometry"]["implementation"] == "exact"
    assert inventory(snapshot) == before


@pytest.mark.parametrize("layout", ["regular", "all-symlinks"])
def test_config_file_symlink_preserves_snapshot_parent_for_cli(tmp_path, monkeypatch, layout):
    _, snapshot, _ = make_snapshot(tmp_path, layout)
    monkeypatch.chdir(tmp_path)
    config = load_config(snapshot / "meshaxiom.json")
    assert Path(config["geometry"]["image_model"]).resolve() == (snapshot / "external/dinov3").resolve()
    for key, spec in EXPECTED_WEIGHTS.items():
        assert Path(config["geometry"][key]["path"]).samefile(snapshot / "weights" / spec["file"])


def test_remote_api_uses_the_returned_symlink_snapshot(tmp_path, monkeypatch):
    _, snapshot, _ = make_snapshot(tmp_path, "all-symlinks")
    calls = []
    def download(repo, **kwargs):
        calls.append((repo, kwargs))
        return snapshot
    monkeypatch.setattr("meshaxiom.hub.download_model", download)
    with MeshAxiomPipeline.from_pretrained("AlayaLab/MeshAxiom-Base", revision="a" * 40) as pipe:
        assert Path(pipe.config["geometry"]["image_model"]).is_dir()
    assert calls[0][0] == "AlayaLab/MeshAxiom-Base"
    assert calls[0][1]["revision"] == "a" * 40


@pytest.mark.parametrize("field,redirect", [
    ("vertex_checkpoint", "../outside.pt"),
    ("vertex_checkpoint", "/tmp/outside.pt"),
    ("topology_checkpoint", "../outside.pt"),
    ("vae_checkpoint", "../outside.pt"),
    ("image_model", "../other-processor"),
    ("image_model", "/tmp/other-processor"),
])
def test_model_config_rejects_declared_path_redirects(tmp_path, field, redirect):
    _, snapshot, config = make_snapshot(tmp_path, "regular")
    if field == "image_model":
        config["geometry"][field] = redirect
    else:
        config["geometry"][field]["path"] = redirect
    (snapshot / "meshaxiom.json").write_text(json.dumps(config))
    with pytest.raises(MeshAxiomError):
        model_config(snapshot)


def test_model_workers_default_to_active_python_when_path_differs(tmp_path, monkeypatch):
    _, snapshot, _ = make_snapshot(tmp_path, "regular")
    wrong_bin = tmp_path / "different-python-bin"
    wrong_bin.mkdir()
    wrong_python = wrong_bin / "python"
    wrong_python.write_text("#!/bin/sh\nexit 91\n")
    wrong_python.chmod(0o755)
    monkeypatch.setenv("PATH", str(wrong_bin))
    config = model_config(snapshot)
    for name in ("vertex_python", "topology_python"):
        assert Path(config["runtime"][name]).samefile(sys.executable)


@pytest.mark.parametrize("stage,reference", [("vertex", False), ("vertex", True),
                                               ("topology", False), ("topology", True)])
def test_clean_resident_worker_enters_model_loader_before_stage_recipe_import(tmp_path, monkeypatch, stage, reference):
    # Private child-only hook stops before loading weights or probing CUDA. Calling
    # the real worker exercises startup and namespace ordering in a fresh process.
    marker = "CPU_IMPORT_ORDER_MODEL_LOADER_REACHED"
    hook = tmp_path / "child-hook"
    hook.mkdir()
    (hook / "sitecustomize.py").write_text(
        "from meshaxiom.workers import _geometry_state\n"
        "from meshaxiom.vendor.recipe.runtime import activate_namespace\n"
        "def cpu_stop(request, stage, *, exact=True):\n"
        "    activate_namespace(stage)\n"
        f"    raise RuntimeError({marker!r})\n"
        "_geometry_state.load = cpu_stop\n")
    monkeypatch.setenv("PYTHONPATH", str(hook) + os.pathsep + os.environ.get("PYTHONPATH", ""))
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    image = tmp_path / "input.png"
    image.write_bytes(b"stopped before image decoding")
    geometry = {key: {"path": str(tmp_path / spec["file"]),
                      **{field: spec[field] for field in ("sha256", "step", "weights")}}
                for key, spec in EXPECTED_WEIGHTS.items()}
    request = tmp_path / "request.json"
    request.write_text(json.dumps({"schema_version": 1,
                                  "config": {"schema_version": 1, "geometry": geometry},
                                  "image": str(image), "output_dir": str(tmp_path), "device": "cuda:0"}))
    worker = ResidentWorker(stage, {stage + "_python": sys.executable,
                                   "scratch_directory": str(tmp_path), "timeout_seconds": 60}, reference=reference)
    try:
        with pytest.raises(MeshAxiomError, match=marker):
            worker.run(request, tmp_path)
        assert marker in (tmp_path / (stage + ".log")).read_text()
    finally:
        worker.close()
    assert worker.process is None
    assert worker._reply is None


def test_failed_cache_preparation_does_not_leak_file_descriptors(tmp_path):
    descriptors = Path('/proc/self/fd')
    if not descriptors.is_dir():
        pytest.skip('Linux process descriptor inventory is required')
    blocked = tmp_path / 'cache-is-a-file'
    blocked.write_text('not a directory')
    before = len(list(descriptors.iterdir()))
    for _ in range(8):
        worker = ResidentWorker('vertex', {'vertex_python': sys.executable,
                                           'cache_directory': str(blocked)})
        try:
            with pytest.raises(OSError):
                worker.run(tmp_path / 'unused-request.json', tmp_path)
        finally:
            worker.close()
        assert worker.process is None
        assert worker._reply is None
    assert len(list(descriptors.iterdir())) == before
