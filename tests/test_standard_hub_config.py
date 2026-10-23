"""Standard Hub configuration, with no network, real weights or GPU execution."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import types

import pytest

from meshaxiom.config import load_config
from meshaxiom.contracts import MeshAxiomError
from meshaxiom.hub import download_model, model_config
from test_hub_layout_regressions import make_snapshot

ROOT = Path(__file__).resolve().parents[1]


def standard_snapshot(tmp_path, *, layout="regular", legacy=True):
    directory, snapshot, config = make_snapshot(tmp_path, layout)
    canonical = snapshot / "config.json"
    data = (snapshot / "meshaxiom.json").read_bytes()
    if layout == "all-symlinks":
        blob = snapshot.parents[1] / "blobs" / hashlib.sha256(data).hexdigest()
        # make_snapshot has already installed this same immutable JSON blob.
        assert blob.read_bytes() == data
        canonical.symlink_to(os.path.relpath(blob, snapshot))
    else:
        canonical.write_bytes(data)
    if not legacy:
        (snapshot / "meshaxiom.json").unlink()
    return directory, snapshot, config


@pytest.mark.parametrize("layout", ["regular", "all-symlinks"])
@pytest.mark.parametrize("legacy", [True, False])
def test_standard_config_is_actually_read_in_real_cache_layouts(tmp_path, monkeypatch, layout, legacy):
    directory, snapshot, _ = standard_snapshot(tmp_path, layout=layout, legacy=legacy)
    paths = []
    def record(path):
        paths.append(path)
        return load_config(path)
    monkeypatch.setattr("meshaxiom.config.load_config", record)
    config = model_config(directory)
    assert paths == [snapshot / "config.json"]
    assert Path(config["geometry"]["image_model"]).resolve() == snapshot / "external/dinov3"
    assert Path(config["geometry"]["vertex_checkpoint"]["path"]).samefile(snapshot / "weights/vertex-ema.pt")
    assert not Path(config["runtime"]["gpu_lock_directory"]).is_relative_to(snapshot)


def test_missing_standard_config_falls_back_to_legacy_snapshot(tmp_path, monkeypatch):
    directory, snapshot, _ = make_snapshot(tmp_path, "all-symlinks")
    paths = []
    def record(path):
        paths.append(path)
        return load_config(path)
    monkeypatch.setattr("meshaxiom.config.load_config", record)
    model_config(directory)
    assert paths == [snapshot / "meshaxiom.json"]


@pytest.mark.parametrize("bad", ["invalid-json", "directory", "dangling-symlink", "unreadable", "invalid-schema"])
def test_present_bad_standard_config_never_falls_back(tmp_path, monkeypatch, bad):
    directory, snapshot, _ = make_snapshot(tmp_path, "regular")
    canonical = snapshot / "config.json"
    if bad == "invalid-json":
        canonical.write_text("{")
    elif bad == "directory":
        canonical.mkdir()
    elif bad == "dangling-symlink":
        canonical.symlink_to("missing-config-blob")
        assert not canonical.exists() and canonical.is_symlink()
    elif bad == "invalid-schema":
        canonical.write_text('{"schema_version":0}')
    else:
        canonical.write_bytes((snapshot / "meshaxiom.json").read_bytes())
        read_text = Path.read_text
        def unreadable(path, *args, **kwargs):
            if path == canonical:
                raise PermissionError("synthetic read denial even when tests run as root")
            return read_text(path, *args, **kwargs)
        monkeypatch.setattr(Path, "read_text", unreadable)
    with pytest.raises(MeshAxiomError):
        model_config(directory)


@pytest.mark.parametrize("legacy_state", ["different", "invalid-json", "unreadable"])
def test_standard_config_is_authoritative_without_reading_legacy(tmp_path, monkeypatch, legacy_state):
    directory, snapshot, config = standard_snapshot(tmp_path)
    legacy = snapshot / "meshaxiom.json"
    if legacy_state == "different":
        config["runtime"]["timeout_seconds"] = 61
        legacy.write_text(json.dumps(config))
    elif legacy_state == "invalid-json":
        legacy.write_text("{")
    read_text = Path.read_text
    def no_legacy_read(path, *args, **kwargs):
        if path == legacy:
            raise PermissionError("canonical loading must not read the compatibility alias")
        return read_text(path, *args, **kwargs)
    monkeypatch.setattr(Path, "read_text", no_legacy_read)
    assert model_config(directory)["runtime"]["timeout_seconds"] == 30


@pytest.mark.parametrize("change", ["outside-checkpoint", "outside-processor", "wrong-EMA-hash", "wrong-EMA-step", "wrong-seed"])
def test_standard_config_preserves_snapshot_and_recipe_guards(tmp_path, change):
    directory, snapshot, config = standard_snapshot(tmp_path, legacy=False)
    geometry = config["geometry"]
    if change == "outside-checkpoint":
        geometry["vertex_checkpoint"]["path"] = "../outside.pt"
    elif change == "outside-processor":
        geometry["image_model"] = "../outside-processor"
    elif change == "wrong-EMA-hash":
        geometry["vertex_checkpoint"]["sha256"] = "0" * 64
    elif change == "wrong-EMA-step":
        geometry["vertex_checkpoint"]["step"] = 180000
    else:
        geometry["topology_seed"] = 0
    (snapshot / "config.json").write_text(json.dumps(config))
    with pytest.raises(MeshAxiomError):
        model_config(directory)


def test_normal_pinned_download_includes_both_config_names_without_extra_requests(tmp_path, monkeypatch):
    calls = []
    def snapshot_download(**kwargs):
        calls.append(kwargs)
        return str(tmp_path)
    monkeypatch.setitem(sys.modules, "huggingface_hub", types.SimpleNamespace(snapshot_download=snapshot_download))
    revision = "a" * 40
    assert download_model("AlayaLab/MeshAxiom-Base", revision=revision, local_files_only=True) == tmp_path
    assert len(calls) == 1
    assert {"config.json", "meshaxiom.json"}.issubset(calls[0]["allow_patterns"])
    assert calls[0]["revision"] == revision
    assert calls[0]["local_files_only"] is True
    for bad_revision in [None, "main"]:
        with pytest.raises(MeshAxiomError):
            download_model("AlayaLab/MeshAxiom-Base", revision=bad_revision)
    assert len(calls) == 1


def test_preparation_synchronizes_final_hashes_configs_and_checksums(tmp_path, monkeypatch):
    # Exercise the real staging control flow using tiny JSON stand-ins. No torch
    # tensors, real checkpoint bytes, deserialization or inference are involved.
    spec = importlib.util.spec_from_file_location("prepare_model_for_config_test", ROOT / "scripts/prepare_model.py")
    prepare_model = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(prepare_model)
    weights = tmp_path / "tiny-synthetic-inputs"
    weights.mkdir()
    hashes = {}
    for kind, step in prepare_model.STEPS.items():
        data = {"kind": kind, "step": step, "weights": "ema", "ema": {"placeholder": 1}}
        if kind == "topology":
            data["v113"] = {"version": "meshaxiom-topology-richknn-v118"}
        path = weights / (kind + "-ema.pt")
        path.write_text(json.dumps(data))
        hashes[kind] = hashlib.sha256(path.read_bytes()).hexdigest()
    fake_torch = types.SimpleNamespace(set_num_threads=lambda n: None, is_tensor=lambda value: False,
                                       load=lambda path, **kwargs: json.loads(Path(path).read_text()))
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setattr(prepare_model, "SOURCE_SHA", hashes)
    output = tmp_path / "staged-model"
    prepare_model.prepare(weights, output)
    assert (output / "config.json").read_bytes() == (output / "meshaxiom.json").read_bytes()
    config = json.loads((output / "config.json").read_text())
    for kind, digest in hashes.items():
        assert config["geometry"][kind + "_checkpoint"]["sha256"] == digest
    assert json.loads((output / "model_index.json").read_text())["configuration"] == "config.json"
    sums = dict(line.split("  ", 1)[::-1] for line in (output / "SHA256SUMS").read_text().splitlines())
    assert {"config.json", "meshaxiom.json"}.issubset(sums)
    for name, digest in sums.items():
        assert hashlib.sha256((output / name).read_bytes()).hexdigest() == digest


def test_published_metadata_template_contains_the_identical_real_configuration():
    template = ROOT / "deploy/huggingface/model"
    assert (template / "config.json").read_bytes() == (template / "meshaxiom.json").read_bytes()
    assert json.loads((template / "model_index.json").read_text())["configuration"] == "config.json"
