"""CPU inference-contract checks; no trained weights or CUDA are loaded."""
import hashlib
import importlib
import json
from pathlib import Path
import sys
import types

import numpy as np
import pytest

from meshaxiom.material_backend.superresolution import (
    HALO, WEIGHTS_SHA256, upscale, upscale_alpha, validate_checkpoint, validate_tile, validate_vendor,
)


@pytest.fixture
def local_model():
    torch = pytest.importorskip("torch")

    class KnownFilter(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.anchor = torch.nn.Parameter(torch.zeros(()), requires_grad=False)

        def forward(self, image):
            # A fixed local filter provides an untiled reference for context
            # boundaries. It is intentionally not an SR quality/model test.
            blurred = torch.nn.functional.avg_pool2d(image, kernel_size=3, stride=1, padding=1)
            return torch.nn.functional.interpolate(blurred, scale_factor=2, mode="nearest") + self.anchor

    return KnownFilter()


def test_tiled_output_matches_independent_cpu_reference(local_model):
    import torch
    rgb = np.random.default_rng(83).integers(0, 256, (70, 98, 3), dtype=np.uint8)
    result, audit = upscale(local_model, rgb, tile=32)
    with torch.inference_mode():
        full = local_model(torch.from_numpy(rgb.transpose(2, 0, 1).copy()).float()[None] / 255)
    expected = np.rint(full[0].permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    assert np.array_equal(result, expected)
    assert audit["tiles"] == 12 and audit["context_halo"] == 192
    assert audit["coverage_min"] == audit["coverage_max"] == 1
    assert audit["output_texels_written_once"] == 140 * 196


def test_odd_sizes_pad_then_crop_to_exact_x2(local_model):
    import torch
    rgb = np.random.default_rng(3).integers(0, 256, (35, 33, 3), dtype=np.uint8)
    result, audit = upscale(local_model, rgb, tile=32)
    padded = np.pad(rgb, ((0, 1), (0, 1), (0, 0)), mode="edge")
    with torch.inference_mode():
        full = local_model(torch.from_numpy(padded.transpose(2, 0, 1).copy()).float()[None] / 255)
    expected = np.rint(full[0].permute(1, 2, 0).numpy() * 255).astype(np.uint8)[:70, :66]
    assert np.array_equal(result, expected)
    assert result.shape == (70, 66, 3)
    assert audit["input_padding_right"] == audit["input_padding_bottom"] == 1
    assert audit["tile_coverage_exact"]


def test_alpha_is_exact_nearest_replication():
    alpha = np.array([[0, 30, 128], [160, 230, 255]], dtype=np.uint8)
    result = upscale_alpha(alpha)
    assert result.shape == (4, 6)
    assert np.array_equal(result[::2, ::2], alpha)
    assert np.array_equal(result[1::2, 1::2], alpha)
    assert set(np.unique(result)) == set(np.unique(alpha))


@pytest.mark.parametrize("tile", [0, 31, 33, 513, 1024, True, 32.0])
def test_invalid_tile_profiles_are_rejected(tile):
    with pytest.raises(ValueError, match="even integer"):
        validate_tile(tile)
    assert HALO == 192


def test_nonfinite_or_wrong_scale_model_outputs_fail(local_model):
    import torch
    rgb = np.zeros((32, 32, 3), dtype=np.uint8)
    class Bad(torch.nn.Module):
        def __init__(self):
            super().__init__(); self.anchor = torch.nn.Parameter(torch.zeros(()))
        def forward(self, image):
            return torch.full((1, 3, image.shape[2] * 2, image.shape[3] * 2), float("nan"))
    with pytest.raises(RuntimeError, match="nonfinite"):
        upscale(Bad(), rgb, tile=32)
    local_model.half()
    with pytest.raises(ValueError, match="FP32"):
        upscale(local_model, rgb, tile=32)


def test_weight_profile_and_corrupt_files_fail_before_torch_loading(tmp_path):
    path = tmp_path / "wrong.pth"; path.write_bytes(b"not model weights")
    with pytest.raises(ValueError, match="pinned"):
        validate_checkpoint({"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        validate_checkpoint({"path": str(path), "sha256": WEIGHTS_SHA256})


def test_vendor_namespace_does_not_modify_basicsr_training_module(monkeypatch):
    pytest.importorskip("torch")
    sentinel = types.ModuleType("basicsr"); sentinel.training_marker = object()
    monkeypatch.setitem(sys.modules, "basicsr", sentinel)
    architecture = importlib.import_module("meshaxiom.vendor.superresolution.rrdbnet_arch")
    assert architecture.RRDBNet.__module__.startswith("meshaxiom.vendor.superresolution")
    assert sys.modules["basicsr"] is sentinel
    binding = validate_vendor()
    assert binding["architecture"]["weights_bundled"] is False
    assert any("Apache" in name for name in binding["architecture"]["licenses"])


def test_worker_rejects_nonempty_output_without_overwriting_completed_receipt(tmp_path):
    from meshaxiom.workers.superresolution import main
    out = tmp_path / "complete"; out.mkdir()
    receipt = out / "report.json"; receipt.write_text('{"status":"COMPLETE"}\n')
    request = tmp_path / "request.json"
    request.write_text(json.dumps({"output_dir": str(out), "image": str(tmp_path / "unused.png"), "config": {}}))
    assert main(["--request", str(request)]) == 1
    assert json.loads(receipt.read_text())["status"] == "COMPLETE"
    assert len(list(out.glob("request-failed-*.json"))) == 1
