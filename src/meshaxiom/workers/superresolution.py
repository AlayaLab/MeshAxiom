"""Optional atlas postprocessing worker; runs independently of mesh inference."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
import traceback


def run(request):
    import numpy as np
    from PIL import Image
    from meshaxiom.material_backend.superresolution import load_model, upscale, upscale_alpha, validate_checkpoint, validate_vendor, validate_tile, sha256

    output_dir = Path(request["output_dir"]).expanduser().resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        entries = {path.name for path in output_dir.iterdir()}
        owned = entries <= {"request.json", "status.json", "superresolution.log"} and (output_dir / "request.json").is_file() and (output_dir / "status.json").is_file()
        if owned:
            owned = json.loads((output_dir / "request.json").read_text()) == request and json.loads((output_dir / "status.json").read_text()).get("status") == "running"
        if not owned:
            raise FileExistsError("Superresolution output must be empty except for its active orchestrator control files")
    output_dir.mkdir(parents=True, exist_ok=True)
    source_path = Path(request["image"]).expanduser().resolve(strict=True)
    settings = request.get("config", {}).get("material", {})
    binding = settings.get("superresolution_checkpoint")
    checkpoint_path = validate_checkpoint(binding)
    vendor = validate_vendor()
    tile = validate_tile(request.get("tile", 512))
    with Image.open(source_path) as source:
        if source.mode in {"I", "I;16", "F"} or "16" in source.mode:
            raise ValueError("This atlas profile supports 8-bit color images; HDR/16-bit inputs require an explicit color conversion")
        has_alpha = "A" in source.getbands() or "transparency" in source.info
        rgba = np.asarray(source.convert("RGBA"))
        rgb = rgba[:, :, :3].copy()
        alpha = rgba[:, :, 3].copy() if has_alpha else None
    import torch

    device = torch.device(request.get("device", "cuda:0"))
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("The superresolution worker requires a configured CUDA device")
    torch.cuda.set_device(device)
    torch.set_num_threads(2)
    torch.set_num_interop_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.cuda.reset_peak_memory_stats(device)
    started = time.monotonic()
    model = load_model(binding, device=str(device))
    result, audit = upscale(model, rgb, tile=tile)
    if alpha is not None:
        result = np.dstack([result, upscale_alpha(alpha)])
    image_path = output_dir / "enhanced.png"
    Image.fromarray(result).save(image_path)
    # Read actual PNG bytes back; alpha and RGB channels must survive encoding.
    with Image.open(image_path) as written:
        if not np.array_equal(np.asarray(written), result):
            raise RuntimeError("Lossless PNG readback did not preserve output pixels")
    report = {
        "status": "COMPLETE", "method": "RealESRGAN_x2plus optional atlas superresolution", "artifacts": {"enhanced.png": str(image_path)},
        "source": {"path": str(source_path), "sha256": sha256(source_path), "dimensions": audit["input_dimensions"]},
        "output": {"sha256": sha256(image_path), "dimensions": audit["output_dimensions"], "lossless_png_readback_exact": True},
        "model": {"path": str(checkpoint_path), "sha256": binding["sha256"], "weights": "params_ema", "strict_load": True},
        "inference": audit, "input_dimensions": audit["input_dimensions"], "output_dimensions": audit["output_dimensions"],
        "scale": 2, "tile_coverage_exact": audit["tile_coverage_exact"], "alpha_present": alpha is not None,
        "alpha_policy": "nearest x2 repeats exact source alpha; alpha is never inferred" if alpha is not None else "source has no alpha",
        "geometry_or_uv_modified": False, "precision": "float32", "tf32_enabled": False,
        "vendor": vendor, "framework": torch.__version__, "cuda_version": torch.version.cuda,
        "device": str(device), "gpu_name": torch.cuda.get_device_name(device),
        "peak_gpu_bytes": torch.cuda.max_memory_allocated(device), "seconds": time.monotonic() - started,
        "limitations": ["Enhanced microtexture is model inference, not a high-resolution ground-truth reconstruction.", "This standalone operation writes an atlas only; it does not reembed textures into a GLB."],
    }
    (output_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", type=Path, required=True)
    args = parser.parse_args(argv)
    request = json.loads(args.request.read_text())
    try:
        report = run(request)
    except Exception as error:
        output = Path(request["output_dir"]).expanduser().resolve()
        output.mkdir(parents=True, exist_ok=True)
        path = output / "report.json"
        if path.exists():
            path = output / ("request-failed-" + str(time.time_ns()) + ".json")
        path.write_text(json.dumps({"status": "FAILED", "error_type": type(error).__name__, "error": str(error)}, indent=2) + "\n")
        traceback.print_exc()
        return 1
    print(json.dumps({"status": report["status"], "report": str(Path(request["output_dir"]).resolve() / "report.json")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
