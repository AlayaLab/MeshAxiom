"""Portable V197 vertex generation from the released EMA checkpoint."""
from __future__ import annotations

import gc
from pathlib import Path
import time

from meshaxiom.vendor.recipe.runtime import (
    activate_namespace, binding, checkpoint, cuda_device, image_pixels, stage_cli, write_json,
)


def run(request: dict) -> str:
    """Run the production MBR8, level-3 consensus vertex recipe in a clean process."""
    import numpy as np
    import torch
    from meshaxiom.workers._geometry_state import load
    from meshaxiom.vendor.recipe.gen_mbr128 import per_level, sample_mbr
    state = load(request, "vertex", exact=False)
    out = Path(request["output_dir"])
    report_path = out / "vertex_report.json"
    if report_path.exists() or (out / "vertices.npz").exists():
        raise FileExistsError("vertex output already exists; use a fresh output directory")
    dev, cfg, net, flow, artifact = (state[k] for k in ("dev", "cfg", "net", "flow", "artifact"))
    pixels, input_audit = image_pixels(cfg, request)
    settings = dict(k=8, select_level=3, seed=request["seed"], cfg="3x4,5x5", threshold=0.4,
                    steps=request["steps"], sampler="dpmpp2m", max_nodes=200000)
    if False:
        state["backend"].maintain()
    torch.cuda.synchronize(dev)
    started = time.monotonic()
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        context = net.cond(lod=torch.tensor([float(request["face_budget"])], device=dev), image=pixels.to(dev)).float()
        if False:
            from p4_vertex_engine import sample_mbr as exact_sample
            levels, info = exact_sample(state["backend"], net, context, K=8, select_level=3, seed=request["seed"], thr=0.4,
                                       cfg=per_level("3x4,5x5", net.depth_total), steps=request["steps"],
                                       sampler="dpmpp2m", max_nodes=200000)
        else:
            levels, info = sample_mbr(net, context, flow, K=8, select_level=3, seed=request["seed"], thr=0.4,
                                     cfg=per_level("3x4,5x5", net.depth_total), steps=request["steps"],
                                     sampler="dpmpp2m", max_nodes=200000)
    torch.cuda.synchronize(dev)
    coords = np.asarray(levels[-1], np.int64)
    if coords.ndim != 2 or coords.shape[1] != 3 or not 0 < len(coords) <= 26000:
        raise ValueError("generated vertex count/shape exceeds the released topology contract")
    keys = coords[:, 0] * 512 * 512 + coords[:, 1] * 512 + coords[:, 2]
    order = np.argsort(keys, kind="stable")
    coords = coords[order]
    if coords.min() < 0 or coords.max() > 511 or not np.all(np.diff(keys[order]) > 0):
        raise ValueError("D9 vertices must be unique, ordered integer grid cells")
    np.savez_compressed(out / "vertices.npz", coords=coords.astype(np.int16))
    report = {
        "status": "COMPLETE", "stage": "vertex", "checkpoint": artifact,
        "implementation": "reference", "models_resident": True,
        "step": artifact["step"], "weights": "ema", "input_mode": "image_only", "gt_used": False,
        "image_sha256": input_audit["input"]["sha256"], "encoder_input_audit": input_audit,
        "face_budget": request["face_budget"], "vertex_lod": request["face_budget"],
        "vertices": len(coords), "coords": binding(out / "vertices.npz"),
        "sampler": settings, "production_recipe": request["steps"] == 20,
        "level_nodes": [len(level) for level in levels], "sample_info": info,
        "coordinate_frame": "z_up", "coordinate_grid": {"depth": 9, "world": "(coords+0.5)/512*2-1"},
        "seconds": round(time.monotonic() - started, 3),
        "artifacts": {"vertices": binding(out / "vertices.npz")},
    }
    write_json(report_path, report)
    return str(report_path)


if __name__ == "__main__":
    stage_cli(run)
