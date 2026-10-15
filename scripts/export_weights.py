#!/usr/bin/env python3
"""Export a single inference-only EMA checkpoint with a complete SHA-bound receipt.

This script never alters its source and does not include optimizer, training RNG,
training datasets, or raw-model weights in the output. Original full checkpoints
remain valid runtime inputs, so export is optional for local preparation.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import time


def digest(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def inference_config(source: dict) -> dict:
    """Preserve all architecture/sampling values, removing training-only paths."""
    cfg = {key: copy.deepcopy(source[key]) for key in
           ("depth", "model", "condition", "diffusion", "sample") if key in source}
    data_keys = ("depth", "num_points", "max_vertices", "max_faces", "image_resolution",
                 "image_background", "image_target_frame")
    cfg["data"] = {key: copy.deepcopy(value) for key, value in source.get("data", {}).items() if key in data_keys}
    if "vae" in source:
        cfg["vae"] = {key: copy.deepcopy(value) for key, value in source["vae"].items()
                      if key in ("sample_latent", "required_step")}
    if "sample_latent" in source.get("train", {}):
        cfg["train"] = {"sample_latent": source["train"]["sample_latent"]}
    model_id = cfg.get("condition", {}).get("model_id")
    if model_id and Path(model_id).is_absolute():
        cfg["condition"]["model_id"] = "facebook/dinov3-vitl16-pretrain-lvd1689m"
    return cfg


def portable_metadata(value):
    """Keep metadata content hashes but omit machine-private source paths."""
    if isinstance(value, dict):
        return {key: portable_metadata(item) for key, item in value.items()
                if key not in ("path", "remote", "local_root", "root", "mtime_ns", "inode")}
    if isinstance(value, (list, tuple)):
        return [portable_metadata(item) for item in value]
    return value


def export(source: Path, output: Path, kind: str, expected_sha: str, expected_step: int) -> dict:
    import torch
    source = source.resolve()
    output = output.resolve()
    if source == output or output.exists():
        raise FileExistsError("export requires a new output path, distinct from its source")
    if len(expected_sha) != 64:
        raise ValueError("a complete source SHA256 is required")
    start = time.monotonic()
    source_sha = digest(source)
    if source_sha != expected_sha.lower():
        raise ValueError("source checkpoint SHA256 does not match")
    ck = torch.load(source, map_location="cpu", mmap=True, weights_only=False)
    if not isinstance(ck, dict) or not ck.get("ema") or not isinstance(ck.get("cfg"), dict):
        raise ValueError("source needs cfg and EMA model tensors")
    if ck.get("step") != expected_step:
        raise ValueError(f"source step {ck.get('step')} != expected {expected_step}")
    state = ck["ema"]
    if not all(isinstance(name, str) and torch.is_tensor(value) for name, value in state.items()):
        raise ValueError("EMA must be a flat named tensor state dictionary")
    exported = {
        "step": int(ck["step"]), "cfg": inference_config(ck["cfg"]), "ema": state,
        "source_checkpoint_sha256": source_sha,
        "source_checkpoint_bytes": source.stat().st_size,
        "source_tree_sha256": ck.get("source_tree_sha256"),
        "inference_only": True, "weights": "ema", "kind": kind, "no_optimizer": True,
        "source_config_sha256": hashlib.sha256(json.dumps(ck["cfg"], sort_keys=True, default=str).encode()).hexdigest(),
    }
    for name in ("latent_stats", "v113", "vae_artifact", "resolved_config_sha256"):
        if name in ck:
            exported[name] = portable_metadata(ck[name])
    if kind == "topology" and not all(name in exported for name in ("latent_stats", "v113", "vae_artifact")):
        raise ValueError("topology inference export needs latent_stats, v113 and VAE basis metadata")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + f".{os.getpid()}.tmp")
    try:
        torch.save(exported, temporary)
        roundtrip = torch.load(temporary, map_location="cpu", mmap=True, weights_only=False)
        if set(roundtrip["ema"]) != set(state):
            raise ValueError("export tensor key roundtrip failed")
        for name, tensor in state.items():
            result = roundtrip["ema"][name]
            if tensor.dtype != result.dtype or tensor.shape != result.shape or not torch.equal(tensor, result):
                raise ValueError(f"export tensor roundtrip failed: {name}")
        del roundtrip
        os.replace(temporary, output)
    finally:
        if temporary.exists():
            temporary.unlink()
    receipt = {
        "schema_version": 1, "status": "PASS", "kind": kind, "step": int(ck["step"]),
        "weights": "ema", "inference_only": True, "optimizer_included": False,
        "source": {"name": source.name, "sha256": source_sha, "size": source.stat().st_size},
        "artifact": {"name": output.name, "sha256": digest(output), "size": output.stat().st_size},
        "source_tree_sha256": ck.get("source_tree_sha256"),
        "exact_tensor_roundtrip": True, "tensors": len(state),
        "embedded_external_backbone": {
            "model_id": ck["cfg"].get("condition", {}).get("model_id"),
            "tensors": sum(".backbone." in name for name in state),
            "numel": sum(tensor.numel() for name, tensor in state.items() if ".backbone." in name),
            "bytes": sum(tensor.numel() * tensor.element_size() for name, tensor in state.items() if ".backbone." in name),
            "retained": any(".backbone." in name for name in state),
            "license": "External model terms apply to any embedded backbone tensors.",
        },
        "seconds": round(time.monotonic() - start, 3),
    }
    receipt_path = output.with_suffix(output.suffix + ".export.json")
    receipt_path.write_text(json.dumps(receipt, indent=2) + "\n")
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--kind", choices=("vertex", "topology", "vae"), required=True)
    parser.add_argument("--sha256", required=True, help="Expected source file SHA256")
    parser.add_argument("--step", type=int, required=True, help="Expected source checkpoint step")
    args = parser.parse_args()
    print(json.dumps(export(args.source, args.output, args.kind, args.sha256, args.step), indent=2))


if __name__ == "__main__":
    main()
