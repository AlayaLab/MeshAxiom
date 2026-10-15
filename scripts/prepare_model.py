#!/usr/bin/env python3
"""Stage the Base model locally, preserving every tensor and auditing metadata.

No network, upload, training, precision conversion or model inference is performed.
The source files are read-only inputs; the destination must be new.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import shutil

SOURCE_SHA = {
    "vertex": "4d60148fdddb65df502becbf9786131836a880fdba226180ce586a8ee10620fb",
    "topology": "3818846563217c1d6819c8c9dbf7d8b8cc43b7a6cd2e96d6b37b576fe147ef45",
    "vae": "7d6833425af71e923756d2ebdb2863e704857bffb8b365ab0c2f5ec3bfb6f4bd",
}
STEPS = {"vertex": 180000, "topology": 10000, "vae": 30000}


def digest(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def tensor_equal(a, b, torch):
    if torch.is_tensor(a):
        return torch.is_tensor(b) and a.shape == b.shape and a.dtype == b.dtype and torch.equal(a, b)
    if isinstance(a, dict):
        return isinstance(b, dict) and a.keys() == b.keys() and all(tensor_equal(a[k], b[k], torch) for k in a)
    if isinstance(a, (list, tuple)):
        return type(a) is type(b) and len(a) == len(b) and all(tensor_equal(x, y, torch) for x, y in zip(a, b))
    return type(a) is type(b) and a == b


def prepare(weights, output, *, link=False):
    import torch
    torch.set_num_threads(2)
    root = Path(__file__).resolve().parents[1]
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError("Model output must be a new directory")
    output.mkdir(parents=True)
    shutil.copytree(root / "deploy/huggingface/model", output, dirs_exist_ok=True)
    (output / "weights").mkdir()
    reports = []
    for kind in ["vertex", "topology", "vae"]:
        source = Path(weights).resolve() / (kind + "-ema.pt")
        source_sha = digest(source)
        if source_sha != SOURCE_SHA[kind]:
            raise ValueError("Wrong source EMA file: " + kind)
        before = torch.load(source, map_location="cpu", mmap=True, weights_only=True)
        if before.get("kind") != kind or before.get("step") != STEPS[kind] or before.get("weights") != "ema":
            raise ValueError("EMA metadata mismatch: " + kind)
        destination = output / "weights" / source.name
        metadata_paths = []
        if kind == "topology":
            after = dict(before)
            after["v113"] = dict(before["v113"], version="meshaxiom-topology-richknn-v118")
            torch.save(after, destination)
            metadata_paths = ["v113.version"]
        elif link:
            os.link(source, destination)
        else:
            shutil.copyfile(source, destination)
        loaded = torch.load(destination, map_location="cpu", mmap=True, weights_only=True)
        comparison = dict(loaded)
        if kind == "topology":
            comparison["v113"] = dict(loaded["v113"], version=before["v113"]["version"])
        if not tensor_equal(before, comparison, torch):
            raise ValueError("A tensor or non-brand metadata value changed: " + kind)
        # Inspect all names and metadata separately from floating-point payloads.
        reports.append({"kind": kind, "file": "weights/" + source.name, "step": STEPS[kind],
                        "weights": "ema", "tensor_count": len(before["ema"]),
                        "all_tensor_values_and_dtypes_equal": True, "only_metadata_changes": metadata_paths,
                        "source_sha256": source_sha, "sha256": digest(destination), "size": destination.stat().st_size})
        del before, loaded, comparison
    manifest = {"status": "PASS", "model": "MeshAxiom-Base", "implementation": "exact", "files": reports,
                "scope": "Metadata branding and complete tensor equality; no new GPU identity claim"}
    (output / "WEIGHTS.json").write_text(json.dumps(manifest, indent=2) + "\n")
    config = json.loads((output / "meshaxiom.json").read_text())
    for row in reports:
        config["geometry"][row["kind"] + "_checkpoint"]["sha256"] = row["sha256"]
    (output / "meshaxiom.json").write_text(json.dumps(config, indent=2) + "\n")
    files = [p for p in output.rglob("*") if p.is_file() and p.name != "SHA256SUMS"]
    (output / "SHA256SUMS").write_text("".join(digest(p) + "  " + str(p.relative_to(output)) + "\n" for p in sorted(files)))
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--link", action="store_true", help="Use hard links for the two unchanged files on the same filesystem")
    args = parser.parse_args()
    print(json.dumps(prepare(args.weights_dir, args.output, link=args.link), indent=2))
