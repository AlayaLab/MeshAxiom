#!/usr/bin/env python3
"""Audit a source/model directory or built source ZIP, tarball, or wheel."""
import argparse
import io
import json
from pathlib import Path
import re
import tarfile
import zipfile

SKIP = {".git", "__pycache__", ".pytest_cache", "build", "dist"}


def entries(path):
    path = Path(path)
    if path.is_dir():
        for p in sorted(path.rglob("*")):
            relative = p.relative_to(path)
            if any(part in SKIP or part.endswith(".egg-info") for part in relative.parts) or not p.is_file():
                continue
            if p.suffix == ".pt":
                with zipfile.ZipFile(p) as archive:
                    data = b"\n".join(archive.read(n) for n in archive.namelist() if n.endswith("data.pkl"))
                yield str(relative), data
            else:
                yield str(relative), p.read_bytes()
    elif zipfile.is_zipfile(path):
        with zipfile.ZipFile(path) as archive:
            for name in archive.namelist():
                if not name.endswith("/"):
                    yield name, archive.read(name)
    elif tarfile.is_tarfile(path):
        with tarfile.open(path) as archive:
            for member in archive:
                if member.isfile():
                    yield member.name, archive.extractfile(member).read()
    else:
        raise ValueError("Expected a directory, ZIP, wheel, or tarball")


def audit(path, forbidden, *, model=False):
    issues = []
    count = 0
    for name, data in entries(path):
        count += 1
        for term in forbidden:
            if term.casefold() in name.casefold():
                issues.append({"file": name, "reason": "forbidden filename", "term": term})
            # Scan textual content and checkpoint pickle metadata, not float/image payloads.
            if Path(name).suffix not in {".png", ".jpg", ".jpeg", ".webp", ".glb"}:
                if term.casefold().encode() in data.lower():
                    issues.append({"file": name, "reason": "forbidden content", "term": term})
        if not model and Path(name).suffix in {".pt", ".pth", ".safetensors"}:
            issues.append({"file": name, "reason": "model file in source payload"})
        if re.search(rb"(?:hf_[A-Za-z0-9]{20,}|LTAI[A-Za-z0-9]{16,})", data):
            issues.append({"file": name, "reason": "possible credential; value withheld"})
    return {"status": "PASS" if not issues else "FAIL", "files_scanned": count, "issues": issues,
            "scope": "Names/text and model pickle metadata; tensor value equality is a separate preparation check"}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path)
    parser.add_argument("--forbid-term", action="append", default=[])
    parser.add_argument("--model", action="store_true")
    args = parser.parse_args()
    report = audit(args.path, args.forbid_term, model=args.model)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    raise SystemExit(0 if report["status"] == "PASS" else 1)
