"""Explicit reference-versus-Exact verification, separate from production generation."""
from pathlib import Path
import json

from .contracts import MeshAxiomError, file_binding, write_json
from .pipeline import MeshAxiomPipeline

FILES = ("vertices.npz", "topology.npz", "mesh.obj", "mesh.json")


def compare_outputs(reference, exact):
    rows = {}
    for name in FILES:
        a, b = Path(reference) / name, Path(exact) / name
        if not a.is_file() or not b.is_file():
            rows[name] = {"equal": False, "missing": True}
            continue
        ar, br = file_binding(a), file_binding(b)
        rows[name] = {"equal": ar["size"] == br["size"] and ar["sha256"] == br["sha256"],
                      "reference": ar, "exact": br}
    return {"status": "PASS" if all(r["equal"] for r in rows.values()) else "FAIL", "files": rows,
            "scope": "Four geometry files for these inputs, seeds, weights, GPU and software stack only"}


def verify_exact(config, image, output, *, face_budget=4000, seed=2026082301, background="keep"):
    """Use original eager forwards as a diagnostic; never silently fall back to them."""
    output = Path(output).expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise MeshAxiomError("Verification output directory must be empty")
    output.mkdir(parents=True, exist_ok=True)
    with MeshAxiomPipeline(config) as reference:
        reference._reference = True
        reference.generate(image, output / "reference", face_budget=face_budget, seed=seed, background=background)
    # Exactly the prepared pixels, without a second matting/crop operation.
    with MeshAxiomPipeline(config) as exact:
        exact.generate(output / "reference/input.png", output / "exact", face_budget=face_budget, seed=seed, background="keep")
    result = compare_outputs(output / "reference", output / "exact")
    result.update(face_budget=face_budget, seed=seed)
    write_json(output / "IDENTITY.json", result)
    if result["status"] != "PASS":
        raise MeshAxiomError("Exact file identity failed; both original outputs and IDENTITY.json are retained")
    return result
