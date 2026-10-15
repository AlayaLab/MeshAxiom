"""JSON-request texture worker; external GPU libraries are imported lazily."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
import traceback


def run(request):
    from meshaxiom.material_backend.alignment import prepare_reference, align_mesh
    from meshaxiom.material_backend.geometry import load_mesh, export_textured_mesh, file_sha256, surface_face_mask
    from meshaxiom.material_backend.pixal import texture_mesh, validate_material_config

    output_dir = Path(request["output_dir"]).expanduser().resolve()
    image_path = Path(request["image"]).expanduser().resolve(strict=True)
    mesh_path = Path(request["mesh"]).expanduser().resolve(strict=True)
    if output_dir.exists() and any(output_dir.iterdir()):
        # The public orchestrator creates only these control files before it
        # starts the worker. It still rejects nonempty directories itself.
        entries = {path.name for path in output_dir.iterdir()}
        controls = {"request.json", "status.json", "material.log"}
        owned = entries <= controls and (output_dir / "request.json").is_file() and (output_dir / "status.json").is_file()
        if owned:
            owned = json.loads((output_dir / "request.json").read_text()) == request and json.loads((output_dir / "status.json").read_text()).get("status") == "running"
        if not owned:
            raise FileExistsError("Material output directory must be empty except for its active orchestrator control files; completed output is never overwritten")
    output_dir.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    config = request.get("config", {})
    settings = config.get("material", {})
    validate_material_config(config)
    texture_size = int(request.get("texture_size", 2048))
    if texture_size not in {256, 512, 1024, 2048, 4096}:
        raise ValueError("texture_size must be a power of two between 256 and 4096")
    vertices, faces, source = load_mesh(mesh_path)
    surface_mask = surface_face_mask(vertices, faces)
    if not surface_mask.any():
        raise ValueError("Mesh has no nondegenerate surface for texture inference")
    reference, mask, reference_policy = prepare_reference(image_path)
    reference.save(output_dir / "reference_aligned.png")
    aligned, camera = align_mesh(vertices, faces[surface_mask], mask, request.get("camera"),
        up_axis=str(settings.get("up_axis", source["up_axis"])), min_iou=float(settings.get("min_alignment_iou", .25)))
    camera["reference_preparation"] = reference_policy
    (output_dir / "camera.json").write_text(json.dumps(camera, indent=2) + "\n")
    surface_mask &= surface_face_mask(aligned, faces)
    if not surface_mask.any():
        raise ValueError("Normalized mesh has no nondegenerate surface for texture inference")
    baked, backend = texture_mesh(config, aligned, faces[surface_mask], reference, camera, output_dir,
        seed=int(request.get("seed", 42)), texture_size=texture_size, device=request.get("device", "cuda:0"))
    artifacts, audit = export_textured_mesh(output_dir, vertices, faces, aligned, baked, inference_face_mask=surface_mask)
    provenance = {
        "status": "PASS", "method": "supplied static geometry with texture-only inference", "source_mesh": source,
        "image_path": str(image_path), "image_sha256": file_sha256(image_path), "camera": camera,
        "geometry_audit": audit, "backend": backend,
        "artifacts": {name: {"path": path, "sha256": file_sha256(path)} for name, path in artifacts.items()},
        "seconds": time.monotonic() - started,
        "limitations": ["Single-view materials on unseen surfaces are inferred.", "Silhouette camera registration does not prove semantic viewpoint alignment.",
            "Static export does not preserve scene hierarchy, skeletons or animations. Exact coincident source positions are compared after position welding."],
    }
    provenance_path = output_dir / "provenance.json"
    provenance_path.write_text(json.dumps(provenance, indent=2) + "\n")
    artifacts["provenance.json"] = str(provenance_path)
    artifacts["camera.json"] = str(output_dir / "camera.json")
    report = {"status": "COMPLETE", "artifacts": artifacts, "geometry_audit": audit, "coordinate_frame": "same_as_input_static_mesh",
        "provenance_sha256": file_sha256(provenance_path), "elapsed_seconds": time.monotonic() - started}
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
        output_dir = Path(request["output_dir"]).expanduser().resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        failure_path = output_dir / "report.json"
        # A duplicate request never rewrites the prior success receipt.
        if failure_path.exists():
            failure_path = output_dir / ("request-failed-" + str(time.time_ns()) + ".json")
        failure_path.write_text(json.dumps({"status": "FAILED", "error_type": type(error).__name__, "error": str(error)}, indent=2) + "\n")
        traceback.print_exc()
        return 1
    print(json.dumps({"status": report["status"], "report": str(Path(request["output_dir"]).resolve() / "report.json")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
