"""CPU mesh I/O and strict oriented-triangle validation after UV splitting."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_mesh(vertices, faces) -> tuple[np.ndarray, np.ndarray]:
    vertices = np.asarray(vertices, dtype=np.float64)
    original_faces = np.asarray(faces)
    if vertices.ndim != 2 or vertices.shape[1:] != (3,) or not len(vertices):
        raise ValueError("Mesh vertices must be a nonempty N x 3 array")
    if original_faces.ndim != 2 or original_faces.shape[1:] != (3,) or not len(original_faces):
        raise ValueError("Only a nonempty triangle mesh is supported")
    if not np.isfinite(vertices).all() or not np.isfinite(original_faces).all():
        raise ValueError("Mesh contains nonfinite values")
    if not np.equal(original_faces, np.floor(original_faces)).all():
        raise ValueError("Face indices must be integers")
    faces = original_faces.astype(np.int64)
    if faces.min() < 0 or faces.max() >= len(vertices):
        raise ValueError("Face indices are outside the vertex array")
    if np.ptp(vertices, axis=0).max() <= 0:
        raise ValueError("Mesh has no extent")
    return vertices.copy(), faces.copy()


def surface_face_mask(vertices, faces, *, relative_area_epsilon=1e-12):
    """Identify triangles with numerical surface area, without editing faces.

    Native collapsed/collinear faces remain part of the final mesh. Only their
    texture inference is skipped, with exact original face restoration later.
    """
    vertices, faces = validate_mesh(vertices, faces)
    doubled_area = np.linalg.norm(np.cross(vertices[faces[:, 1]] - vertices[faces[:, 0]], vertices[faces[:, 2]] - vertices[faces[:, 0]]), axis=1)
    extent = np.ptp(vertices, axis=0).max()
    return doubled_area > relative_area_epsilon * extent * extent


def load_mesh(path: str | Path) -> tuple[np.ndarray, np.ndarray, dict]:
    """Read mesh JSON, triangular OBJ, or static GLB without repairing geometry.

    Scene node transforms are baked, repeated nodes remain repeated geometry.
    Mesh JSON uses its own raw coordinates; GLB and OBJ default to Y-up.
    """
    import trimesh

    path = Path(path).resolve(strict=True)
    suffix = path.suffix.lower()
    metadata = {"source_path": str(path), "source_sha256": file_sha256(path), "format": suffix.lstrip("."), "up_axis": "y"}
    if suffix == ".json":
        payload = json.loads(path.read_text())
        vertices, faces = validate_mesh(payload["vertices"], payload["faces"])
        metadata["up_axis"] = payload.get("metadata", {}).get("up_axis", payload.get("up_axis", "y"))
    elif suffix in {".obj", ".glb"}:
        if suffix == ".obj":
            for line in path.read_text().splitlines():
                if line.lstrip().startswith("f ") and len(line.split()) != 4:
                    raise ValueError("OBJ faces must already be triangles; automatic triangulation is disabled")
        if suffix == ".glb":
            import struct
            with path.open("rb") as stream:
                header = stream.read(20)
                if len(header) != 20 or header[:4] != b"glTF" or struct.unpack("<I", header[4:8])[0] != 2 or header[16:20] != b"JSON":
                    raise ValueError("Only glTF 2 binary meshes are supported")
                scene_document = json.loads(stream.read(struct.unpack("<I", header[12:16])[0]))
            if scene_document.get("skins") or scene_document.get("animations") or any(p.get("targets") for m in scene_document.get("meshes", []) for p in m.get("primitives", [])):
                raise ValueError("Material export is static; animated, skinned and morph-target GLBs must be converted to a chosen static pose explicitly")
            if any(p.get("mode", 4) != 4 for m in scene_document.get("meshes", []) for p in m.get("primitives", [])):
                raise ValueError("Only triangle-list GLB primitives are supported")
        scene = trimesh.load(str(path), force="scene", process=False, maintain_order=True)
        parts = []
        for node in scene.graph.nodes_geometry:
            transform, geometry_name = scene.graph[node]
            mesh = scene.geometry[geometry_name]
            if not isinstance(mesh, trimesh.Trimesh):
                raise ValueError("Input contains non-triangle geometry")
            # Applying a negative-determinant node transform requires winding
            # reversal to preserve the oriented world-space surface.
            part_vertices = trimesh.transformations.transform_points(mesh.vertices, transform)
            part_faces = np.asarray(mesh.faces).copy()
            if np.linalg.det(transform[:3, :3]) < 0:
                part_faces = part_faces[:, ::-1]
            parts.append((part_vertices, part_faces))
        if not parts:
            raise ValueError("No mesh geometry found")
        vertices = np.concatenate([v for v, _ in parts])
        faces, offset = [], 0
        for part_vertices, part_faces in parts:
            faces.append(part_faces + offset)
            offset += len(part_vertices)
        vertices, faces = validate_mesh(vertices, np.concatenate(faces))
        metadata["scene_nodes_baked"] = len(parts)
    else:
        raise ValueError("Mesh input must be OBJ, GLB, or mesh.json")
    metadata.update(vertices=len(vertices), faces=len(faces))
    return vertices, faces, metadata


def canonical_faces(faces: np.ndarray) -> np.ndarray:
    """Canonical cyclic rotation, preserving orientation and multiplicity."""
    faces = np.asarray(faces, dtype=np.int64)
    rows = faces.copy()
    # Lexicographically smallest cyclic rotation also handles repeated indices
    # in retained native degenerate faces.
    for shift in (1, 2):
        other = np.roll(faces, -shift, axis=1)
        lower = (other[:, 0] < rows[:, 0]) | ((other[:, 0] == rows[:, 0]) & (other[:, 1] < rows[:, 1])) | ((other[:, 0] == rows[:, 0]) & (other[:, 1] == rows[:, 1]) & (other[:, 2] < rows[:, 2]))
        rows[lower] = other[lower]
    return rows[np.lexsort((rows[:, 2], rows[:, 1], rows[:, 0]))]


def audit_uv_split(source_vertices, source_faces, output_vertices, output_faces, *, tolerance=1e-5) -> tuple[np.ndarray, dict]:
    """Validate every vertex and oriented triangle, then return source indices.

    Exact coincident source positions are welded in both meshes for comparison;
    their UV-split index identities cannot be inferred from positions alone.
    Distinct source positions never merge. Duplicate triangles remain counted.
    """
    source_vertices, source_faces = validate_mesh(source_vertices, source_faces)
    output_vertices, output_faces = validate_mesh(output_vertices, output_faces)
    if not np.isfinite(tolerance) or not 0 < tolerance <= 1e-4:
        raise ValueError("Invalid geometry audit tolerance")
    unique, first, inverse = np.unique(source_vertices, axis=0, return_index=True, return_inverse=True)
    tree = cKDTree(unique)
    distances, weld_map = tree.query(output_vertices)
    if distances.max() > tolerance:
        raise ValueError(f"UV baking moved or invented a vertex: max displacement {distances.max():.8g}")
    if len(unique) > 1:
        near = tree.query(output_vertices, k=2)[0]
        if np.any(np.isclose(near[:, 0], near[:, 1], rtol=0, atol=1e-12)):
            raise ValueError("Ambiguous vertex correspondence between distinct source positions")
    expected = canonical_faces(inverse[source_faces])
    actual = canonical_faces(weld_map[output_faces])
    if not np.array_equal(expected, actual):
        raise ValueError("UV baking changed the oriented triangle multiset (missing, added, reversed or collapsed faces)")
    used_source = np.unique(inverse[source_faces])
    used_output = np.unique(weld_map[output_faces])
    if not np.array_equal(used_source, used_output):
        raise ValueError("UV baking changed the used vertex set")
    mapping = first[weld_map]
    audit = {
        "status": "PASS", "source_vertices": len(source_vertices), "source_faces": len(source_faces),
        "output_vertices_after_uv_split": len(output_vertices), "output_faces": len(output_faces),
        "source_unique_positions": len(unique), "coincident_source_vertex_count": len(source_vertices) - len(unique),
        "source_unused_vertex_count": len(source_vertices) - len(np.unique(source_faces)),
        "triangle_indices_exact_after_position_weld": True, "oriented_triangle_multiplicity_exact": True,
        "used_source_positions_exact": True, "max_normalized_vertex_displacement": float(distances.max()),
        "audit_tolerance_normalized": tolerance, "geometry_generated": False, "remeshed": False,
        "surface_sha256": hashlib.sha256(expected.astype("<i8").tobytes()).hexdigest(),
        "coincident_index_identity_preserved": len(unique) == len(source_vertices),
    }
    return mapping, audit


def export_textured_mesh(output_dir, original_vertices, original_faces, aligned_vertices, baked_mesh, *, inference_face_mask=None) -> tuple[dict, dict]:
    """Undo upstream axis conversion and write exact original-coordinate vertices."""
    import trimesh

    original_vertices, original_faces = validate_mesh(original_vertices, original_faces)
    exported = np.asarray(baked_mesh.vertices, dtype=np.float64)
    internal = np.stack([exported[:, 0], -exported[:, 2], exported[:, 1]], axis=1)
    faces = np.asarray(baked_mesh.faces)
    expected_inference_mask = surface_face_mask(original_vertices, original_faces) & surface_face_mask(aligned_vertices, original_faces)
    if inference_face_mask is None:
        inference_face_mask = expected_inference_mask
    inference_face_mask = np.asarray(inference_face_mask, dtype=bool)
    if inference_face_mask.shape != (len(original_faces),) or not inference_face_mask.any():
        raise ValueError("Mesh has no nondegenerate surface for texture inference")
    if not np.array_equal(inference_face_mask, expected_inference_mask):
        raise ValueError("Texture inference may exclude only the explicitly detected native degenerate faces")
    # The decoder may reorder or split UV vertices, but it must preserve every
    # actual surface triangle. Excluded native faces are restored separately.
    mapping, baking_audit = audit_uv_split(aligned_vertices, original_faces[inference_face_mask], internal, faces)
    uv = np.asarray(baked_mesh.visual.uv, dtype=np.float64)
    if uv.shape != (len(mapping), 2) or not np.isfinite(uv).all():
        raise ValueError("UV array is missing, nonfinite or not aligned with exported vertices")
    # Keep every input position, including isolated vertices and duplicate seam
    # indices not selected by nearest-position correspondence. These appended
    # vertices are not referenced by any new triangle.
    missing = np.setdiff1d(np.arange(len(original_vertices)), np.unique(mapping))
    if len(missing):
        mapping = np.concatenate([mapping, missing])
        uv = np.concatenate([uv, np.zeros((len(missing), 2))])
    # Reuse the UV at the same position when available. Zero is a deterministic
    # fallback for an isolated position with no surface texel contribution.
    uv_by_position = {}
    for index, original_index in enumerate(mapping[:len(mapping) - len(missing)] if len(missing) else mapping):
        uv_by_position.setdefault(tuple(original_vertices[original_index]), uv[index].copy())
    if len(missing):
        for index, original_index in enumerate(missing, start=len(mapping) - len(missing)):
            uv[index] = uv_by_position.get(tuple(original_vertices[original_index]), np.zeros(2))
    restored_native_faces = original_faces[~inference_face_mask]
    if len(restored_native_faces):
        output_index = np.full(len(original_vertices), -1, dtype=np.int64)
        for index, original_index in enumerate(mapping):
            if output_index[original_index] < 0:
                output_index[original_index] = index
        faces = np.concatenate([faces, output_index[restored_native_faces]])
    exact_vertices = original_vertices[mapping]
    _, audit = audit_uv_split(original_vertices, original_faces, exact_vertices, faces, tolerance=1e-5)
    audit["source_vertices_retained_in_export_mapping"] = len(np.unique(mapping)) == len(original_vertices)
    audit["unreferenced_source_vertices_appended"] = len(missing)
    audit["max_normalized_vertex_displacement"] = baking_audit["max_normalized_vertex_displacement"]
    audit["texture_inference_excluded_degenerate_faces"] = int((~inference_face_mask).sum())
    audit["restored_native_faces_count"] = len(restored_native_faces)
    audit["degenerate_face_policy"] = "Skip only in texture inference/UV baking; restore exact source triangles in final oriented multiset"
    audit["degenerate_area_relative_epsilon"] = 1e-12
    audit["restored_face_uv_policy"] = "Existing UV at identical position when available; deterministic [0,0] for isolated positions"
    audit["export_vertices_exact_source_coordinates"] = bool(np.array_equal(exact_vertices, original_vertices[mapping]))
    audit["export_surface_sha256"] = audit["surface_sha256"]
    material = baked_mesh.visual.material
    if material.baseColorTexture is None or material.metallicRoughnessTexture is None:
        raise ValueError("Backend did not return both PBR textures")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    material.baseColorTexture.save(output_dir / "base_color.png")
    material.metallicRoughnessTexture.save(output_dir / "metallic_roughness.png")
    payload = {
        "vertices": exact_vertices.tolist(), "faces": faces.tolist(), "uv": uv.tolist(),
        "vertex_map": mapping.tolist(), "source_vertices": original_vertices.tolist(), "source_faces": original_faces.tolist(),
        "base_color": "base_color.png", "metallic_roughness": "metallic_roughness.png",
        "coordinate_frame": "same_as_input_static_mesh", "geometry_audit": audit,
    }
    (output_dir / "texture_mesh.json").write_text(json.dumps(payload, separators=(",", ":")) + "\n")
    textured = trimesh.Trimesh(vertices=exact_vertices, faces=faces, process=False,
        visual=trimesh.visual.TextureVisuals(uv=uv, material=material))
    textured.export(output_dir / "textured.glb")
    # Validate actual GLB bytes through a separate reader. Float32 is glTF's
    # storage precision; compare its exact quantized input positions.
    glb_vertices, glb_faces, _ = load_mesh(output_dir / "textured.glb")
    quantized_source = original_vertices.astype(np.float32).astype(np.float64)
    _, glb_audit = audit_uv_split(quantized_source, original_faces, glb_vertices, glb_faces)
    retained = np.array_equal(np.unique(glb_vertices, axis=0), np.unique(quantized_source, axis=0))
    if not retained:
        raise ValueError("Actual GLB roundtrip did not retain every source vertex position")
    audit["glb_roundtrip_oriented_triangles_exact"] = glb_audit["oriented_triangle_multiplicity_exact"]
    audit["glb_roundtrip_max_vertex_displacement"] = glb_audit["max_normalized_vertex_displacement"]
    audit["glb_roundtrip_source_positions_retained"] = True
    audit["export_vertices"] = len(exact_vertices)
    # Rewrite JSON after the actual GLB roundtrip succeeds.
    payload["geometry_audit"] = audit
    (output_dir / "texture_mesh.json").write_text(json.dumps(payload, separators=(",", ":")) + "\n")
    artifacts = {name: str((output_dir / name).resolve()) for name in ("texture_mesh.json", "textured.glb", "base_color.png", "metallic_roughness.png")}
    return artifacts, audit
