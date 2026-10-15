"""CPU checks of actual mesh roundtrips and UV-split geometry protections."""
import json
from pathlib import Path

import numpy as np
from PIL import Image
import pytest
import trimesh

from meshaxiom.material_backend.alignment import align_mesh, prepare_reference
from meshaxiom.material_backend.geometry import audit_uv_split, export_textured_mesh, load_mesh, surface_face_mask


@pytest.fixture
def tetrahedron():
    vertices = np.array([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.], [0., 0., 1.]])
    faces = np.array([[0, 2, 1], [0, 1, 3], [1, 2, 3], [2, 0, 3]])
    return vertices, faces


def split_by_face(vertices, faces):
    return vertices[faces].reshape(-1, 3), np.arange(faces.size).reshape(-1, 3)


def test_uv_splits_and_face_reordering_keep_oriented_surface(tetrahedron):
    vertices, faces = tetrahedron
    split, triangles = split_by_face(vertices, faces)
    triangles = np.roll(triangles[::-1], 1, axis=1)
    mapping, audit = audit_uv_split(vertices, faces, split, triangles)
    assert audit["oriented_triangle_multiplicity_exact"]
    assert audit["output_vertices_after_uv_split"] == 12
    assert np.array_equal(split, vertices[mapping])


@pytest.mark.parametrize("mutation", ["flip", "remove", "add", "move"])
def test_baking_changes_are_rejected(tetrahedron, mutation):
    vertices, faces = tetrahedron
    output = vertices.copy(); changed = faces.copy()
    if mutation == "flip": changed[0] = changed[0, ::-1]
    if mutation == "remove": changed = changed[:-1]
    if mutation == "add": changed = np.concatenate([changed, changed[:1]])
    if mutation == "move": output[0, 0] += .001
    with pytest.raises(ValueError):
        audit_uv_split(vertices, faces, output, changed)


def test_duplicate_triangle_multiplicity_is_preserved(tetrahedron):
    vertices, faces = tetrahedron
    doubled = np.concatenate([faces, faces[:1]])
    with pytest.raises(ValueError, match="triangle multiset"):
        audit_uv_split(vertices, doubled, vertices, faces)


def test_coincident_uv_seam_vertices_have_explicit_weld_semantics(tetrahedron):
    vertices, faces = tetrahedron
    vertices = np.concatenate([vertices, vertices[:1]])
    faces = faces.copy(); faces[0, 0] = 4
    split, output_faces = split_by_face(vertices, faces)
    _, audit = audit_uv_split(vertices, faces, split, output_faces)
    assert audit["coincident_source_vertex_count"] == 1
    assert audit["coincident_index_identity_preserved"] is False
    assert audit["oriented_triangle_multiplicity_exact"]


def test_mesh_json_metadata_and_invalid_indices(tmp_path, tetrahedron):
    vertices, faces = tetrahedron
    path = tmp_path / "mesh.json"
    path.write_text(json.dumps({"vertices": vertices.tolist(), "faces": faces.tolist(), "metadata": {"up_axis": "z"}}))
    actual, triangles, metadata = load_mesh(path)
    assert np.array_equal(actual, vertices) and np.array_equal(triangles, faces)
    assert metadata["up_axis"] == "z"
    faces[0, 0] = 99
    path.write_text(json.dumps({"vertices": vertices.tolist(), "faces": faces.tolist()}))
    with pytest.raises(ValueError, match="outside"):
        load_mesh(path)


def test_real_glb_scene_transforms_are_baked(tmp_path, tetrahedron):
    vertices, faces = tetrahedron
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    transform = np.eye(4); transform[:3, 3] = [3, 4, 5]
    scene = trimesh.Scene()
    scene.add_geometry(mesh, node_name="translated", transform=transform)
    path = tmp_path / "scene.glb"; scene.export(path)
    actual, triangles, metadata = load_mesh(path)
    _, audit = audit_uv_split(vertices + [3, 4, 5], faces, actual, triangles)
    assert audit["status"] == "PASS" and metadata["scene_nodes_baked"] == 1


def test_obj_polygons_are_not_silently_triangulated(tmp_path):
    path = tmp_path / "quad.obj"
    path.write_text("v 0 0 0\nv 1 0 0\nv 1 1 0\nv 0 1 0\nf 1 2 3 4\n")
    with pytest.raises(ValueError, match="already be triangles"):
        load_mesh(path)


def test_texture_export_restores_source_geometry_and_roundtrips_glb(tmp_path, tetrahedron):
    vertices, faces = tetrahedron
    # Include an isolated source vertex: export must retain it without adding
    # a face. This catches silent mesh processing during the actual GLB write.
    vertices = np.concatenate([vertices + [10, 20, 30], [[14., 25., 36.]]])
    normalized = (vertices - [10, 20, 30]) / 20 - .25
    split, triangles = split_by_face(normalized, faces)
    upstream_axes = np.column_stack([split[:, 0], split[:, 2], -split[:, 1]])
    material = trimesh.visual.material.PBRMaterial(baseColorTexture=Image.new("RGBA", (16, 16), (80, 140, 180, 255)),
        metallicRoughnessTexture=Image.new("RGB", (16, 16), (0, 200, 50)))
    baked = trimesh.Trimesh(vertices=upstream_axes, faces=triangles, process=False,
        visual=trimesh.visual.TextureVisuals(uv=np.tile([[0., 0.], [1., 0.], [0., 1.]], (4, 1)), material=material))
    artifacts, audit = export_textured_mesh(tmp_path, vertices, faces, normalized, baked)
    payload = json.loads(Path(artifacts["texture_mesh.json"]).read_text())
    assert audit["glb_roundtrip_oriented_triangles_exact"]
    assert audit["export_vertices_exact_source_coordinates"]
    assert audit["source_vertices_retained_in_export_mapping"]
    assert audit["glb_roundtrip_source_positions_retained"]
    glb_vertices, _, _ = load_mesh(artifacts["textured.glb"])
    assert np.array_equal(np.unique(glb_vertices, axis=0), np.unique(vertices.astype(np.float32).astype(np.float64), axis=0))
    assert set(payload["vertex_map"]) == set(range(len(vertices)))
    assert np.array_equal(np.asarray(payload["vertices"]), vertices[payload["vertex_map"]])
    assert Image.open(artifacts["base_color.png"]).getpixel((0, 0)) == (80, 140, 180, 255)


def test_explicit_camera_is_recorded_and_does_not_modify_source(tetrahedron):
    vertices, faces = tetrahedron; original = vertices.copy()
    aligned, camera = align_mesh(vertices, faces, None, {"camera_angle_x": .35, "distance": 3.5, "azimuth": 30}, up_axis="z")
    assert np.array_equal(vertices, original)
    assert np.abs(aligned).max() < .5
    assert camera["source"] == "supplied" and camera["parameters"]["azimuth"] == 30
    assert camera["silhouette_iou"] is None
    with pytest.raises(ValueError, match="requires"):
        align_mesh(vertices, faces, None, {"distance": 3.5})


def test_missing_camera_and_ambiguous_background_fail_closed(tetrahedron):
    vertices, faces = tetrahedron
    with pytest.raises(ValueError, match="Cannot derive"):
        align_mesh(vertices, faces, None)


def test_alpha_reference_mask_and_no_crop(tmp_path):
    path = tmp_path / "input.png"
    rgba = np.zeros((60, 100, 4), dtype=np.uint8); rgba[15:45, 25:75] = [180, 80, 60, 255]
    Image.fromarray(rgba).save(path)
    reference, mask, policy = prepare_reference(path, size=100)
    assert reference.size == (100, 100) and mask.sum() == 1500
    assert policy["mask_source"] == "supplied_alpha" and policy["crop"] == "none"
    assert reference.getpixel((0, 0)) == (0, 0, 0)


def test_derived_camera_runs_cpu_and_records_fit():
    # A planar rectangular mesh and rectangle mask exercise the actual CPU
    # optimizer without CUDA, proving missing-camera handling isn't a stub.
    vertices = np.array([[-.5, -.3, 0.], [.5, -.3, 0.], [.5, .3, 0.], [-.5, .3, 0.]])
    faces = np.array([[0, 1, 2], [0, 2, 3]])
    mask = np.zeros((64, 64), dtype=bool); mask[16:48, 8:56] = True
    aligned, camera = align_mesh(vertices, faces, mask, fit_size=64)
    assert camera["source"] == "derived_silhouette_registration"
    assert camera["silhouette_iou"] >= .7
    assert np.abs(aligned).max() < .5


def test_native_degenerate_faces_are_restored_in_actual_glb(tmp_path, tetrahedron):
    vertices, faces = tetrahedron
    vertices = np.concatenate([vertices, [[.5, 0., 0.], [2., 2., 2.]]])
    original_faces = np.concatenate([faces, [[0, 4, 1], [2, 2, 3]]])
    source = tmp_path / "source.json"
    source.write_text(json.dumps({"vertices": vertices.tolist(), "faces": original_faces.tolist()}))
    loaded, triangles, _ = load_mesh(source)
    assert np.array_equal(triangles, original_faces)
    keep = surface_face_mask(loaded, triangles)
    assert np.array_equal(keep, [True, True, True, True, False, False])
    aligned = vertices / 5 - .2
    split, baked_faces = split_by_face(aligned, faces)
    axes = np.column_stack([split[:, 0], split[:, 2], -split[:, 1]])
    material = trimesh.visual.material.PBRMaterial(baseColorTexture=Image.new("RGBA", (16, 16), (70, 120, 180, 255)),
        metallicRoughnessTexture=Image.new("RGB", (16, 16), (0, 190, 40)))
    baked = trimesh.Trimesh(vertices=axes, faces=baked_faces, process=False,
        visual=trimesh.visual.TextureVisuals(uv=np.tile([[0., 0.], [1., 0.], [0., 1.]], (4, 1)), material=material))
    artifacts, audit = export_textured_mesh(tmp_path / "out", vertices, original_faces, aligned, baked, inference_face_mask=keep)
    final_vertices, final_faces, _ = load_mesh(artifacts["textured.glb"])
    _, readback = audit_uv_split(vertices, original_faces, final_vertices, final_faces)
    assert readback["oriented_triangle_multiplicity_exact"] and len(final_faces) == 6
    assert audit["texture_inference_excluded_degenerate_faces"] == 2
    assert audit["restored_native_faces_count"] == 2
    assert audit["glb_roundtrip_source_positions_retained"]
    payload = json.loads(Path(artifacts["texture_mesh.json"]).read_text())
    assert np.array_equal(np.asarray(payload["source_faces"]), original_faces)
    assert set(payload["vertex_map"]) == set(range(len(vertices)))


def test_all_degenerate_surface_is_rejected_and_repeated_indices_are_canonical(tmp_path):
    vertices = np.array([[0., 0., 0.], [1., 0., 0.], [2., 0., 0.]])
    faces = np.array([[0, 1, 2], [0, 0, 1]])
    assert not surface_face_mask(vertices, faces).any()
    _, audit = audit_uv_split(vertices, faces, vertices, np.array([[1, 2, 0], [1, 0, 0]]))
    assert audit["oriented_triangle_multiplicity_exact"]
    baked = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    with pytest.raises(ValueError, match="no nondegenerate surface"):
        export_textured_mesh(tmp_path, vertices, faces, vertices, baked)
