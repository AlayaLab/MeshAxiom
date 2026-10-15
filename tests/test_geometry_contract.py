"""CPU checks for real source isolation, integrity failures and lossless exports."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np

from meshaxiom.vendor.recipe import mesh_utils90
from meshaxiom.vendor.recipe.runtime import checkpoint, read_request, write_glb
from meshaxiom.workers.topology import write_mesh_exports

ROOT = Path(__file__).resolve().parents[1]
VENDOR = ROOT / "src/meshaxiom/vendor"


def exporter():
    spec = importlib.util.spec_from_file_location("release_export_weights", ROOT / "scripts/export_weights.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class GeometryContractTests(unittest.TestCase):
    def test_every_packaged_frozen_source_matches_manifest(self):
        manifest = json.loads((VENDOR / "recipe/SOURCE_MANIFEST.json").read_text())
        self.assertGreaterEqual(len(manifest["files"]), 30)
        for entry in manifest["files"]:
            self.assertEqual(hashlib.sha256((VENDOR / entry["packaged_file"]).read_bytes()).hexdigest(), entry["packaged_sha256"])

    def test_both_real_architectures_import_without_internal_checkout(self):
        for stage, imports in (
            ("vertex", "from meshaxiom_core.models.octree_dit import VertexModel; from meshaxiom.vendor.recipe.gen_mbr128 import sample_mbr"),
            ("topology", "from meshaxiom_core.models.topo_dit import TopologyModel; from meshaxiom_core.models.topo_vae import build_vae; from meshaxiom.vendor.recipe.model118rich import TopologyModel113"),
        ):
            code = f"from meshaxiom.vendor.recipe.runtime import activate_namespace; activate_namespace('{stage}'); {imports}; import meshaxiom_core; print(meshaxiom_core.__file__)"
            env = dict(os.environ, PYTHONPATH=str(ROOT / "src"), PYTHONDONTWRITEBYTECODE="1")
            result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env, timeout=90, cwd="/")
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn(f"vendor/{stage}/meshaxiom", result.stdout)

    def test_worker_namespaces_cannot_be_mixed(self):
        code = "from meshaxiom.vendor.recipe.runtime import activate_namespace; activate_namespace('vertex'); import meshaxiom_core; activate_namespace('topology')"
        result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                                env=dict(os.environ, PYTHONPATH=str(ROOT / "src")), timeout=10)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("separate, clean worker", result.stderr)

    def test_sha_and_actual_step_reject_different_checkpoint(self):
        import torch
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "checkpoint.pt"
            torch.save({"step": 180000, "cfg": {"model": {}}, "ema": {"weight": torch.arange(4)}}, path)
            sha = hashlib.sha256(path.read_bytes()).hexdigest()
            spec = {"path": str(path), "sha256": sha, "step": 180000, "weights": "ema"}
            _, report = checkpoint(spec, "vertex")
            self.assertEqual(report["step"], 180000)
            with self.assertRaisesRegex(ValueError, "actual step"):
                checkpoint(dict(spec, step=10000), "vertex")
            with self.assertRaisesRegex(ValueError, "SHA256"):
                checkpoint(dict(spec, sha256="0" * 64), "vertex")

    def test_export_removes_optimizer_and_preserves_every_ema_tensor(self):
        import torch
        export = exporter()
        with tempfile.TemporaryDirectory() as folder:
            source, output = Path(folder) / "source.pt", Path(folder) / "inference.pt"
            cfg = {
                "depth": 9, "model": {"dim": 32}, "condition": {"type": "multi", "fusion": "concat", "image_size": 512},
                "data": {"depth": 9, "image_background": [0, 0, 0], "root": "/private/training"},
                "train": {"optimizer": "private", "sample_latent": True},
            }
            state = {"condition.backbone.weight": torch.arange(12).reshape(3, 4).float(), "head.weight": torch.ones(2, 3)}
            torch.save({"step": 180000, "cfg": cfg, "ema": state, "model": {"bad": torch.ones(1)}, "opt": {"data": torch.ones(4)}, "rng": "private"}, source)
            source_bytes = source.read_bytes()
            receipt = export.export(source, output, "vertex", hashlib.sha256(source_bytes).hexdigest(), 180000)
            result = torch.load(output, map_location="cpu", mmap=True, weights_only=False)
            self.assertTrue(receipt["exact_tensor_roundtrip"])
            self.assertNotIn("model", result)
            self.assertNotIn("opt", result)
            self.assertNotIn("rng", result)
            self.assertNotIn("root", result["cfg"]["data"])
            self.assertEqual(result["cfg"]["condition"], cfg["condition"])
            self.assertEqual(source_bytes, source.read_bytes())
            for name, tensor in state.items():
                self.assertTrue(torch.equal(result["ema"][name], tensor))
            self.assertEqual(receipt["artifact"]["sha256"], hashlib.sha256(output.read_bytes()).hexdigest())
            with self.assertRaises(FileExistsError):
                export.export(source, output, "vertex", hashlib.sha256(source_bytes).hexdigest(), 180000)

    def test_export_orientation_keeps_face_sets_and_order(self):
        vertices = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], np.float64)
        faces = np.array([[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]], np.int64)
        original = faces.copy()
        oriented, report = mesh_utils90.export_faces(vertices, faces)
        np.testing.assert_array_equal(faces, original)
        np.testing.assert_array_equal(np.sort(oriented, axis=1), np.sort(faces, axis=1))
        self.assertEqual(mesh_utils90.mesh_stats(vertices, faces), mesh_utils90.mesh_stats(vertices, oriented))
        self.assertEqual(report["inconsistent_edges_after"], 0)

    def test_glb_preserves_every_vertex_and_export_index(self):
        import struct
        coords = np.array([[128, 128, 128], [400, 128, 128], [128, 400, 128], [128, 128, 400]], np.int16)
        world = (coords.astype(np.float64) + 0.5) / 512 * 2 - 1
        faces = np.array([[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]], np.int64)
        oriented, _ = mesh_utils90.export_faces(world, faces)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "mesh.glb"
            write_glb(path, world, oriented)
            blob = path.read_bytes()
            magic, version, length = struct.unpack_from("<III", blob)
            self.assertEqual((magic, version, length), (0x46546C67, 2, len(blob)))
            json_size, json_kind = struct.unpack_from("<II", blob, 12)
            self.assertEqual(json_kind, 0x4E4F534A)
            document = json.loads(blob[20:20 + json_size])
            binary_size, binary_kind = struct.unpack_from("<II", blob, 20 + json_size)
            self.assertEqual(binary_kind, 0x004E4942)
            binary = blob[28 + json_size:]
            self.assertEqual(binary_size, len(binary))
            position_view, normal_view, index_view = document["bufferViews"]
            restored = np.frombuffer(binary, dtype="<f4", count=world.size, offset=position_view["byteOffset"]).reshape(-1, 3)
            restored_faces = np.frombuffer(binary, dtype="<u4", count=oriented.size, offset=index_view["byteOffset"]).reshape(-1, 3)
            normals = np.frombuffer(binary, dtype="<f4", count=world.size, offset=normal_view["byteOffset"]).reshape(-1, 3)
            np.testing.assert_array_equal(restored, world)
            np.testing.assert_array_equal(restored_faces, oriented)
            self.assertTrue(np.isfinite(normals).all())
            node = document["nodes"][0]
            self.assertEqual(node["extras"]["vertex_storage_up_axis"], "z")
            self.assertEqual(node["extras"]["world_up_axis"], "y")
            # glTF quaternions are XYZW; the active node rotation must map model Z to scene Y.
            x, y, z, w = node["rotation"]
            rotation = np.array([
                [1 - 2 * (y*y + z*z), 2 * (x*y - z*w), 2 * (x*z + y*w)],
                [2 * (x*y + z*w), 1 - 2 * (x*x + z*z), 2 * (y*z - x*w)],
                [2 * (x*z - y*w), 2 * (y*z + x*w), 1 - 2 * (x*x + y*y)],
            ])
            expected_world = world[:, [0, 2, 1]] * np.array([1, 1, -1])
            np.testing.assert_allclose(restored @ rotation.T, expected_world, atol=1e-15, rtol=0)
            np.testing.assert_allclose(rotation @ np.array([0, 0, 1]), [0, 1, 0], atol=1e-15, rtol=0)
            self.assertAlmostEqual(np.linalg.det(rotation), 1.0)

    def test_d9_obj_and_json_coordinate_serialization_is_exact(self):
        # Exercise every D9 cell center, including fractions lost by the legacy JSON round(7).
        grid = np.arange(512, dtype=np.int16)
        coords = np.stack([grid, np.roll(grid, 37), np.roll(grid, 113)], axis=1)
        world = (coords.astype(np.float64) + 0.5) / 512 * 2 - 1
        faces = np.array([[0, 1, 2], [100, 210, 300], [510, 511, 0]], np.int64)
        with tempfile.TemporaryDirectory() as folder:
            out = Path(folder)
            write_mesh_exports(out, world, faces, "D9 exact serialization contract")
            document = json.loads((out / "mesh.json").read_text())
            np.testing.assert_array_equal(np.asarray(document["vertices"]), world)
            np.testing.assert_array_equal(np.asarray(document["faces"]), faces)
            vertices, triangles = [], []
            for line in (out / "mesh.obj").read_text().splitlines():
                if line.startswith("v "):
                    vertices.append([float(value) for value in line.split()[1:]])
                elif line.startswith("f "):
                    triangles.append([int(value) - 1 for value in line.split()[1:]])
            np.testing.assert_array_equal(np.asarray(vertices), world)
            np.testing.assert_array_equal(np.asarray(triangles), faces)
            self.assertEqual(document["up_axis"], "z")

    def test_ready_image_and_complete_config_are_required(self):
        with tempfile.TemporaryDirectory() as folder:
            tmp = Path(folder)
            image = tmp / "input.png"
            image.write_bytes(b"placeholder")
            spec = {"path": str(tmp / "weight.pt"), "sha256": "a" * 64, "step": 180000, "weights": "ema"}
            request = {"config": {"geometry": {key: spec for key in ("vertex_checkpoint", "topology_checkpoint", "vae_checkpoint")}},
                       "image": str(image), "output_dir": str(tmp / "output")}
            path = tmp / "request.json"
            path.write_text(json.dumps(request))
            resolved = read_request(path)
            self.assertEqual(resolved["steps"], 20)
            self.assertEqual(resolved["face_budget"], 8000)
            request["image"] = "input.png"
            path.write_text(json.dumps(request))
            with self.assertRaisesRegex(ValueError, "absolute"):
                read_request(path)


if __name__ == "__main__":
    unittest.main()
