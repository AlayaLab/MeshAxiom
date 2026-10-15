"""CPU orchestration contracts; these tests do not simulate model quality."""
import fcntl
import json
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch
from PIL import Image
from meshaxiom import MeshAxiomPipeline
from meshaxiom.config import load_config, doctor
from meshaxiom.contracts import MeshAxiomError, file_binding, validate_options, write_json
from meshaxiom.preprocess import prepare_image
from meshaxiom.tools import execute_tool

class PipelineContracts(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.image = self.root / "input.png"
        Image.new("RGB", (64, 32), (60, 80, 100)).save(self.image)
        self.config = {"schema_version": 1, "runtime": {"device": "cuda:0", "gpu_lock_directory": str(self.root / "locks"), "lock_timeout_seconds": 0}}
    def tearDown(self):
        self.temporary.cleanup()

    def test_segmented_rgba_is_centered_and_shared(self):
        image = Image.new("RGBA", (200, 100))
        image.paste((220, 80, 10, 255), (40, 20, 160, 80))
        image.save(self.image)
        report = prepare_image(self.image, self.root / "ready.png")
        self.assertEqual(report["background_method"], "supplied_alpha")
        self.assertTrue(report["shared_by_vertex_and_topology"])
        with Image.open(self.root / "ready.png") as prepared:
            self.assertEqual(prepared.size, (512, 512))
            self.assertEqual(prepared.getpixel((0, 0)), (0, 0, 0))

    def test_bad_config_types_have_structured_errors(self):
        for value in [None, [], {"schema_version": 1, "runtime": []}, {"schema_version": 1, "geometry": []}]:
            with self.subTest(value=value), self.assertRaises(MeshAxiomError):
                load_config(value)

    def test_doctor_checks_separate_topology_runtime_and_empty_error_output(self):
        config = load_config(self.config)
        config["runtime"]["vertex_python"] = "/missing/vertex-python"
        config["runtime"]["topology_python"] = "/missing/topology-python"
        with patch("meshaxiom.config.check_assets", return_value=[]), patch("meshaxiom.config.subprocess.run", side_effect=[SimpleNamespace(returncode=0, stderr=""), SimpleNamespace(returncode=2, stderr="")]) as probe:
            report = doctor(config, "geometry", check_runtime=True)
        self.assertEqual([call.args[0][0] for call in probe.call_args_list], ["/missing/vertex-python", "/missing/topology-python"])
        self.assertEqual(report["status"], "setup_required")
        self.assertTrue(any("topology_python runtime imports failed: process exited with code 2" in error for error in report["errors"]))
        self.assertFalse(report["models_loaded"])

    def test_config_relative_worker_interpreter_uses_config_directory(self):
        configs = self.root / "configs"
        configs.mkdir()
        interpreter = self.root / "environments/material/bin/python"
        interpreter.parent.mkdir(parents=True)
        interpreter.write_text("#!/bin/sh\nexit 0\n")
        interpreter.chmod(0o755)
        topology_python = self.root / "environments/topology/bin/python"
        topology_python.parent.mkdir(parents=True)
        topology_python.symlink_to(interpreter)
        config_path = configs / "material.json"
        config_path.write_text(json.dumps({"schema_version": 1, "runtime": {"material_python": "../environments/material/bin/python", "topology_python": "../environments/topology/bin/python"}}))
        resolved = load_config(config_path)
        self.assertEqual(resolved["runtime"]["material_python"], str(interpreter.resolve()))
        self.assertEqual(resolved["runtime"]["topology_python"], str(topology_python))

    def test_parameters_reject_bool_nonfinite_and_unhashable_values(self):
        for options in [dict(seed=True), dict(seed=-1), dict(seed=2**63), dict(seed=1, background={})]:
            with self.subTest(options=options), self.assertRaises(MeshAxiomError):
                validate_options("geometry", face_budget=8000, **options)
        with self.assertRaises(MeshAxiomError):
            validate_options("material", seed=42, texture_size=2048, camera={"distance": float("nan")})

    def fake_worker(self, stage, request, output):
        if stage == "vertex":
            (output / "vertex-marker").write_text("only vertex namespace")
        else:
            artifact = output / "mesh.json"
            artifact.write_text('{"vertices":[[0,0,0]],"faces":[]}')
            write_json(output / "geometry_report.json", {"status": "COMPLETE", "coordinate_frame": "z_up", "artifacts": {"mesh_json": file_binding(artifact)}})

    def test_completion_requires_both_workers_and_bound_artifacts(self):
        pipeline = MeshAxiomPipeline(self.config)
        seen = []
        def worker(stage, request, output):
            seen.append(stage)
            self.fake_worker(stage, request, output)
        with patch("meshaxiom.pipeline.check_assets", return_value=[]), patch.object(pipeline, "_run_worker", side_effect=worker):
            result = pipeline.generate(self.image, self.root / "job", background="keep")
        self.assertEqual(seen, ["vertex", "topology"])
        self.assertEqual(result["status"], "complete")
        self.assertEqual(result["artifacts"]["mesh_json"]["sha256"], file_binding(self.root / "job/mesh.json")["sha256"])

    def test_worker_failure_is_retained_and_lock_released(self):
        pipeline = MeshAxiomPipeline(self.config)
        with patch("meshaxiom.pipeline.check_assets", return_value=[]), patch.object(pipeline, "_run_worker", side_effect=MeshAxiomError("worker exited")), self.assertRaises(MeshAxiomError):
            pipeline.generate(self.image, self.root / "failed", background="keep")
        failure = json.loads((self.root / "failed/FAILED.json").read_text())
        self.assertEqual(failure["stage"], "vertex")
        self.assertEqual(failure["status"], "failed")
        with pipeline._gpu_lock():
            pass
        with self.assertRaises(MeshAxiomError):
            pipeline.generate(self.image, self.root / "failed", background="keep")

    def test_corrupt_and_external_artifacts_cannot_mark_complete(self):
        for mode in ["corrupt", "external"]:
            pipeline = MeshAxiomPipeline(self.config)
            def worker(stage, request, output):
                if stage == "vertex":
                    return
                artifact = output / "artifact.obj" if mode == "corrupt" else self.image
                if mode == "corrupt":
                    artifact.write_text("original")
                binding = file_binding(artifact)
                if mode == "corrupt":
                    artifact.write_text("modified")
                write_json(output / "geometry_report.json", {"status": "COMPLETE", "artifacts": {"obj": binding}})
            with self.subTest(mode=mode), patch("meshaxiom.pipeline.check_assets", return_value=[]), patch.object(pipeline, "_run_worker", side_effect=worker), self.assertRaises(MeshAxiomError):
                pipeline.generate(self.image, self.root / mode, background="keep")
            self.assertFalse((self.root / mode / "result.json").exists())

    def test_shared_gpu_lock_does_not_steal_existing_owner(self):
        pipeline = MeshAxiomPipeline(self.config)
        folder = self.root / "locks"
        folder.mkdir()
        with (folder / "cuda-0.lock").open("a") as owner:
            fcntl.flock(owner, fcntl.LOCK_EX)
            with self.assertRaises(MeshAxiomError):
                with pipeline._gpu_lock():
                    self.fail("existing owner was bypassed")

    def test_capacity_guard_fails_before_model_or_input_preparation(self):
        config = json.loads(json.dumps(self.config))
        config["runtime"]["minimum_free_workspace_gib"] = 10**9
        pipeline = MeshAxiomPipeline(config)
        with self.assertRaises(MeshAxiomError):
            pipeline.generate(self.image, self.root / "capacity", background="keep")
        self.assertFalse((self.root / "capacity/input.png").exists())
        self.assertEqual(json.loads((self.root / "capacity/FAILED.json").read_text())["stage"], "preflight")

    def test_tool_names_and_arguments_fail_before_execution(self):
        pipeline = MeshAxiomPipeline(self.config)
        for call in [{"name": [], "arguments": {}}, {"name": "generate_mesh", "arguments": {"image": [], "output_dir": "x"}}, {"name": "generate_mesh", "arguments": {"image": "x", "output_dir": "y", "checkpoint": "170K"}}]:
            with self.subTest(call=call), self.assertRaises(MeshAxiomError):
                execute_tool(pipeline, call)

if __name__ == "__main__":
    unittest.main()
