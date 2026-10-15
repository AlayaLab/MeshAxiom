"""CPU HTTP/tool contracts with real loopback I/O and a synthetic backend.

These tests never load a checkpoint or assert GPU/model quality acceptance.
"""
from __future__ import annotations

import hashlib
import http.client
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
from types import SimpleNamespace

import pytest
from PIL import Image
from jsonschema import Draft202012Validator

from meshaxiom.config import load_config
from meshaxiom.contracts import MeshAxiomError, file_binding
import meshaxiom.server as server_module
from meshaxiom.server import JobManager, make_server
from meshaxiom.tools import execute_tool


@pytest.fixture
def live_api(tmp_path):
    state = SimpleNamespace(calls=[], started=threading.Event(), release=threading.Event())
    state.release.set()

    class SyntheticPipeline:
        def __init__(self, config):
            self.config = config

        def _result(self, operation, image, output_dir, mesh=None, **options):
            state.calls.append({"operation": operation, "image": image, "mesh": mesh, "options": options})
            state.started.set()
            if not state.release.wait(10):
                raise RuntimeError("Synthetic test backend timed out")
            if options.get("seed") == 19:
                raise MeshAxiomError("Synthetic backend failure")
            output = Path(output_dir)
            output.mkdir(parents=True, exist_ok=True)
            artifact = output / ("mesh.obj" if operation == "geometry" else "enhanced.png" if operation == "superresolution" else "textured.glb")
            if operation == "superresolution":
                # A dummy CPU PNG tests transport only, never Real-ESRGAN.
                Image.new("RGB", (4, 4), (20, 40, 60)).save(artifact)
            else:
                artifact.write_bytes(b"synthetic-" + operation.encode() + b"\n" + Path(image).read_bytes())
            artifacts = {"enhanced.png" if operation == "superresolution" else "mesh": file_binding(artifact)}
            if operation == "material":
                provenance = output / "provenance.json"
                provenance.write_text('{"backend":"synthetic CPU contract"}\n')
                artifacts["provenance.json"] = file_binding(provenance)
            return {"schema_version": 1, "status": "complete", "operation": operation,
                    "output_dir": str(output), "artifacts": artifacts}

        def generate(self, image, output_dir, **options):
            return self._result("geometry", image, output_dir, **options)

        def texture(self, mesh, image, output_dir, **options):
            return self._result("material", image, output_dir, mesh=mesh, **options)

        def enhance(self, image, output_dir):
            return self._result("superresolution", image, output_dir)

    config = load_config({"schema_version": 1})
    manager = JobManager(config, tmp_path / "workspace", pipeline_factory=SyntheticPipeline)
    server = make_server(config, manager.workspace, port=0, manager=manager)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    def request(method, path, body=None, headers=None):
        connection = http.client.HTTPConnection("127.0.0.1", server.server_port, timeout=5)
        try:
            connection.request(method, path, body=body, headers=headers or {})
            response = connection.getresponse()
            return response.status, dict(response.getheaders()), response.read()
        finally:
            connection.close()

    def json_request(method, path, payload=None, headers=None):
        body = None if payload is None else json.dumps(payload).encode()
        status, response_headers, data = request(method, path, body, {"Content-Type": "application/json", **(headers or {})})
        return status, response_headers, json.loads(data)

    def upload(filename, content):
        status, _, data = request("POST", "/v1/uploads", content, {"X-Filename": filename, "Content-Type": "application/octet-stream"})
        assert status == 201
        return json.loads(data)["upload_id"]

    def wait_job(job_id):
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            status, _, job = json_request("GET", "/v1/jobs/" + job_id)
            assert status == 200
            if job["status"] in {"complete", "failed"}:
                return job
            time.sleep(.01)
        pytest.fail("Loopback test job did not finish")

    api = SimpleNamespace(request=request, json=json_request, upload=upload,
                          wait_job=wait_job, manager=manager, state=state, port=server.server_port)
    try:
        yield api
    finally:
        state.release.set()
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
        manager.close()


def test_upload_geometry_then_independent_material_and_output_sha(live_api):
    api = live_api
    source = b"synthetic image input; backend intentionally avoids decoding"
    upload_id = api.upload("../../reference.PNG", source)
    assert upload_id.endswith(".png") and "/" not in upload_id
    assert (api.manager.uploads / upload_id).read_bytes() == source

    status, _, accepted = api.json("POST", "/v1/jobs", {
        "operation": "geometry", "image_upload": upload_id, "face_budget": 4200,
        "seed": 7, "background": "keep"})
    assert status == 202
    geometry = api.wait_job(accepted["id"])
    assert geometry["status"] == "complete"
    binding = geometry["result"]["artifacts"]["mesh"]
    status, _, downloaded = api.request("GET", binding["download_url"])
    assert status == 200
    assert len(downloaded) == binding["size"]
    assert hashlib.sha256(downloaded).hexdigest() == binding["sha256"]
    assert api.state.calls[0]["options"] == {"seed": 7, "face_budget": 4200, "background": "keep"}

    status, _, accepted = api.json("POST", "/v1/jobs", {
        "operation": "material", "image_upload": upload_id, "mesh_path": binding["path"],
        "seed": 42, "texture_size": 1024, "camera": {"distance": 3.0}})
    assert status == 202
    material = api.wait_job(accepted["id"])
    assert material["status"] == "complete"
    assert [row["operation"] for row in api.state.calls] == ["geometry", "material"]
    assert api.state.calls[1]["mesh"] == binding["path"]
    assert api.state.calls[1]["options"] == {"seed": 42, "texture_size": 1024, "camera": {"distance": 3.0}}
    assert "provenance.json" in material["result"]["artifacts"]
    for material_binding in material["result"]["artifacts"].values():
        status, _, downloaded = api.request("GET", material_binding["download_url"])
        assert status == 200 and len(downloaded) == material_binding["size"]
        assert hashlib.sha256(downloaded).hexdigest() == material_binding["sha256"]


def test_mesh_upload_dispatch_and_persisted_failure(live_api):
    api = live_api
    image = api.upload("input.png", b"image")
    mesh = api.upload("triangle.obj", b"v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n")
    status, _, accepted = api.json("POST", "/v1/jobs", {
        "operation": "material", "image_upload": image, "mesh_upload": mesh, "seed": 19})
    assert status == 202
    failed = api.wait_job(accepted["id"])
    assert failed["status"] == "failed"
    assert failed["error"] == "Synthetic backend failure"
    assert failed["error_type"] == "MeshAxiomError"
    assert "result" not in failed
    assert api.state.calls[0]["mesh"] == str(api.manager.uploads / mesh)
    record = json.loads((api.manager.workspace / "records" / (failed["id"] + ".json")).read_text())
    assert record == failed
    assert api.request("GET", f"/v1/jobs/{failed['id']}/artifacts/mesh")[0] == 404


def test_independent_superresolution_http_dispatch_and_png_download(live_api):
    api = live_api
    source = io.BytesIO()
    Image.new("RGB", (2, 2), (20, 40, 60)).save(source, format="PNG")
    image = api.upload("atlas.png", source.getvalue())
    status, _, accepted = api.json("POST", "/v1/jobs", {"operation": "superresolution", "image_upload": image})
    assert status == 202
    job = api.wait_job(accepted["id"])
    assert job["status"] == "complete" and job["operation"] == "superresolution"
    assert api.state.calls == [{"operation": "superresolution", "image": str(api.manager.uploads / image),
                                "mesh": None, "options": {}}]
    assert set(job["result"]["artifacts"]) == {"enhanced.png"}
    binding = job["result"]["artifacts"]["enhanced.png"]
    status, headers, data = api.request("GET", binding["download_url"])
    assert status == 200 and headers["Content-Type"] == "image/png"
    assert len(data) == binding["size"] and hashlib.sha256(data).hexdigest() == binding["sha256"]
    with Image.open(io.BytesIO(data)) as dummy:
        assert dummy.format == "PNG" and dummy.size == (4, 4)
    # Enhancement is independent of geometry/material and has no sampler seed.
    status, _, error = api.json("POST", "/v1/jobs", {"operation": "superresolution", "image_upload": image, "seed": 42})
    assert status == 400 and error["status"] == "failed"


def test_artifact_download_denies_traversal_external_and_symlink_paths(live_api, tmp_path):
    api = live_api
    image = api.upload("image.png", b"image")
    status, _, accepted = api.json("POST", "/v1/jobs", {"operation": "geometry", "image_upload": image})
    assert status == 202
    job = api.wait_job(accepted["id"])
    outside = tmp_path / "outside.txt"
    outside.write_bytes(b"must never be served")
    link = Path(job["output_dir"]) / "escape.txt"
    link.symlink_to(outside)
    result = job["result"]
    result["artifacts"]["outside"] = file_binding(outside)
    result["artifacts"]["symlink"] = {"path": str(link), "size": outside.stat().st_size,
                                      "sha256": hashlib.sha256(outside.read_bytes()).hexdigest()}
    api.manager.update(job["id"], result=result)
    prefix = f"/v1/jobs/{job['id']}/artifacts/"
    for artifact in ["outside", "symlink", "../outside.txt", "%2e%2e%2foutside.txt", "mesh/../../outside.txt"]:
        assert api.request("GET", prefix + artifact)[0] == 404
    assert api.request("GET", "/v1/jobs/" + "0" * 32)[0] == 404


def test_artifact_changed_after_completion_is_not_served_with_stale_binding(live_api):
    api = live_api
    image = api.upload("image.png", b"image")
    status, _, accepted = api.json("POST", "/v1/jobs", {"operation": "geometry", "image_upload": image})
    assert status == 202
    job = api.wait_job(accepted["id"])
    artifact = job["result"]["artifacts"]["mesh"]
    path = Path(artifact["path"])
    original = path.read_bytes()
    # Same byte length exercises SHA integrity rather than only a size guard.
    path.write_bytes(b"X" + original[1:])
    assert hashlib.sha256(path.read_bytes()).hexdigest() != artifact["sha256"]
    status, _, body = api.request("GET", artifact["download_url"])
    assert status in {404, 409}
    assert json.loads(body).get("error")


def test_queue_accepts_eight_active_jobs_then_refuses_ninth(live_api):
    api = live_api
    image = api.upload("image.png", b"image")
    api.state.release.clear()
    accepted = []
    for seed in range(8):
        status, _, job = api.json("POST", "/v1/jobs", {"operation": "geometry", "image_upload": image, "seed": seed})
        assert status == 202
        accepted.append(job["id"])
    assert api.state.started.wait(2)
    status, _, refusal = api.json("POST", "/v1/jobs", {"operation": "geometry", "image_upload": image})
    assert status == 400 and "Queue is full" in refusal["error"]
    assert len(api.manager.jobs) == 8
    api.state.release.set()
    assert all(api.wait_job(job_id)["status"] == "complete" for job_id in accepted)


@pytest.mark.parametrize("extra", [
    {"operation": []}, {"background": {}}, {"operation": "material", "camera": {"up_axis": []}},
    {"seed": True}, {"face_budget": 0}, {"recipe": "170K"},
])
def test_invalid_json_option_types_are_json_errors_and_do_not_enqueue(live_api, extra):
    api = live_api
    image = api.upload("image.png", b"image")
    status, _, error = api.json("POST", "/v1/jobs", {"operation": "geometry", "image_upload": image, **extra})
    assert status == 400
    assert error["status"] == "failed" and error["error"]
    assert api.manager.jobs == {}


def test_upload_limits_invalid_ids_and_cross_origin(live_api):
    api = live_api
    for name, content in [("script.py", b"print(1)"), ("empty.png", b"")]:
        status, _, error = api.request("POST", "/v1/uploads", content, {"X-Filename": name})
        assert status == 400 and json.loads(error)["status"] == "failed"
    image = api.upload("image.png", b"image")
    status, _, error = api.json("POST", "/v1/jobs", {"operation": "geometry", "image_upload": image},
                                {"Origin": "http://unrelated.example"})
    assert status == 403 and "Cross-origin" in error["error"]
    for token in ["../../private.png", "f" * 32 + ".png"]:
        status, _, error = api.json("POST", "/v1/jobs", {"operation": "geometry", "image_upload": token})
        assert status == 400


def test_http_declared_upload_and_json_size_limits(live_api, monkeypatch):
    api = live_api
    monkeypatch.setattr(server_module, "MAX_UPLOAD", 8)
    status, _, body = api.request("POST", "/v1/uploads", b"123456789", {"X-Filename": "image.png"})
    assert status == 400 and "size limit" in json.loads(body)["error"]
    monkeypatch.setattr(server_module, "MAX_REQUEST", 8)
    status, _, body = api.request("POST", "/v1/jobs", b'{"operation":"geometry"}')
    assert status == 400 and "size limit" in json.loads(body)["error"]
    assert list(api.manager.uploads.iterdir()) == [] and api.manager.jobs == {}


def test_published_schemas_and_page_match_single_recipe(live_api):
    api = live_api
    status, _, health = api.json("GET", "/health")
    assert status == 200 and health["model_loading"] == "lazy_worker_processes"
    status, _, doctor = api.json("GET", "/v1/doctor")
    assert status == 200 and doctor["models_loaded"] is False and doctor["status"] == "setup_required"
    status, _, openapi = api.json("GET", "/openapi.json")
    assert status == 200 and openapi["openapi"] == "3.1.0"
    assert {"/v1/uploads", "/v1/jobs", "/v1/jobs/{job_id}", "/v1/jobs/{job_id}/artifacts/{artifact}"} <= set(openapi["paths"])
    status, _, tools = api.json("GET", "/agent-tools.json")
    assert status == 200 and {item["name"] for item in tools["tools"]} == {"generate_mesh", "texture_mesh", "enhance_texture"}
    assert "170" not in json.dumps(openapi) + json.dumps(tools)
    status, headers, page = api.request("GET", "/")
    assert status == 200 and "text/html" in headers["Content-Type"]
    assert b"/v1/uploads" in page and b"/v1/jobs" in page


def test_superresolution_openapi_validation_matches_http_source_contract(live_api):
    api = live_api
    image = api.upload("atlas.png", b"synthetic source for HTTP schema contract")
    image_path = str(api.manager.uploads / image)
    status, _, openapi = api.json("GET", "/openapi.json")
    assert status == 200
    schema = openapi["paths"]["/v1/jobs"]["post"]["requestBody"]["content"]["application/json"]["schema"]
    Draft202012Validator.check_schema(schema)
    validator = Draft202012Validator(schema)
    valid = [{"operation": "superresolution", "image_upload": image},
             {"operation": "superresolution", "image_path": image_path}]
    invalid = [
        {"operation": "superresolution"},
        {"operation": "superresolution", "image_path": image_path, "image_upload": image},
        {"operation": "superresolution", "image_path": image_path, "image_upload": ""},
        {"operation": "superresolution", "image_path": "", "image_upload": image},
        {"operation": "superresolution", "image_path": ""},
        {"operation": "superresolution", "image_upload": ""},
        {"operation": "superresolution", "image_upload": "../atlas.png"},
        {"operation": "superresolution", "image_upload": "f" * 32 + ".obj"},
        {"operation": "superresolution", "image_upload": image, "seed": 42},
        {"operation": "superresolution", "image_upload": image, "face_budget": 8000},
        {"operation": "superresolution", "image_upload": image, "background": "auto"},
        {"operation": "superresolution", "image_upload": image, "texture_size": 2048},
        {"operation": "superresolution", "image_upload": image, "camera": {}},
        {"operation": "superresolution", "image_upload": image, "mesh_path": "mesh.obj"},
        {"operation": "superresolution", "image_upload": image, "mesh_upload": image},
    ]
    for payload in valid:
        assert validator.is_valid(payload), payload
        status, _, accepted = api.json("POST", "/v1/jobs", payload)
        assert status == 202, payload
        assert api.wait_job(accepted["id"])["status"] == "complete"
    count = len(api.manager.jobs)
    for payload in invalid:
        assert not validator.is_valid(payload), payload
        status, _, error = api.json("POST", "/v1/jobs", payload)
        assert status == 400 and error["status"] == "failed", payload
    assert len(api.manager.jobs) == count
    # The conditional only narrows enhancement; model operations retain their
    # existing fields rather than being limited to enhancement's image-only API.
    assert validator.is_valid({"operation": "geometry", "image_upload": image, "seed": 7, "face_budget": 4000})
    assert validator.is_valid({"operation": "material", "image_upload": image, "mesh_path": "mesh.obj", "seed": 42, "camera": {}})


def test_agent_tool_dispatches_without_hidden_stage_or_recipe(tmp_path):
    calls = []

    class Recorder:
        def generate(self, **arguments):
            calls.append(("geometry", arguments))
            return {"status": "complete", "operation": "geometry"}

        def texture(self, **arguments):
            calls.append(("material", arguments))
            return {"status": "complete", "operation": "material"}

        def enhance(self, **arguments):
            calls.append(("superresolution", arguments))
            return {"status": "complete", "operation": "superresolution"}

    pipeline = Recorder()
    geometry = {"image": str(tmp_path / "image.png"), "output_dir": str(tmp_path / "mesh"), "face_budget": 5000}
    material = {"mesh": str(tmp_path / "mesh.glb"), "image": geometry["image"], "output_dir": str(tmp_path / "texture")}
    enhancement = {"image": str(tmp_path / "atlas.png"), "output_dir": str(tmp_path / "enhanced")}
    assert execute_tool(pipeline, {"name": "generate_mesh", "arguments": geometry})["operation"] == "geometry"
    assert execute_tool(pipeline, {"name": "texture_mesh", "arguments": material})["operation"] == "material"
    assert execute_tool(pipeline, {"name": "enhance_texture", "arguments": enhancement})["operation"] == "superresolution"
    assert calls == [("geometry", geometry), ("material", material), ("superresolution", enhancement)]
    with pytest.raises(MeshAxiomError):
        execute_tool(pipeline, {"name": "generate_mesh", "arguments": {**geometry, "recipe": "170K"}})


def test_cli_malformed_nested_config_has_structured_failure(tmp_path):
    config = tmp_path / "bad-config.json"
    config.write_text(json.dumps({"schema_version": 1, "geometry": {"image_model": ["invalid"]}}))
    env = os.environ.copy()
    source = str(Path(__file__).resolve().parents[1] / "src")
    env["PYTHONPATH"] = source + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    completed = subprocess.run([sys.executable, "-m", "meshaxiom.cli", "doctor", "--config", str(config)],
                               env=env, capture_output=True, text=True, timeout=10)
    assert completed.returncode == 1 and not completed.stdout
    error = json.loads(completed.stderr)
    assert error["status"] == "failed" and error["error_type"] == "MeshAxiomError"
