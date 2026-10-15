"""Loopback upload UI and bounded asynchronous agent jobs; no GPU web process."""
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import hashlib
import mimetypes
import os
from pathlib import Path
import re
import threading
from urllib.parse import urlparse
import uuid
from .config import doctor
from .contracts import MeshAxiomError, utc_now, validate_options, write_json
from .pipeline import MeshAxiomPipeline

MAX_UPLOAD = 64 * 1024 * 1024
MAX_REQUEST = 64 * 1024

class JobManager:
    def __init__(self, config, workspace, *, pipeline_factory=MeshAxiomPipeline):
        self.config = config
        self.workspace = Path(workspace).expanduser().resolve()
        self.workspace.mkdir(parents=True, exist_ok=True)
        self.uploads = self.workspace / "uploads"
        self.uploads.mkdir(exist_ok=True)
        self.jobs = {}
        self.lock = threading.Lock()
        self.pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="meshaxiom")
        self.pipeline_factory = pipeline_factory
        self._pipeline = None

    def resolve_input(self, request, key, extensions):
        token = request.get(key + "_upload")
        path = request.get(key + "_path")
        if (key + "_upload" in request) == (key + "_path" in request):
            raise MeshAxiomError(f"Supply exactly one {key}_upload or {key}_path")
        if token:
            if not isinstance(token, str) or not re.fullmatch(r"[a-f0-9]{32}\.[a-z0-9]+", token):
                raise MeshAxiomError("Invalid upload ID")
            p = self.uploads / token
        else:
            if not isinstance(path, str):
                raise MeshAxiomError("Input path must be a string")
            p = Path(path).expanduser().resolve()
        if not p.is_file() or p.suffix.lower() not in extensions:
            raise MeshAxiomError("Missing or unsupported " + key + " input")
        return str(p)

    def submit(self, request):
        if not isinstance(request, dict):
            raise MeshAxiomError("Job request must be a JSON object")
        allowed = {"operation", "image_upload", "image_path", "mesh_upload", "mesh_path", "face_budget", "seed", "background", "texture_size", "camera"}
        if set(request) - allowed:
            raise MeshAxiomError("Unknown job field: " + ", ".join(sorted(set(request) - allowed)))
        operation = request.get("operation", "geometry")
        seed = request.get("seed", 2026082301 if operation == "geometry" else 42)
        options = {"seed": seed}
        if operation == "geometry":
            options.update(face_budget=request.get("face_budget", 8000), background=request.get("background", "auto"))
        elif operation == "material":
            options.update(texture_size=request.get("texture_size", 2048), camera=request.get("camera"))
        elif operation == "superresolution":
            options = {}
            if set(request) - {"operation", "image_path", "image_upload"}:
                raise MeshAxiomError("Superresolution accepts only operation and its image input")
        else:
            raise MeshAxiomError("Unknown operation")
        validate_options(operation, **options if options else {"seed": 0})
        image = self.resolve_input(request, "image", {".png", ".jpg", ".jpeg", ".webp"})
        mesh = self.resolve_input(request, "mesh", {".obj", ".glb", ".json"}) if operation == "material" else None
        with self.lock:
            if sum(j["status"] in {"queued", "running"} for j in self.jobs.values()) >= 8:
                raise MeshAxiomError("Queue is full; wait for existing jobs")
            job_id = uuid.uuid4().hex
            output = self.workspace / "jobs" / job_id
            self.jobs[job_id] = {"schema_version": 1, "id": job_id, "operation": operation, "status": "queued", "created_at": utc_now(), "output_dir": str(output)}
            write_json(self.workspace / "records" / (job_id + ".json"), self.jobs[job_id])
        self.pool.submit(self._run, job_id, operation, image, mesh, options)
        return self.get(job_id)

    def _run(self, job_id, operation, image, mesh, options):
        self.update(job_id, status="running", started_at=utc_now())
        try:
            if self._pipeline is None:
                self._pipeline = self.pipeline_factory(self.config)
            pipeline = self._pipeline
            output = self.get(job_id)["output_dir"]
            if operation == "geometry":
                result = pipeline.generate(image, output, **options)
            elif operation == "material":
                result = pipeline.texture(mesh, image, output, **options)
            else:
                result = pipeline.enhance(image, output)
            artifacts = {name: {**row, "download_url": f"/v1/jobs/{job_id}/artifacts/{name}"} for name, row in result["artifacts"].items()}
            self.update(job_id, status="complete", result={**result, "artifacts": artifacts}, finished_at=utc_now())
        except Exception as exc:
            self.update(job_id, status="failed", error=str(exc), error_type=type(exc).__name__, finished_at=utc_now())

    def update(self, job_id, **fields):
        with self.lock:
            self.jobs[job_id].update(fields)
            write_json(self.workspace / "records" / (job_id + ".json"), self.jobs[job_id])

    def get(self, job_id):
        if not re.fullmatch(r"[a-f0-9]{32}", job_id):
            raise KeyError("Unknown job")
        with self.lock:
            if job_id not in self.jobs:
                raise KeyError("Unknown job")
            return json.loads(json.dumps(self.jobs[job_id]))

    def close(self):
        self.pool.shutdown(wait=True)
        if self._pipeline is not None and hasattr(self._pipeline, "close"):
            self._pipeline.close()

def make_server(config, workspace, *, port=7860, manager=None):
    manager = manager or JobManager(config, workspace)
    class Handler(BaseHTTPRequestHandler):
        server_version = "MeshAxiom/0.2"

        def json(self, status, value):
            self.send_bytes(status, json.dumps(value, ensure_ascii=False).encode(), "application/json")

        def send_bytes(self, status, body, content_type):
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def body(self, limit):
            try:
                size = int(self.headers.get("Content-Length", "0"))
            except ValueError as exc:
                raise MeshAxiomError("Invalid body size") from exc
            if size <= 0 or size > limit:
                raise MeshAxiomError("Request body is empty or exceeds the size limit")
            return self.rfile.read(size)

        def do_GET(self):
            path = urlparse(self.path).path
            try:
                if path == "/":
                    self.send_bytes(200, Path(__file__).with_name("webui.html").read_bytes(), "text/html; charset=utf-8")
                elif path == "/health":
                    self.json(200, {"status": "ok", "schema_version": 1, "model_loading": "lazy_worker_processes"})
                elif path == "/v1/doctor":
                    self.json(200, doctor(config))
                elif path == "/openapi.json":
                    self.send_bytes(200, Path(__file__).with_name("schemas").joinpath("openapi.json").read_bytes(), "application/json")
                elif path == "/agent-tools.json":
                    self.send_bytes(200, Path(__file__).with_name("schemas").joinpath("agent-tools.json").read_bytes(), "application/json")
                elif match := re.fullmatch(r"/v1/jobs/([a-f0-9]{32})", path):
                    self.json(200, manager.get(match.group(1)))
                elif match := re.fullmatch(r"/v1/jobs/([a-f0-9]{32})/artifacts/([A-Za-z0-9_.-]+)", path):
                    job = manager.get(match.group(1))
                    row = job.get("result", {}).get("artifacts", {}).get(match.group(2))
                    if job["status"] != "complete" or not row:
                        raise KeyError("Artifact not available")
                    file = Path(row["path"]).resolve()
                    if not file.is_relative_to(Path(job["output_dir"]).resolve()):
                        raise KeyError("Invalid artifact")
                    with file.open("rb") as source:
                        actual = hashlib.file_digest(source, "sha256").hexdigest()
                        size = os.fstat(source.fileno()).st_size
                        if actual != row["sha256"] or size != row["size"]:
                            self.json(409, {"error": "Artifact changed after job completion"})
                            return
                        source.seek(0)
                        self.send_response(200)
                        self.send_header("Content-Type", mimetypes.guess_type(file.name)[0] or "application/octet-stream")
                        self.send_header("Content-Length", str(size))
                        self.send_header("Content-Disposition", 'attachment; filename="' + file.name + '"')
                        self.send_header("X-Content-Type-Options", "nosniff")
                        self.end_headers()
                        while chunk := source.read(1024 * 1024):
                            self.wfile.write(chunk)
                else:
                    self.json(404, {"error": "Unknown route"})
            except (KeyError, FileNotFoundError):
                self.json(404, {"error": "Not found"})

        def do_POST(self):
            path = urlparse(self.path).path
            try:
                origin = self.headers.get("Origin")
                if origin and urlparse(origin).netloc != self.headers.get("Host"):
                    self.json(403, {"error": "Cross-origin requests are not accepted"})
                    return
                if path == "/v1/uploads":
                    name = self.headers.get("X-Filename", "")
                    suffix = Path(name).suffix.lower()
                    if suffix not in {".png", ".jpg", ".jpeg", ".webp", ".glb", ".obj", ".json"}:
                        raise MeshAxiomError("Unsupported upload extension")
                    body = self.body(MAX_UPLOAD)
                    token = uuid.uuid4().hex + suffix
                    target = manager.uploads / token
                    with target.open("xb") as file:
                        file.write(body)
                    self.json(201, {"upload_id": token, "bytes": len(body)})
                elif path == "/v1/jobs":
                    request = json.loads(self.body(MAX_REQUEST))
                    job = manager.submit(request)
                    self.json(202, {**job, "status_url": "/v1/jobs/" + job["id"]})
                else:
                    self.json(404, {"error": "Unknown route"})
            except (MeshAxiomError, ValueError, OSError) as exc:
                self.json(400, {"status": "failed", "error": str(exc)})

        def log_message(self, format, *args):
            pass
    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    server.job_manager = manager
    return server

def serve(config, workspace, *, port=7860):
    server = make_server(config, workspace, port=port)
    print(json.dumps({"status": "listening", "url": f"http://127.0.0.1:{server.server_port}", "agent_tools": f"http://127.0.0.1:{server.server_port}/agent-tools.json"}), flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        server.job_manager.close()

