"""Shared execution used by human and agent interfaces."""
import contextlib
import fcntl
import json
import os
import re
import shutil
import subprocess
import sys
import time
import threading
from pathlib import Path
from .config import load_config, check_assets
from .contracts import MeshAxiomError, file_binding, validate_options, utc_now, write_json
from .preprocess import prepare_image

class MeshAxiomPipeline:
    def __init__(self, config):
        self.config = load_config(config)
        self._workers = {}
        self._lease = None
        self._request_lock = threading.RLock()
        self._reference = False

    def __enter__(self):
        return self

    def __exit__(self, *unused):
        self.close()

    def _close_models(self):
        for worker in self._workers.values():
            worker.close()
        self._workers.clear()

    def close(self):
        """Release resident models and this pipeline's cooperative GPU lease."""
        with self._request_lock:
            self._close_models()
            if self._lease is not None:
                fcntl.flock(self._lease.fileno(), fcntl.LOCK_UN)
                self._lease.close()
                self._lease = None

    @classmethod
    def from_pretrained(cls, model, *, revision=None, cache_dir=None, local_files_only=False, token=None, device="cuda:0"):
        from .hub import download_model, model_config
        path = Path(model).expanduser()
        directory = path if path.is_dir() else download_model(str(model), revision=revision, cache_dir=cache_dir,
                                                              local_files_only=local_files_only, token=token)
        return cls(model_config(directory, device=device))

    def generate(self, image, output_dir, *, face_budget=8000, seed=2026082301, background="auto"):
        validate_options("geometry", seed=seed, face_budget=face_budget, background=background)
        with self._request_lock:
            return self._execute("geometry", image, output_dir, face_budget=face_budget, seed=seed, background=background)

    def texture(self, mesh, image, output_dir, *, seed=42, texture_size=2048, camera=None):
        validate_options("material", seed=seed, texture_size=texture_size, camera=camera)
        p = Path(mesh).expanduser().resolve()
        if not p.is_file() or p.suffix.lower() not in {".obj", ".glb", ".json"}:
            raise MeshAxiomError("Material input must be an existing OBJ, GLB or mesh JSON")
        with self._request_lock:
            self._close_models()
            return self._execute("material", image, output_dir, mesh=str(p), seed=seed, texture_size=texture_size, camera=camera)

    def enhance(self, image, output_dir):
        """Optional deterministic 2x atlas enhancement, independent of geometry."""
        with self._request_lock:
            self._close_models()
            return self._execute("superresolution", image, output_dir)

    def _execute(self, operation, image, output_dir, **options):
        image = Path(image).expanduser().resolve()
        if not image.is_file():
            raise MeshAxiomError("Input image does not exist")
        output = Path(output_dir).expanduser().resolve()
        if output.exists() and any(output.iterdir()):
            raise MeshAxiomError("Output directory must be empty; completed and failed jobs are retained")
        output.mkdir(parents=True, exist_ok=True)
        status = {"schema_version": 1, "operation": operation, "status": "running", "started_at": utc_now(), "stage": "preflight", "artifacts": {}}
        write_json(output / "status.json", status)
        try:
            runtime = self.config["runtime"]
            _space_guard(output, runtime)
            with self._gpu_lock():
                assets = check_assets(self.config, operation, verify_hashes=True)
                if operation == "geometry":
                    prepared = prepare_image(image, output / "input.png", background=options["background"])
                    write_json(output / "input_report.json", prepared)
                    worker_image = output / "input.png"
                else:
                    prepared = {"input": file_binding(image)}
                    worker_image = image
                request = {"schema_version": 1, "operation": operation, "config": self.config, "image": str(worker_image),
                           "output_dir": str(output), "device": runtime["device"], **options}
                write_json(output / "request.json", request)
                stages = ["vertex", "topology"] if operation == "geometry" else [operation]
                for stage in stages:
                    status["stage"] = stage
                    write_json(output / "status.json", status)
                    _space_guard(output, runtime)
                    self._run_worker(stage, output / "request.json", output)
                report_file = output / ("geometry_report.json" if operation == "geometry" else "report.json")
                if not report_file.is_file():
                    raise MeshAxiomError("Worker did not produce its completion report")
                report = json.loads(report_file.read_text())
                if report.get("status") not in {"COMPLETE", "PASS", "complete", "success", "SUCCEEDED"}:
                    raise MeshAxiomError("Worker report did not confirm completion")
                artifacts = report.get("artifacts", {})
                if not artifacts:
                    raise MeshAxiomError("Worker completion contains no artifacts")
                validated = {}
                for name, binding in artifacts.items():
                    if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9_.-]+", name):
                        raise MeshAxiomError("Worker artifact name is not a valid download identifier")
                    p = Path(binding["path"] if isinstance(binding, dict) else binding).resolve()
                    if not p.is_relative_to(output) or not p.is_file():
                        raise MeshAxiomError("Worker artifact escapes its output directory or is missing")
                    actual = file_binding(p)
                    if isinstance(binding, dict) and (binding.get("sha256", actual["sha256"]) != actual["sha256"] or binding.get("size", actual["size"]) != actual["size"]):
                        raise MeshAxiomError("Worker artifact binding mismatch: " + name)
                    validated[name] = actual
                result = {**status, "status": "complete", "stage": "complete", "finished_at": utc_now(),
                          "artifacts": validated, "input": prepared, "parameters": options, "models": assets,
                          "report": file_binding(report_file), "output_dir": str(output),
                          "coordinate_frame": report.get("coordinate_frame"), "geometry_audit": report.get("geometry_audit")}
                if "artifact_coordinate_frames" in report:
                    result["artifact_coordinate_frames"] = report["artifact_coordinate_frames"]
                write_json(output / "result.json", result)
                write_json(output / "status.json", result)
                return result
        except Exception as exc:
            self.close()
            failure = {**status, "status": "failed", "finished_at": utc_now(), "error": str(exc), "error_type": type(exc).__name__}
            write_json(output / "FAILED.json", failure)
            write_json(output / "status.json", failure)
            if isinstance(exc, MeshAxiomError):
                raise
            raise MeshAxiomError(f"{operation} failed during {status['stage']}: {exc}") from exc

    def _run_worker(self, stage, request, output):
        runtime = self.config["runtime"]
        if stage in {"vertex", "topology"}:
            from .session import ResidentWorker
            if stage not in self._workers:
                self._workers[stage] = ResidentWorker(stage, runtime, reference=self._reference)
            return self._workers[stage].run(request, output)
        interpreter = runtime[stage + "_python"]
        if not Path(interpreter).is_file():
            raise MeshAxiomError("Python interpreter is missing: " + interpreter)
        env = os.environ.copy()
        source = str(Path(__file__).resolve().parents[1])
        env["PYTHONPATH"] = source + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
        env.update(OMP_NUM_THREADS="2", MKL_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2", TOKENIZERS_PARALLELISM="false", PYTHONDONTWRITEBYTECODE="1")
        with (output / (stage + ".log")).open("w") as log:
            try:
                process = subprocess.Popen([interpreter, "-m", "meshaxiom.workers." + stage, "--request", str(request)], env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                try:
                    code = process.wait(timeout=runtime.get("timeout_seconds", 3600))
                except subprocess.TimeoutExpired:
                    # Only this worker's private process group is terminated.
                    import signal
                    os.killpg(process.pid, signal.SIGTERM)
                    try:
                        process.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid, signal.SIGKILL)
                        process.wait()
                    raise MeshAxiomError(f"{stage} worker exceeded its finite timeout")
            except OSError as exc:
                raise MeshAxiomError("Cannot start " + stage + " worker") from exc
        if code:
            raise MeshAxiomError(f"{stage} worker failed with exit code {code}; see {output / (stage + '.log')}")

    @contextlib.contextmanager
    def _gpu_lock(self):
        runtime = self.config["runtime"]
        device = runtime.get("device", "cuda:0")
        match = re.fullmatch(r"cuda:(\d+)", device)
        if not match:
            raise MeshAxiomError("Model inference needs a CUDA device such as cuda:0")
        folder = Path(runtime["gpu_lock_directory"])
        folder.mkdir(parents=True, exist_ok=True)
        pattern = runtime.get("gpu_lock_pattern", "cuda-{index}.lock")
        if "/" in pattern or ".." in pattern:
            raise MeshAxiomError("gpu_lock_pattern must be a filename")
        path = folder / pattern.format(index=match.group(1))
        timeout = min(runtime.get("timeout_seconds", 3600), runtime.get("lock_timeout_seconds", 60))
        if self._lease is not None:
            yield
            return
        lock = path.open("a")
        started = time.monotonic()
        try:
            while True:
                try:
                    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    break
                except BlockingIOError:
                    if time.monotonic() - started >= timeout:
                        raise MeshAxiomError("GPU is reserved by another process")
                    time.sleep(.2)
        except BaseException:
            lock.close()
            raise
        self._lease = lock
        # Held while models are resident, including the idle time between jobs.
        yield

def _space_guard(output, runtime):
    scratch = Path(runtime.get("scratch_directory", "/tmp"))
    for path, threshold in [(scratch, runtime.get("minimum_free_disk_gib", 0)), (output, runtime.get("minimum_free_workspace_gib", 0))]:
        if shutil.disk_usage(path).free < threshold * 1024**3:
            raise MeshAxiomError(f"Free space below configured {threshold} GiB threshold: {path}")
