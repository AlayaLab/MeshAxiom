"""Bounded, persistent worker processes for the two isolated model namespaces."""
from __future__ import annotations

import json
import os
from pathlib import Path
import selectors
import signal
import subprocess
import sys
import threading
import uuid

from .contracts import MeshAxiomError


class ResidentWorker:
    """Keep one stage resident; a private pipe carries replies, never model logs."""

    def __init__(self, stage, runtime, *, reference=False):
        if stage not in {"vertex", "topology"}:
            raise ValueError("Unknown geometry stage")
        self.stage = stage
        self.runtime = runtime
        self._lock = threading.Lock()
        self.process = None
        self._reply = None
        self._buffer = b""
        self.reference = reference

    def _start(self):
        env = os.environ.copy()
        # Only the child is changed; no global environment or installation edits.
        for key in list(env):
            if key.startswith(("MESHAXIOM_", "NCCL_", "TORCHELASTIC_")) or key in {
                "RANK", "WORLD_SIZE", "LOCAL_RANK", "MASTER_ADDR", "MASTER_PORT",
                "P7_COARSE_DIT", "P7_COARSE", "P7_EXACT_WARMUP",
            }:
                env.pop(key, None)
        source = str(Path(__file__).resolve().parents[1])
        env["PYTHONPATH"] = source + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
        env.update(OMP_NUM_THREADS="2", MKL_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2",
                   TOKENIZERS_PARALLELISM="false", PYTHONDONTWRITEBYTECODE="1")
        cache = Path(self.runtime.get("cache_directory", Path(self.runtime.get("scratch_directory", "/tmp")) / "meshaxiom-cache"))
        for variable, part in [("NUMBA_CACHE_DIR", "numba"), ("TORCHINDUCTOR_CACHE_DIR", "inductor"),
                               ("TRITON_CACHE_DIR", "triton")]:
            path = cache / self.stage / part
            path.mkdir(parents=True, exist_ok=True)
            env[variable] = str(path)
        # Cache preparation can fail (read-only filesystem, a file in place of
        # a directory). Allocate pipe FDs only after those operations succeed.
        read_fd, write_fd = os.pipe()
        try:
            self.process = subprocess.Popen(
                [self.runtime[self.stage + "_python"], "-m", "meshaxiom.workers.resident",
                 "--stage", self.stage, "--reply-fd", str(write_fd)] + (["--reference"] if self.reference else []),
                stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                env=env, pass_fds=(write_fd,), start_new_session=True,
            )
        except BaseException:
            os.close(read_fd)
            raise
        finally:
            os.close(write_fd)
        self._reply = read_fd

    def run(self, request, output):
        with self._lock:
            if self.process is None:
                self._start()
            if self.process.poll() is not None:
                raise MeshAxiomError(f"{self.stage} resident worker exited; create a new pipeline after inspecting the failed request")
            job_id = uuid.uuid4().hex
            payload = {"id": job_id, "request": str(request), "log": str(Path(output) / (self.stage + ".log"))}
            try:
                self.process.stdin.write((json.dumps(payload) + "\n").encode())
                self.process.stdin.flush()
                with selectors.DefaultSelector() as selector:
                    selector.register(self._reply, selectors.EVENT_READ)
                    import time
                    deadline = time.monotonic() + self.runtime.get("timeout_seconds", 3600)
                    while b"\n" not in self._buffer:
                        remaining = deadline - time.monotonic()
                        if remaining <= 0 or not selector.select(remaining):
                            self.close()
                            raise MeshAxiomError(f"{self.stage} worker exceeded its finite timeout")
                        data = os.read(self._reply, 65536)
                        if not data:
                            raise MeshAxiomError(f"{self.stage} worker exited without a completion reply; see {payload['log']}")
                        self._buffer += data
                line, self._buffer = self._buffer.split(b"\n", 1)
                result = json.loads(line)
                if result.get("id") != job_id or result.get("status") != "complete":
                    raise MeshAxiomError(f"{self.stage} worker failed: {result.get('error', 'invalid reply')}; see {payload['log']}")
                return result
            except BaseException:
                # No partially failed CUDA context is reused for another request.
                self.close()
                raise

    def close(self):
        process, self.process = self.process, None
        if process is not None:
            if process.stdin:
                process.stdin.close()
            if process.poll() is None:
                try:
                    process.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    try:
                        os.killpg(process.pid, signal.SIGTERM)
                    except ProcessLookupError:
                        pass
                    try:
                        process.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        try:
                            os.killpg(process.pid, signal.SIGKILL)
                        except ProcessLookupError:
                            pass
                        process.wait()
        if self._reply is not None:
            os.close(self._reply)
            self._reply = None
        self._buffer = b""
