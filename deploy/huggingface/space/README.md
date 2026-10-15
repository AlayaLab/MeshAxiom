---
title: MeshAxiom
emoji: 🧊
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
---

MeshAxiom image-to-mesh demo using Exact inference.

Prepare a Docker Space from the repository root: copy this README and Dockerfile
to the root of that Space while keeping `src/`, `pyproject.toml`, and
`deploy/huggingface/space/app.py`. Set `MESHAXIOM_MODEL_ID` to the actual published
model and `MESHAXIOM_MODEL_REVISION` to its full commit SHA, or mount a prepared
model directory and set `MESHAXIOM_MODEL_DIR`.

Use `HF_TOKEN` as a Space secret when access is required. It is read from the
environment only. The container does not embed credentials or weights.

Version `0.2.0.dev1` passed one-image reference/Exact identity at 4K and 8K requests on H200 with NVIDIA PyTorch 25.03, including resident worker reuse. Other hardware and a cold Docker build still require target validation. Model
loading, image preparation and first-use graph capture add to interactive latency.
The two stage workers retain their models between requests; one request runs at a
time. Restarting the Space releases its worker processes.

See `docs/HUGGINGFACE.md` in the GitHub source for staging and validation.
