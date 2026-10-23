---
library_name: meshaxiom
pipeline_tag: image-to-3d
tags:
  - meshaxiom
  - image-to-mesh
  - mesh
  - 3d
  - exact
language:
  - en
  - zh
---

# MeshAxiom-Base

**MeshAxiom: An Agent-Native 3D Foundation Stack for Programmable Worlds**

[Project](https://alaya-lab.github.io/MeshAxiom/) · [Code](https://github.com/AlayaLab/MeshAxiom)

Public Base weights for **Exact** image-to-native-mesh inference. This repository
contains vertex EMA 100K, topology EMA 10K and topology VAE EMA 30K. Install the
MeshAxiom source package for the Python API, CLI and agent tools. No optimizer,
training data or alternative inference preset is included.

## Usage

```python
from huggingface_hub import HfApi
from meshaxiom import MeshAxiomPipeline

# Resolve once, then reuse this immutable revision for reproducible runs.
revision = HfApi().model_info("AlayaLab/MeshAxiom-Base").sha

with MeshAxiomPipeline.from_pretrained(
    "AlayaLab/MeshAxiom-Base",
    revision=revision,
) as model:
    result = model.generate("image.png", "outputs/object", face_budget=4000)
```

Use MeshAxiom code version `0.2.0.dev3` for the standard `config.json` entry point.
The identical `meshaxiom.json` is retained for version `0.2.0.dev2`; the new loader also accepts older pinned snapshots without `config.json`.
No Hugging Face login is required to download these weights. A local model directory
can also be used. This repository supplies model data; the source package supplies
the inference implementation. `model_index.json` is descriptive metadata, not a
Diffusers pipeline.

`config.json` contains the actual MeshAxiom inference configuration. Normal Hub
loads download and read this file; its standard name also enables the Hub's
download statistics. These statistics are Hub requests, not unique users or completed generations.

Input is a 512×512 RGB image on a black background. The recipe uses MBR8, 20
DPM++2M steps and topology CFG 3. The requested face budget guides generation;
actual face counts can differ. OBJ/JSON coordinates are Z-up, and exported GLB
files include a transform into glTF's Y-up frame.

## Verification and limitations

All 829 vertex EMA tensors exactly match the evaluated 100K checkpoint. The
unchanged topology and VAE files contain another 1,173 tensors. The vertex model
loaded strictly on CPU and both geometry stages passed the GPU check below.
Version `0.2.0.dev2` passes 95 CPU tests. See `WEIGHTS.json`, `VALIDATION.json`
and `SHA256SUMS` for model identities and the scope of verification.

Sho Kuno contributed the Exact acceleration. His tested platform is H200 with
NVIDIA PyTorch 25.03. This integrated release also passed a bounded H200 check: one image at 4K and 8K requests, with all four geometry files identical to the reference and resident workers reused between requests. This covers only the recorded inputs, seeds, weights and stack. A cold Docker build remains unverified; run `meshaxiom verify-exact` on your target setup.
Geometry can have open boundaries or degenerate faces. Earlier paper results and
speed measurements used the previous 180K vertex checkpoint; they are not
measurements of this 100K release.

## License

The authors' standalone release license is pending; see `LICENSE.md`. These
checkpoints contain frozen DINOv3 tensors subject to `external/dinov3/LICENSE.md`.
The included architecture and processor configurations avoid a separate backbone
weight download. Third-party terms and notices are retained.

## Paper

The arXiv link and citation will be added when the preprint is available.

## Acknowledgements

MeshAxiom-Base's geometry pipeline is a paper-based reimplementation of the two-stage vertex and topology diffusion framework in [Nexus](https://arxiv.org/abs/2607.13563). MeshAxiom builds on this foundation with Exact inference acceleration, agent/tool interfaces, and downstream 3D workflows. We thank the Nexus authors for their foundational work.
