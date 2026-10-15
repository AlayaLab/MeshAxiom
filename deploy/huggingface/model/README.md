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
contains vertex EMA 180K, topology EMA 10K and topology VAE EMA 30K. Install the
MeshAxiom source package for the Python API, CLI and agent tools. No optimizer,
training data or alternative inference preset is included.

```python
from meshaxiom import MeshAxiomPipeline

with MeshAxiomPipeline.from_pretrained(
    "AlayaLab/MeshAxiom-Base",
    revision="859a522c77e25770619f16e89a57447a0bb9bc16",
) as model:
    result = model.generate("image.png", "outputs/object", face_budget=4000)
```

The pinned revision is the weight-identity baseline; current validation is recorded in this repository's latest `VALIDATION.json`. Install MeshAxiom code version `0.2.0.dev1` or later for the Hub-cache and worker-startup fixes. A local model directory
can also be used. This repository supplies model data; the source package supplies
the inference implementation. `model_index.json` is descriptive metadata, not a
Diffusers pipeline.

Input is a 512×512 RGB image on a black background. The recipe uses MBR8, 20
DPM++2M steps and topology CFG 3. The requested face budget guides generation;
actual face counts can differ. OBJ/JSON coordinates are Z-up, and exported GLB
files include a transform into glTF's Y-up frame.

## Verification and limitations

All 2,128 EMA tensors were preserved and the three models loaded strictly on CPU.
Version `0.2.0.dev1` passes 95 CPU tests. Uploaded weight hashes and authenticated sample byte ranges
match the verified local files. See `WEIGHTS.json`, `VALIDATION.json` and
`SHA256SUMS` for the recorded model identities and validation scope.

Sho Kuno contributed the Exact acceleration. His tested platform is H200 with
NVIDIA PyTorch 25.03. This integrated release also passed a bounded H200 check: one image at 4K and 8K requests, with all four geometry files identical to the reference and resident workers reused between requests. This covers only the recorded inputs, seeds, weights and stack. A cold Docker build remains unverified; run `meshaxiom verify-exact` on your target setup.
Geometry can have open boundaries or degenerate faces.

## License

The authors' standalone release license is pending; see `LICENSE.md`. These
checkpoints contain frozen DINOv3 tensors subject to `external/dinov3/LICENSE.md`.
The included architecture and processor configurations avoid a separate backbone
weight download. Third-party terms and notices are retained.

## Paper

The arXiv link and citation will be added when the preprint is available.
