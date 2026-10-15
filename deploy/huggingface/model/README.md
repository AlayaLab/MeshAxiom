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

MeshAxiom: An Agent-Native 3D Foundation Stack for Programmable Worlds.

[Project](https://alaya-lab.github.io/MeshAxiom/) · [Source](https://github.com/AlayaLab/MeshAxiom)

This local model payload contains vertex EMA 180K, topology EMA 10K and topology
VAE EMA 30K for image-to-native-mesh inference. Use the MeshAxiom source package's
Exact implementation. No optimizer, training data or distilled model is included.

```python
from meshaxiom import MeshAxiomPipeline
with MeshAxiomPipeline.from_pretrained("/path/to/this-model-directory") as model:
    result = model.generate("image.png", "outputs/object", face_budget=4000)
```

After publication, a Hub ID and full commit SHA can replace the local directory.
The source package supplies the inference implementation; this payload supplies
data only. `model_index.json` is descriptive metadata, not a Diffusers pipeline.

The face request conditions generation and does not guarantee an exact count.
The model uses 512×512 RGB-on-black input, MBR8, 20 DPM++2M steps and topology CFG 3.
Output coordinates are Z-up; the generated GLB explicitly transforms them to Y-up.

Sho Kuno contributed the Exact inference implementation. Existing author timing
and identity evidence is described in the source documentation; this payload's
tensor-roundtrip receipt is `WEIGHTS.json`. It does not by itself prove new GPU
identity, speed, semantic quality or out-of-distribution performance.

## License and limitations

The authors' public release license is pending; see `LICENSE.md`. The checkpoints
include frozen DINOv3 tensors subject to `external/dinov3/LICENSE.md`. Only the
architecture and processor configurations are additionally included.

The acceleration author's tested platform is H200 with NVIDIA PyTorch 25.03.
Verify the target environment with `meshaxiom verify-exact`. Geometry can have
open boundaries or degenerate faces.

This is release preparation, with no public repository revision assigned yet.
