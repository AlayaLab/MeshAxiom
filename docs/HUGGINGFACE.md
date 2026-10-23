# Hugging Face deployment

GitHub contains the installable source package. A separate Hub model repository
contains the three Base EMA files, a model card, checksums, `model_index.json`,
`config.json`, its identical `meshaxiom.json` compatibility alias, and local DINOv3 configuration/license files. The directory
`deploy/huggingface/model` is the small metadata template. The private model
preparation process fills the final checksums and checks every EMA tensor.

To stage from the original accepted EMA exports without changing their values:

```bash
python scripts/prepare_model.py --weights-dir /path/to/accepted-ema-weights --output /path/to/new-model-directory
```

`--link` is available on the same filesystem for the accepted files. All three
already branded EMA files are copied or linked without changing their bytes.
Every tensor, dtype, key and metadata value is checked on readback. The staging
receipt and `SHA256SUMS` belong with the model payload.

The public model repository is `AlayaLab/MeshAxiom-Base`. No login is required
to download these weights. Resolve its current revision once and reuse that full
commit SHA for reproducible loads:

```python
from huggingface_hub import HfApi
from meshaxiom import MeshAxiomPipeline

revision = HfApi().model_info("AlayaLab/MeshAxiom-Base").sha
pipeline = MeshAxiomPipeline.from_pretrained(
    "AlayaLab/MeshAxiom-Base",
    revision=revision,
)
```

The implementation uses `huggingface_hub.snapshot_download` with a fixed data
allowlist. The returned cache remains read-only; runtime locks and caches are
outside that snapshot. See the [Hub download documentation](https://huggingface.co/docs/huggingface_hub/guides/download).
No public ID or revision is fabricated by the preparation code.

Version `0.2.0.dev3` reads `config.json` as the actual inference configuration.
It falls back to `meshaxiom.json` only when the standard entry is absent, so older
pinned revisions and local models continue to work. An unreadable or invalid
standard entry is rejected; when present it takes precedence over the legacy alias.
Preparation writes both files from one configuration after fixing
the checkpoint hashes. The model remains a `meshaxiom` library model; it does not
use a Transformers or Diffusers loader. Normal requests for the standard filename
allow Hub download statistics, which do not measure unique users or completed generations.

For an interactive Space, use `deploy/huggingface/space`. Its Dockerfile is built
from the repository root. Copy the Space README/Dockerfile to the new Space root
and retain the source tree and application path, then set:

- `MESHAXIOM_MODEL_ID` and `MESHAXIOM_MODEL_REVISION`, or `MESHAXIOM_MODEL_DIR` for a local model.
- `HF_TOKEN` as a Space secret if model access requires it.

The app loads lazily, keeps both stage workers resident, serializes requests and
provides mesh downloads. Version `0.2.0.dev2` also passed a bounded H200 identity check using one image at 4K and 8K requests, including resident worker reuse. Choose GPU memory only after a target canary; an arbitrary
small GPU is not assumed sufficient. Run `doctor` and `verify-exact` before serving.
See [Docker Spaces](https://huggingface.co/docs/hub/spaces-sdks-docker) for the
container port and runtime hardware requirements.

The code and model weights are public. The Docker
Space remains a deployment template. The staging scripts do not upload or create a Space. The authors'
license assignment and validation scope are tracked in
`docs/RELEASE_PREPARATION.md`.
