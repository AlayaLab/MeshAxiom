# Hugging Face deployment

GitHub contains the installable source package. A separate Hub model repository
contains the three Base EMA files, a model card, checksums, `model_index.json`,
`meshaxiom.json`, and local DINOv3 configuration/license files. The directory
`deploy/huggingface/model` is the small metadata template. The private model
preparation process fills the final checksums and checks every EMA tensor.

To stage from the original accepted EMA exports without changing their values:

```bash
python scripts/prepare_model.py --weights-dir /path/to/accepted-ema-weights --output /path/to/new-model-directory
```

`--link` is available on the same filesystem for the two byte-unchanged files.
The topology file gets the new product name in one metadata field; every tensor,
dtype, key, configuration and non-brand metadata value is checked on readback.
The staging receipt and `SHA256SUMS` belong with the model payload. SHA256 binds
file content, so the renamed topology archive has its own checksum.

The public model repository is `AlayaLab/MeshAxiom-Base`. Its verified model
revision is `b80240ce4f77a28bc4debdcca895e96c78ddb294`. Consumers can use:

```python
from meshaxiom import MeshAxiomPipeline
pipeline = MeshAxiomPipeline.from_pretrained(
    "AlayaLab/MeshAxiom-Base",
    revision="b80240ce4f77a28bc4debdcca895e96c78ddb294",
)
```

The implementation uses `huggingface_hub.snapshot_download` with a fixed data
allowlist. The returned cache remains read-only; runtime locks and caches are
outside that snapshot. See the [Hub download documentation](https://huggingface.co/docs/huggingface_hub/guides/download).
No public ID or revision is fabricated by the preparation code.

For an interactive Space, use `deploy/huggingface/space`. Its Dockerfile is built
from the repository root. Copy the Space README/Dockerfile to the new Space root
and retain the source tree and application path, then set:

- `MESHAXIOM_MODEL_ID` and `MESHAXIOM_MODEL_REVISION`, or `MESHAXIOM_MODEL_DIR` for a local model.
- `HF_TOKEN` as a Space secret if model access requires it.

The app loads lazily, keeps both stage workers resident, serializes requests and
provides mesh downloads. Version `0.2.0.dev1` also passed a bounded H200 identity check using one image at 4K and 8K requests, including resident worker reuse. Choose GPU memory only after a target canary; an arbitrary
small GPU is not assumed sufficient. Run `doctor` and `verify-exact` before serving.
See [Docker Spaces](https://huggingface.co/docs/hub/spaces-sdks-docker) for the
container port and runtime hardware requirements.

The code and model weights are public; the Docker Space remains a deployment
template. The staging scripts do not upload or create a Space. The authors'
license assignment and validation scope are tracked in
`docs/RELEASE_PREPARATION.md`.
