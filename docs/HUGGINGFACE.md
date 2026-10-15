# Hugging Face preparation

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

Once model publication is approved and complete, record the actual repository ID
and full commit SHA. Consumers can use:

```python
from meshaxiom import MeshAxiomPipeline
pipeline = MeshAxiomPipeline.from_pretrained(model_id, revision=model_commit_sha)
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
provides mesh downloads. Hardware validation currently comes from the author's
H200 environment. Choose GPU memory only after a target canary; an arbitrary
small GPU is not assumed sufficient. Run `doctor` and `verify-exact` before serving.
See [Docker Spaces](https://huggingface.co/docs/hub/spaces-sdks-docker) for the
container port and runtime hardware requirements.

This preparation step performs no Hub upload, account modification or Space
creation. Public licensing and target GPU verification are tracked in
`docs/RELEASE_PREPARATION.md`.
