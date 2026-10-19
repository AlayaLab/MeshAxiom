# Release status

MeshAxiom Exact inference code is available on GitHub and the Base model weights
are public on [Hugging Face](https://huggingface.co/AlayaLab/MeshAxiom-Base).
The optional Docker Space remains a deployment template. All public generation interfaces
use Exact. Internal naming and the topology metadata label have been migrated to
MeshAxiom. The private preparation record retains the original source mappings.

The release includes one EMA recipe: vertex 100K, topology 10K and VAE 30K.
Training checkpoints, optimizer state and training data are excluded from the
source repository. The separate model payload contains only the accepted EMA
checkpoints and required metadata. Material and
texture enhancement interfaces remain independent.

The verified model revision is `adb65c712a1c603e9792c35fc927399deac03643`.
The authors' code/weight license assignment (including the acceleration
contribution) remains pending. Version `0.2.0.dev2` passes 95 CPU tests and a bounded H200 reference/Exact identity check: one image at 4K and 8K, four identical geometry files per request, and resident worker reuse. Repository availability
does not assign a new license; the embedded DINOv3 and other third-party terms
still apply. `VALIDATION.json` records the checks performed on this integrated
candidate. A hosted Space and a successful cold Docker build are not claimed.

GitHub source/wheel archives must contain no model checkpoints, runtime outputs,
private infrastructure paths, credentials, obsolete package names or development
datasets. Model payload archives must contain the verified inference files and
checksum manifest. The release audit checks the actual built artifacts as well
as the source tree. No automatic public upload is part of these scripts.
