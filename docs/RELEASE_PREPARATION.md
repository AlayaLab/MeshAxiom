# Release status

MeshAxiom Exact inference code is available on GitHub and the Base model weights
are public on [Hugging Face](https://huggingface.co/AlayaLab/MeshAxiom-Base).
The optional Docker Space remains a deployment template. All public generation interfaces
use Exact. Internal naming and the topology metadata label have been migrated to
MeshAxiom. The private preparation record retains the original source mappings.

The release includes one EMA recipe: vertex 180K, topology 10K and VAE 30K.
Training checkpoints, optimizer state and training data are excluded from the
source repository. The separate model payload contains only the accepted EMA
checkpoints and required metadata. Material and
texture enhancement interfaces remain independent.

The verified model revision is `859a522c77e25770619f16e89a57447a0bb9bc16`.
The authors' code/weight license assignment (including the acceleration
contribution) and target GPU `verify-exact` remain pending. Public availability
does not assign a new license; the embedded DINOv3 and other third-party terms
still apply. `VALIDATION.json` records the checks performed on this integrated
candidate. A hosted Space and a successful cold Docker build are not claimed.

GitHub source/wheel archives must contain no model checkpoints, runtime outputs,
private infrastructure paths, credentials, obsolete package names or development
datasets. Model payload archives must contain the verified inference files and
checksum manifest. The release audit checks the actual built artifacts as well
as the source tree. No automatic public upload is part of these scripts.
