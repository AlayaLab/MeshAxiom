# Release preparation status

This tree prepares MeshAxiom Exact inference for GitHub, a separate Hugging Face
Base model payload and an optional Docker Space. All public generation interfaces
use Exact. Internal naming and the topology metadata label have been migrated to
MeshAxiom. The private preparation record retains the original source mappings.

The release includes one EMA recipe: vertex 180K, topology 10K and VAE 30K.
Training checkpoints, optimizer state and training data are excluded from the
source repository. The separate model payload contains only the accepted EMA
checkpoints and required metadata. Material and
texture enhancement interfaces remain independent.

Before public release, finalize the authors' code/weight license (including the
acceleration contribution), respect the embedded DINOv3 terms, publish the actual
Hub model revision and complete `verify-exact` on the target GPU. `VALIDATION.json`
records the precise checks performed on this integrated candidate. The Docker
Space is a deployment template; preparation does not mean it is hosted or that a
cold build has passed.

GitHub source/wheel archives must contain no model checkpoints, runtime outputs,
private infrastructure paths, credentials, obsolete package names or development
datasets. Model payload archives must contain the verified inference files and
checksum manifest. The release audit checks the actual built artifacts as well
as the source tree. No automatic public upload is part of these scripts.
