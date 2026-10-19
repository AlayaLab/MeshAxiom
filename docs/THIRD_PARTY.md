# Authors and third-party components

## Geometry methodology: Nexus

The geometry generation method follows [Nexus: Native Mesh Generation with Diffusion](https://arxiv.org/abs/2607.13563) (Wang et al., 2026): separate vertex and topology diffusion stages, hierarchical sparse-octree vertex generation, continuous per-vertex topology representations, and Spacetime Interval-based connectivity reconstruction. These core designs are credited to Nexus and are not claimed as original MeshAxiom methods.

MeshAxiom supplies its paper-based reimplementation, Sho Kuno's Exact inference acceleration, agent/tool interfaces and downstream asset/scene workflows. See the [README attribution and citation](https://github.com/AlayaLab/MeshAxiom#relationship-to-nexus) for the technical relationship and bibliography.

This acknowledgment credits the research method. It does not imply an official Nexus release or endorsement of MeshAxiom by the Nexus authors, and it does not alter any code or checkpoint license.

## Exact implementation and dependencies

Sho Kuno provided the Exact CUDA-graph vertex/topology implementation and the
vectorized post-processing integrated here. Public source identity hashes and
adaptation notes are recorded in `src/meshaxiom/vendor/kuno/ORIGIN.json`.
The delivered source did not include a standalone license; the authors' license
assignment for the integrated contribution remains pending. Public availability
does not assign a new license.

The Base geometry models include frozen DINOv3 image-backbone tensors. Their
use and redistribution are governed by the DINOv3 terms included with the model
configuration. They are not relicensed by this package. PyTorch, Transformers,
NumPy, SciPy, Pillow, Numba and the Hub client retain their own licenses.

Independent material generation also uses Pixal3D, TRELLIS.2 and NAF. Obtain and
retain their licenses and model terms with those dependencies. BasicSR's
Apache-2.0 and RealESRGAN's BSD-3-Clause files and notices are retained in
`src/meshaxiom/vendor/superresolution`. Their attributions are preserved during
product renaming. No training dataset, authentication, machine configuration,
optimizer state or external material weights belong in the source archive.
