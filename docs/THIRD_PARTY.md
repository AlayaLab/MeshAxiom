# Authors and third-party components

Sho Kuno provided the Exact CUDA-graph vertex/topology implementation and the
vectorized post-processing integrated here. Public source identity hashes and
adaptation notes are recorded in `src/meshaxiom/vendor/kuno/ORIGIN.json`.
The delivered source did not include a standalone license; the authors' release
license must cover it before publication.

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
