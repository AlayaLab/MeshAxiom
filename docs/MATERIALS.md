# Independent materials and texture enhancement

Material tools are independent from image-to-mesh inference. Configure
`configs/material.json` with local Pixal3D/TRELLIS.2/image-encoder/NAF paths and
use it directly; it does not require the three Base geometry checkpoints.

```bash
meshaxiom texture --config configs/material.json --mesh object.obj --image image.png --texture-size 2048 --output outputs/textured
meshaxiom enhance --config configs/material.json --image atlas.png --output outputs/enhanced
```

The texturing backend uses external Pixal3D, TRELLIS.2, DINOv3, NAF and CUDA
extensions. Their repositories, model weights and licenses are separate
prerequisites; this geometry source package does not bundle their model weights.
The texture worker preserves every original directed face, including existing
degenerate faces, and retains UV seam vertices separately from the source mesh.
Its completion report contains the geometry audit. A camera may be supplied
explicitly; otherwise the existing reference-fitting procedure is used.

Enhancement uses the separately configured `RealESRGAN_x2plus.pth`, divides the
atlas into tiles with overlap, and preserves alpha with exact nearest-neighbor
2× expansion. It does not change mesh geometry or invoke the Base geometry model.
The BasicSR and RealESRGAN notices are retained under `vendor/superresolution`.
