# Prepared example

`reference.png` is the existing generated cockpit reference prepared on a black
512×512 background. Use `background: "keep"` so no matting or crop changes its
pixels. It is a finite illustration, not a benchmark or training-data example.

After configuring the verified 100K weights and the external material models:

```bash
meshaxiom run-tool --config configs/base.json --request examples/generate.json
meshaxiom run-tool --config configs/material.json --request examples/texture.json
meshaxiom run-tool --config configs/material.json --request examples/enhance.json
```

This workflow was tested on a single H200 with the default 20-step geometry
recipe, requested face budget 4000, material atlas 1024, and separate 2× atlas
enhancement. The actual geometry had 2,210 vertices and 4,322 faces. See the
candidate's `VALIDATION.json` for recorded acceptance scope. Exact bitwise
replay across different CUDA/PyTorch stacks is not promised by these examples.

Use fresh output directories when trying another image or seed. The enhanced PNG
is a separate artifact; the material GLB retains its original generated atlas.
