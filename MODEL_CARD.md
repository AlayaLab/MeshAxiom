# MeshAxiom-Base model card

MeshAxiom: An Agent-Native 3D Foundation Stack for Programmable Worlds.

The single released recipe is vertex EMA 180K, topology EMA 10K, topology VAE EMA
30K. Inference uses Sho Kuno's Exact implementation with the same tensors,
MBR8, 20 DPM++2M steps, and topology CFG 3. Material generation is independent.

## Checkpoint identities

These are the final inference-only file identities, after metadata branding.
They exclude optimizer, raw non-EMA model weights, training RNG and training data.

| Component | Step | Tensors | Bytes | SHA256 |
| --- | ---: | ---: | ---: | --- |
| Vertex | 180000 | 955 | 11235479045 | `4d60148fdddb65df502becbf9786131836a880fdba226180ce586a8ee10620fb` |
| Topology | 10000 | 830 | 9423507316 | `7fb0c7cc2035616f811326d3e64c1393de39d8f652ab7aa122ab84a987d2a19f` |
| Vae | 30000 | 343 | 446157647 | `7d6833425af71e923756d2ebdb2863e704857bffb8b365ab0c2f5ec3bfb6f4bd` |

All 2,128 EMA tensors retain their original keys, dtypes, shapes and exact values.
The topology export changes only the `v113.version` product label; the vertex and
VAE exports are byte-unchanged. The complete readback receipt is
[WEIGHTS.json](deploy/huggingface/model/WEIGHTS.json).

## Input and output

Both geometry stages use the same prepared 512×512 RGB-on-black image. Face
requests are generation conditions, not forced triangle counts. Native raw faces
are preserved; only export winding is adjusted. OBJ/JSON are Z-up and GLB has an
explicit Y-up scene transform. Meshes may contain open boundaries or degenerate
faces; counts and audits are reported rather than silently repaired.

## Image backbone and external models

The vertex and topology checkpoints each contain 415 frozen DINOv3 backbone
tensors. These are external pretrained tensors, not weights trained entirely by
MeshAxiom. Architecture and processor configurations are shipped separately;
strict EMA loading supplies the complete backbone without a second weight download.
The DINOv3 license is retained in the model payload. Public redistribution must
respect its terms.

Independent texturing uses external Pixal3D/TRELLIS.2/DINOv3/NAF components and
does not represent newly trained MeshAxiom material weights. Texture enhancement
uses separately acquired RealESRGAN weights. See [materials](docs/MATERIALS.md).

## Validation and release status

[VALIDATION.json](VALIDATION.json) describes checks on this integrated candidate.
The author's original Exact identity and timing evidence does not establish
identity on every GPU, stack or input. The bundled `verify-exact` command compares
the newly branded reference against Exact on the target machine.

The [MeshAxiom-Base model repository](https://huggingface.co/AlayaLab/MeshAxiom-Base)
is public. The verified model revision is
`859a522c77e25770619f16e89a57447a0bb9bc16`. The authors' code/weight license
assignment and target GPU identity verification remain pending. An interactive
Space is not hosted yet.
