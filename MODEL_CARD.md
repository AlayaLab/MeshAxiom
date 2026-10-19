# MeshAxiom-Base model card

MeshAxiom: An Agent-Native 3D Foundation Stack for Programmable Worlds.

The single released recipe is vertex EMA 100K, topology EMA 10K, topology VAE EMA
30K. Inference uses Sho Kuno's Exact implementation with the same tensors,
MBR8, 20 DPM++2M steps, and topology CFG 3. Material generation is independent.

## Relationship to Nexus

The geometry generation method follows [Nexus: Native Mesh Generation with Diffusion](https://arxiv.org/abs/2607.13563) (Wang et al., 2026): separate vertex and topology diffusion stages, hierarchical sparse-octree vertex generation, continuous per-vertex topology representations, and Spacetime Interval-based connectivity reconstruction. These core designs are credited to Nexus and are not claimed as original MeshAxiom methods.

MeshAxiom supplies its paper-based reimplementation, Sho Kuno's Exact inference acceleration, agent/tool interfaces and downstream asset/scene workflows. See the [README attribution and citation](https://github.com/AlayaLab/MeshAxiom#relationship-to-nexus) for the technical relationship and bibliography.

## Checkpoint identities

These are the final inference-only file identities, after metadata branding.
They exclude optimizer, raw non-EMA model weights, training RNG and training data.

| Component | Step | Tensors | Bytes | SHA256 |
| --- | ---: | ---: | ---: | --- |
| Vertex | 100000 | 829 | 9421814821 | `7743cc03ad05abcadab204b8a00f7d312ad2b26d7de1eb64ba009a75c12054f2` |
| Topology | 10000 | 830 | 9423507316 | `7fb0c7cc2035616f811326d3e64c1393de39d8f652ab7aa122ab84a987d2a19f` |
| Vae | 30000 | 343 | 446157647 | `7d6833425af71e923756d2ebdb2863e704857bffb8b365ab0c2f5ec3bfb6f4bd` |

All 2,002 EMA tensors retain their original keys, dtypes, shapes and exact values.
The 100K vertex export removes training-only metadata. Topology and VAE files
keep their previously verified identities. The complete readback receipt is
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

[VALIDATION.json](VALIDATION.json) records 95 passing CPU tests and one-image H200 reference/Exact identity at 4K and 8K requests, including resident worker reuse.
Earlier timing and paper results used the previous 180K vertex checkpoint and
are not measurements of this 100K release. Exact identity does not establish
identity on every GPU, stack or input. The bundled `verify-exact` command compares
the newly branded reference against Exact on the target machine.

The [MeshAxiom-Base model repository](https://huggingface.co/AlayaLab/MeshAxiom-Base)
is public. The verified model revision is
`adb65c712a1c603e9792c35fc927399deac03643`. The authors' code/weight license
assignment remains pending. An interactive
Space is not hosted yet.
