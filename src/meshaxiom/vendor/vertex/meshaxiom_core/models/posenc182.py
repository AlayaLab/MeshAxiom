"""V182 (09-27): position-aware cross-attention, built into the model (the V150 local adapter, promoted to a cluster run).

DINOv3 (ViT-L/16, RoPE only: no absolute position embedding in its weights) patch tokens reach the vertex DiT through one linear
projection, so the image tokens carry no explicit 2D position, and the octree-node queries of the cross-attention carry no absolute
3D position (RoPE is relative and only used in self-attention).  The 09-24 attention probe found that only a few heads localise.
With MESHAXIOM_POSENC_BUILD=1 the model gets
  * key side  : main-view patch tokens += k_pos(fourier2d(patch centre))  (cond.encoder.k_pos, zero-initialised)
  * query side: image cross-attention query input += q_pos_b(fourier3d(node))  (dit.blocks.<b>.q_pos, zero-initialised, one per block)
The V181 side-view branch stays position-free: side views are rotated by arbitrary relative yaws, and one shared separable 2D/3D code
would impose the main-view geometry on them (a rotated per-view query code would be needed; not in V182).
Both adapters are zero-initialised, so a model built from a checkpoint without them is exactly that checkpoint at initialisation."""
import math

import torch
import torch.nn as nn

NF3, NF2 = 8, 8
Q_DIM, K_DIM = 3 + 2 * 3 * NF3, 2 + 2 * 2 * NF2
DEPTH_TOTAL = 9


def fourier(x, nf):
    """x (..., k) in [-1, 1] -> (..., k + 2 k nf): identity, sin and cos at frequencies pi * 2^0..2^(nf-1)."""
    f = (2.0 ** torch.arange(nf, device=x.device, dtype=torch.float32)) * math.pi
    a = x.float()[..., None] * f
    return torch.cat([x.float(), torch.sin(a).flatten(-2), torch.cos(a).flatten(-2)], -1)


def q_features(pos, depth_total=DEPTH_TOTAL):
    """node positions in finest-grid units ((c + 0.5) * 2^(D-d), in [0, 2^D)) -> Fourier features of pos / 2^(D-1) - 1."""
    return fourier(pos / float(1 << (depth_total - 1)) - 1.0, NF3)


def k_features(g, device):
    """g x g DINOv3 patch grid in row-major (row, col) order -> Fourier features of (u = col, v = row) centres in [-1, 1]."""
    r = (torch.arange(g, device=device, dtype=torch.float32) + 0.5) / g * 2 - 1
    vv, uu = torch.meshgrid(r, r, indexing="ij")
    return fourier(torch.stack([uu.reshape(-1), vv.reshape(-1)], -1), NF2)


def zero_linear(i, o):
    lin = nn.Linear(i, o)
    nn.init.zeros_(lin.weight); nn.init.zeros_(lin.bias)
    return lin


def enabled():
    import os
    return os.environ.get("MESHAXIOM_POSENC_BUILD") == "1"


def fresh_prefixes(depth):
    """Parameter-name prefixes of everything this module adds (for --init-fresh-prefixes and the LR multiplier)."""
    return tuple(f"dit.blocks.{b}.q_pos." for b in range(depth)) + ("cond.encoder.k_pos.",)
