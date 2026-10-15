"""The shared MeshAxiom diffusion transformer, plus the stage-1 vertex model.

``MeshAxiomDiT`` is the backbone of *both* diffusion stages (SPEC §4): stage 1 runs
it on 8-value octree occupancy tokens (PAPER.txt:311-314), stage 2 on per-vertex
topology latents.  Tokens carry a 3D RoPE on their spatial position (SPEC §1)
and, when ``max_level > 0``, a learnable octree-depth embedding that is added to
the token stream before all blocks *and* to the adaLN conditioning vector.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from meshaxiom_core.data.mesh_ops import expand_children
from meshaxiom_core.diffusion.flow import RectifiedFlow, occ_threshold
from meshaxiom_core.models.condition import build_condition_encoder
from meshaxiom_core.models.layers import DiTBlock, FinalLayer, RoPE3D, TimestepEmbedder, padding_mask


def cap_occupancy(occ: torch.Tensor, threshold: float, max_nodes: int):
    """Keep at most ``max_nodes`` above-threshold child predictions, exactly.

    Thresholding at the kth value with ``>=`` is not a hard cap when scores tie (a
    common case for an undertrained bf16 model).  Selecting the actual top-k indices
    makes the memory guard independent of ties and returns the pre-cap count for
    provenance.
    """
    before = int((occ > threshold).sum())
    if max_nodes <= 0 or before <= max_nodes:
        return occ, before, False
    flat = occ.flatten()
    keep = flat.topk(max_nodes, sorted=False).indices
    capped = torch.full_like(flat, threshold - 1.0)
    capped[keep] = flat[keep]
    return capped.reshape_as(occ), before, True


class MeshAxiomDiT(nn.Module):
    """DiT over a set of positioned tokens, cross-attending to the condition."""

    def __init__(self, in_dim: int, out_dim: int, dim: int = 1024, depth: int = 24,
                 heads: int = 16, ctx_dim: int | None = None, mlp_mult: float = 4.0,
                 max_level: int = 0, qk_norm: bool = True):
        super().__init__()
        self.dim, self.max_level = dim, max_level
        self.in_proj = nn.Linear(in_dim, dim)
        self.t_embed = TimestepEmbedder(dim)
        self.level_embed = nn.Embedding(max_level + 1, dim) if max_level > 0 else None
        self.rope = RoPE3D(dim // heads)
        self.blocks = nn.ModuleList(
            [DiTBlock(dim, heads, ctx_dim, mlp_mult, qk_norm) for _ in range(depth)]
        )
        self.final = FinalLayer(dim, out_dim)
        # Set by the trainers via train.grad_checkpoint.  The formal 32-rank batch-8
        # path recomputes activations to preserve headroom; the 64-rank batch-4 H200
        # path retains them for speed.  Off by default; inference is unaffected.
        self.grad_checkpoint = False

    def forward(self, x, t, pos, ctx=None, lengths=None, level=None):
        """``x`` (B,N,in_dim), ``t`` (B,) or (B,N), ``pos`` (B,N,3) in finest-grid units."""
        c = self.t_embed(t)
        if c.dim() == 2:
            c = c[:, None]
        h = self.in_proj(x)
        if self.level_embed is not None:
            assert level is not None, "max_level > 0 requires the target octree depth"
            lv = self.level_embed(level)[:, None]
            c = c + lv
            h = h + lv
        rope = self.rope.cos_sin(pos)
        mask = padding_mask(lengths, x.shape[1]) if lengths is not None else None
        if self.grad_checkpoint and self.training and torch.is_grad_enabled():
            from torch.utils.checkpoint import checkpoint
            for blk in self.blocks:
                # non-reentrant is required under DDP (reentrant breaks its grad hooks)
                h = checkpoint(blk, h, c, rope=rope, mask=mask, ctx=ctx,
                               use_reentrant=False)
        else:
            for blk in self.blocks:
                h = blk(h, c, rope=rope, mask=mask, ctx=ctx)
        return self.final(h, c)


def node_positions(coords, depth_total: int, depth: int) -> torch.Tensor:
    """Canonical token position of octree nodes (SPEC §1): ``(c + 0.5) * 2**(D-d)``."""
    return (coords.float() + 0.5) * float(1 << (depth_total - depth))


class VertexModel(nn.Module):
    """Stage-1 bundle: condition encoder + ``MeshAxiomDiT`` over occupancy tokens.

    Also owns the coarse-to-fine sampling loop of PAPER.txt:319-324 so that
    training, tests and ``meshaxiom_core/sample.py`` share one implementation.
    """

    def __init__(self, cfg):
        super().__init__()
        self.cond, ctx_dim = build_condition_encoder(cfg)
        m = dict(cfg["model"])
        self.depth_total = int(cfg["depth"])
        self.target_mode = str(cfg.get("diffusion", {}).get("target_mode", "unit"))
        self.dit = MeshAxiomDiT(
            in_dim=8, out_dim=8, dim=int(m.get("dim", 1024)), depth=int(m.get("depth", 24)),
            heads=int(m.get("heads", 16)), ctx_dim=ctx_dim, mlp_mult=float(m.get("mlp_mult", 4.0)),
            max_level=self.depth_total, qk_norm=bool(m.get("qk_norm", True)),
        )

    def forward(self, x, t, parents, level, lengths=None, ctx=None, points=None, normals=None,
                image=None, camera=None, lod=None, drop=None):
        """``parents`` (B,N,3) node coords at depth ``level-1`` (one depth per batch).

        The condition is encoded inside ``forward`` (unless ``ctx`` is given) so
        that a single DDP-wrapped call covers the encoder as well.
        """
        if ctx is None:
            ctx = self.cond(points=points, normals=normals, image=image,
                            camera=camera, lod=lod, drop=drop)
        pos = node_positions(parents, self.depth_total, int(level[0]) - 1)
        return self.dit(x, t, pos, ctx=ctx, lengths=lengths, level=level)

    @torch.no_grad()
    def sample_octree(self, ctx, steps: int = 20, sampler: str = "dpmpp2m", cfg_scale: float = 1.0,
                      uncond_ctx=None, max_nodes: int = 0, flow: RectifiedFlow | None = None,
                      generator=None, level_generators=None, verbose: bool = False,
                      return_info: bool = False,
                      occ_thr: float | None = None):
        """Coarse-to-fine octree sampling from a single condition (batch size 1).

        Returns the occupied node coordinates of every level 0..D; the last entry
        is the generated vertex set.
        """
        assert ctx.shape[0] == 1, "coarse-to-fine sampling runs one shape at a time"
        flow = flow or RectifiedFlow()
        dev = ctx.device
        # SPEC's decision threshold (0.5 for unit targets) is the default; FINDINGS
        # Sec. 47 localised the dominant error to missing stage-1 vertices, and a lower
        # threshold keeps borderline occupancies, so it is exposed for sweeping.
        thr = occ_threshold(self.target_mode) if occ_thr is None else float(occ_thr)
        if level_generators is not None:
            if generator is not None:
                raise ValueError("pass either generator or level_generators, not both")
            if len(level_generators) != self.depth_total:
                raise ValueError(f"expected {self.depth_total} level generators, got "
                                 f"{len(level_generators)}")
        levels = [np.zeros((1, 3), dtype=np.int64)]
        info = {"max_nodes": int(max_nodes), "cap_events": [], "empty_fallback_events": []}
        for d in range(1, self.depth_total + 1):
            parents = levels[-1]
            pc = torch.from_numpy(parents).to(dev)[None]
            pos = node_positions(pc, self.depth_total, d - 1)
            lvl = torch.full((1,), d, dtype=torch.long, device=dev)

            def model_fn(x, t, _pos=pos, _lvl=lvl, _ctx=ctx):
                return self.dit(x, t, _pos, ctx=_ctx, level=_lvl)

            unc = None
            if cfg_scale != 1.0:
                uc = uncond_ctx if uncond_ctx is not None else self.cond.null_ctx(1).to(ctx.dtype)

                def unc(x, t, _pos=pos, _lvl=lvl, _uc=uc):
                    return self.dit(x, t, _pos, ctx=_uc, level=_lvl)

            level_generator = (level_generators[d - 1]
                               if level_generators is not None else generator)
            x0 = torch.randn(1, len(parents), 8, device=dev, generator=level_generator)
            occ = flow.sample(model_fn, x0, steps=steps, sampler=sampler,
                              cfg_scale=cfg_scale, uncond_fn=unc)[0].float()
            occ, before, capped = cap_occupancy(occ, thr, max_nodes)
            if capped:
                info["cap_events"].append(
                    {"level": d, "predicted_nodes": before, "kept_nodes": int(max_nodes)})
                print(f"  [level {d}] node cap: {before} -> {max_nodes}")
            child = expand_children(parents, occ.cpu().numpy(), thr)
            if len(child) == 0:  # guard (SPEC §9): fall back to the top-1 child per parent
                info["empty_fallback_events"].append(
                    {"level": d, "parents": int(len(parents)),
                     "policy": "argmax_child_per_parent"})
                slot = occ.argmax(1, keepdim=True)
                onehot = torch.zeros_like(occ).scatter_(1, slot, 1.0)
                child = expand_children(parents, onehot.cpu().numpy(), 0.5)
                print(f"  [level {d}] empty prediction, fell back to argmax children")
            levels.append(child)
            if verbose:
                print(f"  [level {d}] {len(parents):6d} parents -> {len(child):6d} nodes")
        return (levels, info) if return_info else levels
