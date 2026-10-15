"""Topology latent diffusion (Sec. 3.2.1, PAPER.txt:477-483).

A thin wrapper over the shared :class:`~meshaxiom_core.models.octree_dit.MeshAxiomDiT`
backbone: the tokens are the *vertices*, so ``in_dim = out_dim = latent_dim``,
the 3D RoPE runs on the vertex positions and the condition ``C`` is injected by
cross-attention exactly as in the vertex stage.  There is no octree level to
disambiguate here, hence ``max_level=0``.

The wrapper also owns the latent normalisation: the topology VAE latents are
whitened by a dataset-wide ``(mean, std)`` (computed once, stored in the VAE
checkpoint, SPEC 7) so that the flow-matching target has ~unit variance.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from meshaxiom_core.diffusion.flow import RectifiedFlow
from meshaxiom_core.models.condition import build_condition_encoder
from meshaxiom_core.models.octree_dit import MeshAxiomDiT


class TopoDiT(nn.Module):
    def __init__(self, latent_dim: int = 64, dim: int = 2048, depth: int = 24, heads: int = 16,
                 ctx_dim: int | None = None, mlp_mult: float = 4.0, qk_norm: bool = True):
        super().__init__()
        self.latent_dim = latent_dim
        self.net = MeshAxiomDiT(in_dim=latent_dim, out_dim=latent_dim, dim=dim, depth=depth,
                            heads=heads, ctx_dim=ctx_dim, mlp_mult=mlp_mult, max_level=0,
                            qk_norm=qk_norm)
        self.register_buffer("latent_mean", torch.zeros(latent_dim))
        self.register_buffer("latent_std", torch.ones(latent_dim))

    @torch.no_grad()
    def set_latent_stats(self, mean, std, eps: float = 1e-6):
        m = torch.as_tensor(mean, dtype=torch.float32).reshape(-1)
        s = torch.as_tensor(std, dtype=torch.float32).reshape(-1).clamp_min(eps)
        assert m.numel() == s.numel() == self.latent_dim, (m.shape, s.shape, self.latent_dim)
        self.latent_mean.copy_(m.to(self.latent_mean.device))
        self.latent_std.copy_(s.to(self.latent_std.device))

    def normalize_latent(self, h: torch.Tensor) -> torch.Tensor:
        return (h - self.latent_mean) / self.latent_std

    def denormalize_latent(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.latent_std + self.latent_mean

    def forward(self, x, t, pos, ctx=None, lengths=None):
        """``x`` (B,V,latent_dim) noisy latents, ``pos`` (B,V,3) in finest-grid units."""
        return self.net(x, t, pos, ctx=ctx, lengths=lengths)


def as_latent_stats(obj) -> dict | None:
    """Normalise a checkpoint's ``latent_stats`` entry to ``{"mean": ..., "std": ...}``."""
    if obj is None:
        return None
    return {"mean": obj["mean"], "std": obj["std"]} if isinstance(obj, dict) else \
           {"mean": obj[0], "std": obj[1]}


def build_topo_dit(cfg, ctx_dim: int | None) -> TopoDiT:
    m = dict(cfg.get("model", {}))
    return TopoDiT(latent_dim=int(m.get("latent_dim", 64)), dim=int(m.get("dim", 2048)),
                   depth=int(m.get("depth", 24)), heads=int(m.get("heads", 16)),
                   ctx_dim=ctx_dim, mlp_mult=float(m.get("mlp_mult", 4.0)),
                   qk_norm=bool(m.get("qk_norm", True)))


class TopologyModel(nn.Module):
    """Stage-2 bundle: condition encoder + ``TopoDiT`` over the per-vertex latents.

    Mirrors :class:`~meshaxiom_core.models.octree_dit.VertexModel` so that both stages
    have the same checkpoint layout (``cond.*`` / ``dit.*``) and both training,
    tests and ``meshaxiom_core/sample.py`` go through one sampling implementation.
    """

    def __init__(self, cfg):
        super().__init__()
        self.cond, ctx_dim = build_condition_encoder(cfg)
        self.depth_total = int(cfg["depth"])
        self.dit = build_topo_dit(cfg, ctx_dim)

    @property
    def latent_dim(self) -> int:
        return self.dit.latent_dim

    def forward(self, x, t, coords, lengths=None, ctx=None, points=None, normals=None,
                image=None, camera=None, lod=None, drop=None):
        """``coords`` (B,V,3) integer vertex coords at depth D (SPEC 1: pos = c + 0.5)."""
        if ctx is None:
            ctx = self.cond(points=points, normals=normals, image=image,
                            camera=camera, lod=lod, drop=drop)
        return self.dit(x, t, coords.float() + 0.5, ctx=ctx, lengths=lengths)

    @torch.no_grad()
    def sample_latents(self, ctx, coords, steps: int = 20, sampler: str = "dpmpp2m",
                       cfg_scale: float = 1.0, uncond_ctx=None, lengths=None,
                       flow: RectifiedFlow | None = None, generator=None) -> torch.Tensor:
        """-> the *denormalised* per-vertex latents ``H_V`` (B, V, latent_dim)."""
        flow = flow or RectifiedFlow()
        b, v, _ = coords.shape
        x0 = torch.randn(b, v, self.latent_dim, device=coords.device, generator=generator)

        def model_fn(x, t, _c=ctx):
            return self.forward(x, t, coords, lengths=lengths, ctx=_c)

        unc = None
        if cfg_scale != 1.0:
            uc = uncond_ctx if uncond_ctx is not None else self.cond.null_ctx(b).to(ctx.dtype)

            def unc(x, t, _uc=uc):
                return self.forward(x, t, coords, lengths=lengths, ctx=_uc)

        x1 = flow.sample(model_fn, x0, steps=steps, sampler=sampler, cfg_scale=cfg_scale,
                         uncond_fn=unc)
        return self.dit.denormalize_latent(x1)
