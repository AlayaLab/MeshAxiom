"""Batched farthest-point sampling (query selection of the VecSet encoder).

Plain PyTorch: one ``argmax`` per selected point, so it runs on GPU without a
custom kernel.  Cost is ``O(k * P)`` distance evaluations per batch element.
"""

from __future__ import annotations

import torch


def _start_index(pts: torch.Tensor, start: str, generator=None) -> torch.Tensor:
    b = pts.shape[0]
    if start == "farthest":
        # farthest point from the centroid: permutation invariant, so the whole
        # FPS trajectory (set *and* order) is invariant to the input ordering.
        return (pts - pts.mean(1, keepdim=True)).pow(2).sum(-1).argmax(1)
    if start == "random":
        return torch.randint(0, pts.shape[1], (b,), device=pts.device, generator=generator)
    if start == "first":
        return torch.zeros(b, dtype=torch.long, device=pts.device)
    raise ValueError(f"unknown FPS start rule {start!r}")


@torch.no_grad()
def farthest_point_sample(points: torch.Tensor, num_samples: int, start: str = "farthest",
                          generator=None) -> torch.Tensor:
    """``points`` (B, P, 3) -> (B, num_samples) long indices, greedy FPS.

    ``start`` selects the seed: ``"farthest"`` (from the centroid, default —
    deterministic and permutation invariant), ``"random"`` or ``"first"``.
    If ``num_samples > P`` the index list simply cycles.
    """
    b, p, _ = points.shape
    if num_samples >= p:
        return (torch.arange(num_samples, device=points.device) % p).expand(b, -1).contiguous()
    pts = points.float()
    idx = torch.empty(b, num_samples, dtype=torch.long, device=points.device)
    dist = torch.full((b, p), float("inf"), device=points.device)
    cur = _start_index(pts, start, generator)
    for i in range(num_samples):
        idx[:, i] = cur
        d = (pts - pts.gather(1, cur[:, None, None].expand(b, 1, 3))).pow(2).sum(-1)
        dist = torch.minimum(dist, d)
        cur = dist.argmax(1)
    return idx


@torch.no_grad()
def sample_query_indices(points: torch.Tensor, num_samples: int, method: str = "fps",
                         start: str = "farthest", generator=None) -> torch.Tensor:
    """FPS with a ``"random"`` fallback (SPEC §3), for ablations / speed tests."""
    if method == "fps":
        import os
        if os.environ.get("MESHAXIOM_TRITON_FPS") == "1" and points.is_cuda:   # V197: the V153 Triton kernel (same greedy FPS and tie rule; r2 trained with it)
            from meshaxiom_core.models.fps_fast153 import farthest_point_sample_fast
            return farthest_point_sample_fast(points, num_samples, start, generator)
        return farthest_point_sample(points, num_samples, start, generator)
    if method == "random":
        b, p, _ = points.shape
        r = torch.rand(b, p, device=points.device, generator=generator)
        return r.argsort(dim=1)[:, :num_samples]
    raise ValueError(f"unknown query sampling method {method!r}")
