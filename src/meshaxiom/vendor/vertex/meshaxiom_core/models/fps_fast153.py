"""(V197 copy of meshaxiom_trainer_speed_20260924_v153/fps_fast.py, unchanged code) Fast farthest-point sampling (09-24, V153): one Triton program per batch element keeps all P points and the running min-distance
vector in registers and does the k greedy iterations with block-wide reductions, instead of the 1024-iteration Python loop of
meshaxiom_core/models/fps.py (51 ms for B=2, P=8192, k=1024 on H200).  Same algorithm and tie rule (first index among the maxima, like
torch.argmax); the start index is computed exactly as meshaxiom_core.models.fps._start_index.  Falls back to the reference loop when Triton is
unavailable or P > 16384."""
import torch

try:
    import triton, triton.language as tl
    _HAVE_TRITON = True
except Exception:  # pragma: no cover
    _HAVE_TRITON = False


if _HAVE_TRITON:
    @triton.jit
    def _fps_kernel(pts_ptr, start_ptr, out_ptr, P: tl.constexpr, K: tl.constexpr, BLOCK: tl.constexpr):
        b = tl.program_id(0)
        idx = tl.arange(0, BLOCK)
        m = idx < P
        base = pts_ptr + b * P * 3
        x = tl.load(base + idx * 3 + 0, mask=m, other=0.0)
        y = tl.load(base + idx * 3 + 1, mask=m, other=0.0)
        z = tl.load(base + idx * 3 + 2, mask=m, other=0.0)
        dist = tl.full([BLOCK], float("inf"), tl.float32)
        cur = tl.load(start_ptr + b)
        for i in range(K):
            tl.store(out_ptr + b * K + i, cur)
            sel = idx == cur
            cx = tl.sum(tl.where(sel, x, 0.0)); cy = tl.sum(tl.where(sel, y, 0.0)); cz = tl.sum(tl.where(sel, z, 0.0))
            dx = x - cx; dy = y - cy; dz = z - cz
            d = dx * dx + dy * dy + dz * dz
            dist = tl.minimum(dist, d)
            dist = tl.where(m, dist, -1.0)          # padded slots never win
            best = tl.max(dist)
            cur = tl.min(tl.where(dist == best, idx, BLOCK))


def _start_index(pts, start, generator=None):
    from meshaxiom_core.models.fps import _start_index as ref
    return ref(pts, start, generator)


@torch.no_grad()
def farthest_point_sample_fast(points, num_samples, start="farthest", generator=None):
    b, p, _ = points.shape
    if num_samples >= p or not _HAVE_TRITON or p > 16384 or not points.is_cuda:
        from meshaxiom_core.models.fps import farthest_point_sample
        return farthest_point_sample(points, num_samples, start, generator)
    pts = points.float().contiguous()
    cur0 = _start_index(pts, start, generator).to(torch.int32).contiguous()
    out = torch.empty(b, num_samples, dtype=torch.int32, device=points.device)
    block = triton.next_power_of_2(p)
    _fps_kernel[(b,)](pts, cur0, out, P=p, K=num_samples, BLOCK=block, num_warps=16)
    return out.long()
