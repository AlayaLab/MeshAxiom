"""Exact all-pair spacetime BCE statistics without an O(V^2) saved graph.

Every unordered pair i < j is evaluated, once.  The forward returns TP/FP/FN/TN
loss sums and counts; the caller pools these across meshes/ranks BEFORE dividing.
Backward recomputes tiles and uses the analytic derivative of the quadratic
spacetime interval.  This is an execution strategy, not negative subsampling.
Only first derivatives and the fixed-threshold spacetime metric are supported.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch.autograd.function import once_differentiable


def _tiles(z, edge_keys, chunk, threshold):
    n, dim = z.shape
    sign = torch.ones(dim, device=z.device, dtype=z.dtype)
    sign[dim // 2:] = -1
    signed = z * sign
    norms = (signed * z).sum(-1)
    src, dst = edge_keys // n, edge_keys % n
    for beg in range(0, n - 1, chunk):
        end = min(beg + chunk, n)
        col = beg + 1
        logit = (norms[beg:end, None] + norms[None, col:]
                 - 2 * (signed[beg:end] @ z[col:].T) - threshold)
        valid = (torch.arange(beg, end, device=z.device)[:, None]
                 < torch.arange(col, n, device=z.device)[None, :])
        truth = torch.zeros_like(valid)
        use = (src >= beg) & (src < end)
        truth[src[use] - beg, dst[use] - col] = True
        yield beg, end, col, logit, valid, truth, signed


class _StreamedEdgeTerms(torch.autograd.Function):
    @staticmethod
    def forward(ctx, z, edge_keys, chunk, threshold):
        ctx.save_for_backward(z, edge_keys)
        ctx.chunk, ctx.threshold = chunk, threshold
        sums = torch.zeros(4, dtype=torch.float64, device=z.device)
        counts = torch.zeros(4, dtype=torch.long, device=z.device)
        with torch.autocast(z.device.type, enabled=False):
            for _, _, _, logit, valid, truth, _ in _tiles(z, edge_keys, chunk, threshold):
                pred = logit > 0
                bce = F.softplus(torch.where(truth, -logit, logit))
                for group, (p, t) in enumerate(((True, True), (True, False),
                                              (False, True), (False, False))):
                    mask = valid & (pred == p) & (truth == t)
                    sums[group] += bce[mask].sum(dtype=torch.float64)
                    counts[group] += mask.sum()
        ctx.mark_non_differentiable(counts)
        return sums.to(z.dtype), counts

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_sums, _grad_counts):
        z, edge_keys = ctx.saved_tensors
        grad = torch.zeros_like(z)
        if grad_sums is None:
            return grad, None, None, None
        with torch.autocast(z.device.type, enabled=False):
            weights = grad_sums.to(z.dtype)
            for beg, end, col, logit, valid, truth, signed in _tiles(
                    z, edge_keys, ctx.chunk, ctx.threshold):
                pred = logit > 0
                group = (~pred).long() * 2 + (~truth).long()
                derivative = ((logit.sigmoid() - truth.to(z.dtype))
                              * weights[group] * valid)
                grad[beg:end] += 2 * (
                    derivative.sum(1)[:, None] * signed[beg:end]
                    - derivative @ signed[col:])
                grad[col:] += 2 * (
                    derivative.sum(0)[:, None] * signed[col:]
                    - derivative.T @ signed[beg:end])
        return grad, None, None, None


def streamed_edge_terms(z, edge_keys, chunk=512, threshold=0.0):
    """Return four BCE sums and int64 counts, preserving gradients to z.

    edge_keys must contain each GT unordered edge once, encoded as i*V+j,
    with 0 <= i < j < V.  TopologyLoss._edge_keys already provides this form.
    No learned threshold or non-spacetime metric is silently approximated.
    """
    if z.ndim != 2 or z.shape[1] % 2 or z.dtype not in (torch.float32, torch.float64):
        raise ValueError('expected [V, even C] float32/64 spacetime embeddings')
    if z.shape[0] < 1 or not isinstance(chunk, int) or chunk < 1:
        raise ValueError('V and chunk must be positive')
    if edge_keys.ndim != 1 or edge_keys.dtype != torch.long or edge_keys.device != z.device:
        raise ValueError('edge keys must be 1D int64 on the embedding device')
    if isinstance(threshold, torch.Tensor):
        if threshold.numel() != 1 or threshold.requires_grad:
            raise ValueError('only a fixed scalar threshold is supported')
        threshold = float(threshold)
    return _StreamedEdgeTerms.apply(z, edge_keys, chunk, float(threshold))
