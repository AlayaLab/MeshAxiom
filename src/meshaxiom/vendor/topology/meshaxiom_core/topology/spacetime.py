"""Spacetime intervals: the topology indicator of MeshAxiom (Sec. 3.2).

Per-vertex embeddings are split into a spatial part ``s`` and a temporal part
``t``.  Connectivity is read off from an *indefinite* (Minkowski-signature)
indicator, positive == connected:

  1st order (edges, Eq. 3)
      d_uv = ||s_u - s_v||^2 - ||t_u - t_v||^2

  2nd order (faces, Eq. 5)
      d_uvw = As^2_uvw - At^2_uvw
      where A^2 of (a, b) = ||a||^2 ||b||^2 - <a, b>^2   (4x squared triangle
      area, Eq. 4) evaluated on the spatial / temporal components separately.

Also implemented, for the ablations of Tab. 9:
  * ``minkowski``  -- couples s and t inside a pseudo-Euclidean inner product
                     (Eq. 8/9) instead of decoupling the two determinants.
  * ``euclidean``  -- positive-definite L2 with a learned threshold.
  * ``spherical``  -- cosine similarity.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


# --------------------------------------------------------------------------- #
# indicators
# --------------------------------------------------------------------------- #
def split_st(z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """(..., c) -> spatial (..., c/2), temporal (..., c/2)."""
    c = z.shape[-1]
    assert c % 2 == 0, f"embedding dim must be even, got {c}"
    return z[..., : c // 2], z[..., c // 2 :]


def edge_indicator(z: torch.Tensor, ij: torch.Tensor | None = None, metric: str = "spacetime",
                   thresh: torch.Tensor | float = 0.0) -> torch.Tensor:
    """1st-order indicator.

    ``z``  : (N, c) per-vertex embeddings.
    ``ij`` : (P, 2) vertex pairs, or ``None`` for the dense (N, N) matrix.
    Returns the indicator (logit); > 0 means "edge".
    """
    if metric == "spherical":
        zn = F.normalize(z, dim=-1)
        d = (zn[ij[:, 0]] * zn[ij[:, 1]]).sum(-1) if ij is not None else zn @ zn.T
        return d - thresh
    if metric == "euclidean":
        if ij is not None:
            d = (z[ij[:, 0]] - z[ij[:, 1]]).pow(2).sum(-1)
        else:
            d = torch.cdist(z, z).pow(2)
        # paper-literal (PAPER.txt:373-375): the edge EXISTS when the distance is
        # GREATER than the threshold; the embedding then learns to place connected
        # pairs far apart.  (Physically inverted, but that is what the ablation says.)
        return d - thresh
    s, t = split_st(z)
    if ij is not None:
        ds = (s[ij[:, 0]] - s[ij[:, 1]]).pow(2).sum(-1)
        dt = (t[ij[:, 0]] - t[ij[:, 1]]).pow(2).sum(-1)
    else:
        ds = torch.cdist(s, s).pow(2)
        dt = torch.cdist(t, t).pow(2)
    return ds - dt - thresh


def _sq_area(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """||a||^2 ||b||^2 - <a,b>^2  ==  4 * (triangle area)^2 (Eq. 4)."""
    return a.pow(2).sum(-1) * b.pow(2).sum(-1) - (a * b).sum(-1).pow(2)


def face_indicator(z: torch.Tensor, tri: torch.Tensor, metric: str = "spacetime",
                   thresh: torch.Tensor | float = 0.0) -> torch.Tensor:
    """2nd-order indicator on triplets ``tri`` (T, 3). > 0 means "face"."""
    u, v, w = tri[:, 0], tri[:, 1], tri[:, 2]
    if metric == "spherical":
        zn = F.normalize(z, dim=-1)
        c = ((zn[u] * zn[v]).sum(-1) + (zn[u] * zn[w]).sum(-1) + (zn[v] * zn[w]).sum(-1)) / 3.0
        return c - thresh
    if metric == "euclidean":
        return _sq_area(z[u] - z[w], z[v] - z[w]) - thresh  # paper-literal direction
    s, t = split_st(z)
    a_s = _sq_area(s[u] - s[w], s[v] - s[w])
    a_t = _sq_area(t[u] - t[w], t[v] - t[w])
    if metric == "minkowski":
        # Eq. 8/9: pseudo-Euclidean inner product, determinant of the Gram matrix
        su, sv = s[u] - s[w], s[v] - s[w]
        tu, tv = t[u] - t[w], t[v] - t[w]
        muu = su.pow(2).sum(-1) - tu.pow(2).sum(-1)
        mvv = sv.pow(2).sum(-1) - tv.pow(2).sum(-1)
        muv = (su * sv).sum(-1) - (tu * tv).sum(-1)
        return muu * mvv - muv.pow(2) - thresh
    return a_s - a_t - thresh  # default: 2nd Order Spacetime Interval (Eq. 5)


# --------------------------------------------------------------------------- #
# balanced BCE (Eq. 7)
# --------------------------------------------------------------------------- #
def balanced_bce_terms(logit: torch.Tensor, label: torch.Tensor):
    """Per-group BCE ``(sums, counts)`` over the TP/TN/FP/FN groups of Eq. 7.

    Groups are defined by (hard prediction, ground-truth label).  Returning raw
    sums and counts (instead of the finished mean) lets a DDP caller all-reduce
    the counts and rescale, so that the *global* batch is pooled into four
    groups regardless of how samples were sharded across ranks.
    """
    bce = F.binary_cross_entropy_with_logits(logit, label, reduction="none")
    pred = (logit > 0).float()
    sums, counts = [], []
    for p in (1.0, 0.0):
        for l in (1.0, 0.0):
            m = (pred == p) & (label == l)
            sums.append(bce[m].sum())
            counts.append(m.sum())
    return torch.stack(sums), torch.stack(counts)


def balanced_bce(logit: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
    """Mean of the BCE averaged inside each of the TP/TN/FP/FN groups (Eq. 7).

    Empty groups are skipped.  This is the single-process form; DDP training
    uses :func:`balanced_bce_terms` so the groups pool over the global batch.
    """
    sums, counts = balanced_bce_terms(logit, label)
    ok = counts > 0
    if not ok.any():
        return logit.new_zeros(())
    return (sums[ok] / counts[ok].float()).sum() / int(ok.sum())


def prf(logit: torch.Tensor, label: torch.Tensor) -> dict[str, float]:
    """Precision / recall / F1 of the hard decision ``logit > 0``."""
    pred = logit > 0
    gt = label > 0.5
    tp = (pred & gt).sum().item()
    fp = (pred & ~gt).sum().item()
    fn = (~pred & gt).sum().item()
    p = tp / max(tp + fp, 1)
    r = tp / max(tp + fn, 1)
    return {"precision": p, "recall": r, "f1": 2 * p * r / max(p + r, 1e-9)}


# --------------------------------------------------------------------------- #
# extraction (Sec. 3.2.2): edges first, then 3-cycles verified by 2nd order
# --------------------------------------------------------------------------- #
@torch.no_grad()
def extract_edges(z: torch.Tensor, metric: str = "spacetime", thresh: torch.Tensor | float = 0.0,
                  chunk: int = 4096, max_edges: int = 0,
                  return_info: bool = False) -> torch.Tensor | tuple[torch.Tensor, dict]:
    """All pairs (i < j) whose 1st-order indicator is positive. -> (E, 2) long.

    ``max_edges > 0`` keeps only the strongest that many edges.  An untrained (or
    badly trained) model can declare a near-complete graph, and the 3-cycle
    enumeration that follows is superlinear in the edge count, so the cap is what
    keeps recovery from exhausting memory.  Truncation is by indicator value, not
    by index, which would bias towards low-numbered vertices.
    """
    n = z.shape[0]
    pairs, vals = [], []
    for beg in range(0, n, chunk):
        end = min(beg + chunk, n)
        if metric in ("spacetime", "minkowski"):  # minkowski only changes the face rule
            s, t = split_st(z)
            d = torch.cdist(s[beg:end], s).pow(2) - torch.cdist(t[beg:end], t).pow(2) - thresh
        elif metric == "euclidean":
            # Keep extraction identical to ``edge_indicator``: the paper literally
            # declares an edge when squared Euclidean distance is ABOVE the learned
            # threshold (PAPER.txt:371-375), even though that direction is unusual.
            d = torch.cdist(z[beg:end], z).pow(2) - thresh
        else:
            zn = F.normalize(z, dim=-1)
            d = zn[beg:end] @ zn.T - thresh
        i, j = torch.nonzero(d > 0, as_tuple=True)
        keep = (i + beg) < j
        i, j = i[keep], j[keep]
        pairs.append(torch.stack([i + beg, j], dim=1))
        vals.append(d[i, j])
    if not pairs:
        edges = torch.zeros((0, 2), dtype=torch.long, device=z.device)
        info = {"cap_hit": False, "candidate_edges": 0, "returned_edges": 0,
                "max_edges": int(max_edges)}
        return (edges, info) if return_info else edges
    edges = torch.cat(pairs, 0)
    candidate_edges = len(edges)
    cap_hit = bool(max_edges and candidate_edges > max_edges)
    if cap_hit:
        print(f"WARNING: edge cap hit ({candidate_edges} > {max_edges}); keeping the "
              f"strongest {max_edges} by indicator value", flush=True)
        sel = torch.cat(vals, 0).topk(max_edges).indices
        edges = edges[sel.sort().values]
    info = {"cap_hit": cap_hit, "candidate_edges": int(candidate_edges),
            "returned_edges": int(len(edges)), "max_edges": int(max_edges)}
    return (edges, info) if return_info else edges


@torch.no_grad()
def candidate_triangles(edges: torch.Tensor, num_vertices: int,
                        max_triangles: int = 0,
                        return_info: bool = False) -> torch.Tensor | tuple[torch.Tensor, dict]:
    """All closed 3-cycles (u < v < w) of the edge graph. -> (T, 3) long.

    ``max_triangles > 0`` stops the enumeration once that many are found; the
    count is quadratic in the mean degree, so an unconverged model can otherwise
    produce hundreds of millions of candidates.
    """
    import numpy as np
    from scipy.sparse import csr_matrix

    e = edges.detach().cpu().numpy()
    if len(e) == 0:
        out = torch.zeros((0, 3), dtype=torch.long, device=edges.device)
        info = {"cap_hit": False, "enumerated_candidates": 0,
                "returned_candidates": 0, "max_triangles": int(max_triangles),
                "order_dependent": False}
        return (out, info) if return_info else out
    n = num_vertices
    both = np.concatenate([e, e[:, ::-1]], 0)
    adj = csr_matrix(
        (np.ones(len(both), dtype=np.int8), (both[:, 0], both[:, 1])), shape=(n, n)
    )
    adj.sort_indices()
    indptr, indices = adj.indptr, adj.indices
    tris, total, stopped_early = [], 0, False
    for u, v in e:  # u < v guaranteed by extract_edges
        nu = indices[indptr[u] : indptr[u + 1]]
        nv = indices[indptr[v] : indptr[v + 1]]
        w = np.intersect1d(nu[nu > v], nv[nv > v], assume_unique=True)
        if len(w):
            tris.append(np.stack([np.full(len(w), u), np.full(len(w), v), w], 1))
            total += len(w)
            if max_triangles and total >= max_triangles:
                stopped_early = True  # enumeration is incomplete even if len==cap exactly
                break
    if not tris:
        out = torch.zeros((0, 3), dtype=torch.long, device=edges.device)
        info = {"cap_hit": False, "enumerated_candidates": 0,
                "returned_candidates": 0, "max_triangles": int(max_triangles),
                "order_dependent": False}
        return (out, info) if return_info else out
    out = np.concatenate(tris, 0)
    if stopped_early and not (max_triangles and len(out) > max_triangles):
        print(f"WARNING: 3-cycle enumeration stopped at the cap ({total} found); the "
              f"remaining edges were never scanned - result is vertex-order dependent",
              flush=True)
    if max_triangles and len(out) > max_triangles:
        # the kept subset then depends on vertex numbering (edges are walked in key
        # order), i.e. the sort-free property is lost -- this only happens in the
        # degenerate near-complete-graph regime, but must never happen silently
        print(f"WARNING: 3-cycle cap hit ({len(out)} > {max_triangles}); result is "
              f"vertex-order dependent", flush=True)
        out = out[:max_triangles]
    out = torch.from_numpy(out.astype("int64")).to(edges.device)
    info = {"cap_hit": bool(stopped_early), "enumerated_candidates": int(total),
            "returned_candidates": int(len(out)), "max_triangles": int(max_triangles),
            "order_dependent": bool(stopped_early)}
    return (out, info) if return_info else out


@torch.no_grad()
def extract_mesh_topology(z_edge: torch.Tensor, z_face: torch.Tensor, metric: str = "spacetime",
                          edge_thresh=0.0, face_thresh=0.0, max_edges: int = 4_000_000,
                          max_triangles: int = 20_000_000, chunk: int = 4_000_000,
                          return_info: bool = False):
    """Full "edges-first" recovery.  Returns ``(edges, faces)`` index tensors.

    The two caps and the chunked verification bound the memory of this step
    regardless of how bad the generated embeddings are: a model early in training
    declares an almost complete edge graph, whose 3-cycles do not fit in memory.
    """
    edges, edge_info = extract_edges(z_edge, metric, edge_thresh, max_edges=max_edges,
                                     return_info=True)
    cand, triangle_info = candidate_triangles(
        edges, z_edge.shape[0], max_triangles=max_triangles, return_info=True)
    info = {"edge": edge_info, "triangles": triangle_info,
            "modified_by_cap": bool(edge_info["cap_hit"] or triangle_info["cap_hit"])}
    if len(cand) == 0:
        info["returned_faces"] = 0
        return (edges, cand, info) if return_info else (edges, cand)
    keep = torch.cat([
        face_indicator(z_face, cand[b : b + chunk], metric, face_thresh) > 0
        for b in range(0, len(cand), chunk)
    ])
    faces = cand[keep]
    info["returned_faces"] = int(len(faces))
    return (edges, faces, info) if return_info else (edges, faces)
