"""P5 — fast, bit-identical CPU post-processing for the Topology stage (meshaxiom_e2e_20261009, 2026-10-15).

What the shipped post-processing does after the DiT (release decode_policy90.decode, policy v4 / p2_cfg1_s20):
  VAE decode x2 (GPU) -> extract_mesh_topology x2 (GPU cdist edge test + CPU 3-cycle enumeration + GPU face test)
  -> prune_edge_capacity x2 (GPU score + CPU greedy) -> merge_faces_by_capacity (GPU scores + CPU greedy, attached)
  -> torch.unique of the edges -> mesh_stats (CPU) -> export_faces / orient_faces (CPU) -> mesh.obj / mesh.json / topology.npz.
P0's profile and P1's strict timing: the CPU loops (Python, per edge / per face) are 0.25-0.3 s per request (P2 row), the GPU
parts a few ms, the VAE 0.022 s.

This module replaces ONLY the integer / ordering work with vectorised numpy (+ numba when available; same semantics in pure
Python otherwise).  Every floating-point decision stays in the release's own calls on the same device, dtype and tensor shapes
(extract_edges' cdist test, face_indicator on the identical candidate / face tensors, mesh_stats' numpy expressions verbatim,
orient_faces' per-piece numpy volume), so the outputs are bit-identical by construction and checked against the references
(code/test_p5_cpu.py on CPU, code/p5_dump_post_inputs.py on GPU, p5/CORRECTNESS.md for dev100 400/400):
  candidate_triangles_fast    : all 3-cycles (u, v, w>v adjacent to both) of the edge graph, in the release's order (edge order, w ascending)
  prune_edge_capacity_fast    : greedy per-edge capacity in np.argsort(-score, stable) order over integer edge ids
  merge_faces_by_capacity_fast: the same greedy (primary one pass, fallback up to 64 attached rounds) over integer edge ids
  mesh_stats_fast             : key-based unique faces / edges, scipy connected components, the release's float expressions
  export_faces_fast           : CSR adjacency + numba BFS in the release's visiting order, per-piece outward test unchanged
  write_obj_fast              : the same text built in memory and written once
The release files are never edited: `patched()` swaps the module attributes the release / harness code looks up at call time
(spacetime.candidate_triangles, decode_policy90.prune_edge_capacity / merge_faces_by_capacity, e2e_common.mesh_stats /
export_faces / write_obj) and restores them on exit — the same swap-and-restore style as code/topo_fast.py.
Rules: new file of the study; release code, topo_fast.py and e2e_common.py read-only; CPU tests at nice 10 on 2 cores.
"""
from __future__ import annotations

import contextlib
import hashlib
import importlib.util
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

from meshaxiom.vendor.recipe import decode_policy90 as DP
from meshaxiom.vendor.recipe import mesh_utils90 as MU
import meshaxiom_core.topology.spacetime as ST  # noqa: E402
from meshaxiom_core.topology.spacetime import face_indicator  # noqa: E402

USE_NUMBA = os.environ.get('P5_NUMBA', '1') != '0'
try:
    if USE_NUMBA:
        from numba import njit  # noqa: E402
    HAVE_NUMBA = bool(USE_NUMBA)
except Exception:  # noqa: BLE001
    HAVE_NUMBA = False
    USE_NUMBA = False

VERSION = 'meshaxiom-e2e-p5-post-fast-v1'


# ----------------------------------------------------------------------------------------------- timers
TIMERS = {}


def reset_timers():
    TIMERS.clear()


def timers():
    return dict(TIMERS)


def _acc(name, dt):
    TIMERS[name] = TIMERS.get(name, 0.0) + dt
    TIMERS[name + '_n'] = TIMERS.get(name + '_n', 0) + 1


# ----------------------------------------------------------------------------------------------- 3-cycle enumeration
def enumerate_3cycles(e, n):
    """e (E, 2) int64 (rows in the order they are given), n vertices -> (tris (T, 3) int64, kept count per edge (E,) int64).
    For edge i = (u, v): every w > v adjacent to both u and v, w ascending = the release's
    np.intersect1d(N(u)[N(u) > v], N(v)[N(v) > v]) in edge order (candidate_triangles walks the edges in order)."""
    e = np.ascontiguousarray(np.asarray(e, dtype=np.int64)).reshape(-1, 2)
    E = len(e)
    if E == 0:
        return np.zeros((0, 3), np.int64), np.zeros(0, np.int64)
    n = int(n)
    both = np.concatenate([e, e[:, ::-1]], 0)
    key = np.unique(both[:, 0] * n + both[:, 1])                 # symmetric adjacency: unique (row, col), sorted = CSR with sorted columns
    rows = key // n
    cols = key - rows * n
    indptr = np.searchsorted(rows, np.arange(n + 1, dtype=np.int64))
    ar = np.arange(n, dtype=np.int64)
    gt = np.searchsorted(key, ar * n + ar + 1)                    # first position of a neighbour > r inside row r (== indptr[r+1] if none)
    u, v = e[:, 0], e[:, 1]
    start = gt[v]
    cnt = indptr[v + 1] - start                                   # |{w in N(v): w > v}| per edge
    total = int(cnt.sum())
    if total == 0:
        return np.zeros((0, 3), np.int64), np.zeros(E, np.int64)
    idx = np.repeat(np.arange(E, dtype=np.int64), cnt)
    off = np.cumsum(cnt) - cnt
    within = np.arange(total, dtype=np.int64) - np.repeat(off, cnt)
    w = cols[start[idx] + within]
    uu = u[idx]
    q = uu * n + w                                                # (u, w) must be an edge too
    pos = np.minimum(np.searchsorted(key, q), len(key) - 1)
    keep = key[pos] == q
    tris = np.stack([uu[keep], v[idx][keep], w[keep]], 1)
    return tris, np.bincount(idx[keep], minlength=E).astype(np.int64)


@torch.no_grad()
def candidate_triangles_fast(edges, num_vertices, max_triangles=0, return_info=False):
    """= meshaxiom_core.topology.spacetime.candidate_triangles (same output tensor, order, dtype, device, info dict and warnings)."""
    t0 = time.perf_counter()
    e = edges.detach().cpu().numpy()
    if len(e) == 0:
        out = torch.zeros((0, 3), dtype=torch.long, device=edges.device)
        info = {"cap_hit": False, "enumerated_candidates": 0, "returned_candidates": 0, "max_triangles": int(max_triangles), "order_dependent": False}
        _acc('enum_s', time.perf_counter() - t0)
        return (out, info) if return_info else out
    tris, cnt = enumerate_3cycles(e, num_vertices)
    total = int(cnt.sum())
    stopped_early = False
    if total == 0:
        out = torch.zeros((0, 3), dtype=torch.long, device=edges.device)
        info = {"cap_hit": False, "enumerated_candidates": 0, "returned_candidates": 0, "max_triangles": int(max_triangles), "order_dependent": False}
        _acc('enum_s', time.perf_counter() - t0)
        return (out, info) if return_info else out
    out = tris
    if max_triangles:
        cum = np.cumsum(cnt)
        hit = np.flatnonzero(cum >= max_triangles)
        if len(hit):                                              # the release breaks after the first edge that reaches the cap
            k = int(hit[0]); total = int(cum[k]); out = tris[:total]; stopped_early = True
    if stopped_early and not (max_triangles and len(out) > max_triangles):
        print(f"WARNING: 3-cycle enumeration stopped at the cap ({total} found); the "
              f"remaining edges were never scanned - result is vertex-order dependent", flush=True)
    if max_triangles and len(out) > max_triangles:
        print(f"WARNING: 3-cycle cap hit ({len(out)} > {max_triangles}); result is "
              f"vertex-order dependent", flush=True)
        out = out[:max_triangles]
    out_t = torch.from_numpy(np.ascontiguousarray(out.astype("int64"))).to(edges.device)
    info = {"cap_hit": bool(stopped_early), "enumerated_candidates": int(total), "returned_candidates": int(len(out)),
            "max_triangles": int(max_triangles), "order_dependent": bool(stopped_early)}
    _acc('enum_s', time.perf_counter() - t0)
    return (out_t, info) if return_info else out_t


# ----------------------------------------------------------------------------------------------- greedy capacity loops
def _edge_ids(fs):
    """fs (F, 3) int64 with sorted rows -> (ids (F, 3) int64 into the unique edge table, number of unique edges).
    Edge keys as the release: (a, b), (b, c), (a, c) of the sorted triple."""
    if len(fs) == 0:
        return np.zeros((0, 3), np.int64), 0
    n = int(fs.max()) + 1
    k = np.stack([fs[:, 0] * n + fs[:, 1], fs[:, 1] * n + fs[:, 2], fs[:, 0] * n + fs[:, 2]], 1)
    uk, inv = np.unique(k, return_inverse=True)
    return np.ascontiguousarray(inv.reshape(k.shape).astype(np.int64)), int(len(uk))


def _greedy_cap_py(order, eid, n_edges, cap):
    count = np.zeros(max(n_edges, 1), np.int32); keep = np.zeros(len(eid), bool)
    for i in order.tolist():
        a, b, c = int(eid[i, 0]), int(eid[i, 1]), int(eid[i, 2])
        if count[a] < cap and count[b] < cap and count[c] < cap:
            keep[i] = True; count[a] += 1; count[b] += 1; count[c] += 1
    return keep


def _greedy_merge_py(order_a, eid_a, order_b, eid_b, n_edges, cap, attached, max_rounds):
    count = np.zeros(max(n_edges, 1), np.int32)
    out_a = []
    for i in order_a.tolist():
        a, b, c = int(eid_a[i, 0]), int(eid_a[i, 1]), int(eid_a[i, 2])
        if count[a] < cap and count[b] < cap and count[c] < cap:
            out_a.append(i); count[a] += 1; count[b] += 1; count[c] += 1
    out_b = []
    order = order_b.tolist()
    for _ in range(max_rounds):
        rest = []; added = 0
        for i in order:
            a, b, c = int(eid_b[i, 0]), int(eid_b[i, 1]), int(eid_b[i, 2])
            if not (count[a] < cap and count[b] < cap and count[c] < cap):
                continue
            if attached and not (count[a] >= 1 or count[b] >= 1 or count[c] >= 1):
                rest.append(i); continue
            out_b.append(i); added += 1; count[a] += 1; count[b] += 1; count[c] += 1
        order = rest
        if not added or not order:
            break
    return np.asarray(out_a, np.int64), np.asarray(out_b, np.int64)


def _bfs_orient_py(indptr, dst, same, m):
    from collections import deque
    flip = np.full(m, -1, np.int8); piece = np.full(m, -1, np.int64); npieces = 0; bad = np.zeros(m, bool)
    for s0 in range(m):
        if flip[s0] >= 0:
            continue
        flip[s0] = 0; piece[s0] = npieces; q = deque([s0])
        while q:
            f = q.popleft()
            for p in range(int(indptr[f]), int(indptr[f + 1])):
                g = int(dst[p]); want = int(flip[f]) ^ (1 if same[p] else 0)
                if flip[g] < 0:
                    flip[g] = want; piece[g] = npieces; q.append(g)
                elif flip[g] != want:
                    bad[npieces] = True
        npieces += 1
    return flip, piece, npieces, bad


if HAVE_NUMBA:
    @njit(cache=True)
    def _greedy_cap_nb(order, eid, n_edges, cap):
        count = np.zeros(max(n_edges, 1), np.int32)
        keep = np.zeros(eid.shape[0], np.bool_)
        for t in range(order.shape[0]):
            i = order[t]
            a = eid[i, 0]; b = eid[i, 1]; c = eid[i, 2]
            if count[a] < cap and count[b] < cap and count[c] < cap:
                keep[i] = True
                count[a] += 1; count[b] += 1; count[c] += 1
        return keep

    @njit(cache=True)
    def _greedy_merge_nb(order_a, eid_a, order_b, eid_b, n_edges, cap, attached, max_rounds):
        count = np.zeros(max(n_edges, 1), np.int32)
        out_a = np.empty(order_a.shape[0], np.int64); na = 0
        for t in range(order_a.shape[0]):
            i = order_a[t]
            a = eid_a[i, 0]; b = eid_a[i, 1]; c = eid_a[i, 2]
            if count[a] < cap and count[b] < cap and count[c] < cap:
                out_a[na] = i; na += 1
                count[a] += 1; count[b] += 1; count[c] += 1
        out_b = np.empty(order_b.shape[0], np.int64); nb = 0
        order = order_b.copy(); m = order.shape[0]
        rest = np.empty(m, np.int64)
        for r in range(max_rounds):
            nrest = 0; added = 0
            for t in range(m):
                i = order[t]
                a = eid_b[i, 0]; b = eid_b[i, 1]; c = eid_b[i, 2]
                if not (count[a] < cap and count[b] < cap and count[c] < cap):
                    continue
                if attached and not (count[a] >= 1 or count[b] >= 1 or count[c] >= 1):
                    rest[nrest] = i; nrest += 1
                    continue
                out_b[nb] = i; nb += 1; added += 1
                count[a] += 1; count[b] += 1; count[c] += 1
            for t in range(nrest):
                order[t] = rest[t]
            m = nrest
            if added == 0 or m == 0:
                break
        return out_a[:na], out_b[:nb]

    @njit(cache=True)
    def _bfs_orient_nb(indptr, dst, same, m):
        flip = np.full(m, -1, np.int8); piece = np.full(m, -1, np.int64); npieces = 0
        bad = np.zeros(m, np.bool_)
        q = np.empty(m, np.int64)
        for s0 in range(m):
            if flip[s0] >= 0:
                continue
            flip[s0] = 0; piece[s0] = npieces
            head = 0; tail = 0; q[tail] = s0; tail += 1
            while head < tail:
                f = q[head]; head += 1
                for p in range(indptr[f], indptr[f + 1]):
                    g = dst[p]
                    want = flip[f] ^ (1 if same[p] else 0)
                    if flip[g] < 0:
                        flip[g] = want; piece[g] = npieces
                        q[tail] = g; tail += 1
                    elif flip[g] != want:
                        bad[npieces] = True
            npieces += 1
        return flip, piece, npieces, bad


def greedy_cap(order, eid, n_edges, cap):
    if USE_NUMBA and HAVE_NUMBA:
        return _greedy_cap_nb(np.ascontiguousarray(order, dtype=np.int64), np.ascontiguousarray(eid, dtype=np.int64), int(n_edges), int(cap))
    return _greedy_cap_py(order, eid, n_edges, cap)


def greedy_merge(order_a, eid_a, order_b, eid_b, n_edges, cap, attached, max_rounds):
    if USE_NUMBA and HAVE_NUMBA:
        return _greedy_merge_nb(np.ascontiguousarray(order_a, dtype=np.int64), np.ascontiguousarray(eid_a, dtype=np.int64),
                                np.ascontiguousarray(order_b, dtype=np.int64), np.ascontiguousarray(eid_b, dtype=np.int64),
                                int(n_edges), int(cap), bool(attached), int(max_rounds))
    return _greedy_merge_py(order_a, eid_a, order_b, eid_b, n_edges, cap, attached, max_rounds)


def bfs_orient(indptr, dst, same, m):
    if USE_NUMBA and HAVE_NUMBA:
        return _bfs_orient_nb(np.ascontiguousarray(indptr, dtype=np.int64), np.ascontiguousarray(dst, dtype=np.int64), np.ascontiguousarray(same, dtype=np.bool_), int(m))
    return _bfs_orient_py(indptr, dst, same, m)


# ----------------------------------------------------------------------------------------------- prune / merge (= decode_policy90)
def prune_core(fa, score, cap=2):
    """fa (F, 3) int64 (rows as given), score (F,) float32 -> keep mask (F,) bool; = the greedy of prune_edge_capacity."""
    order = np.argsort(-score, kind='stable')
    eid, ne = _edge_ids(np.sort(fa, axis=1))
    return greedy_cap(order, eid, ne, cap)


def prune_edge_capacity_fast(z_face, faces, metric, cap=2):
    """= decode_policy90.prune_edge_capacity (same face_indicator call, same return)."""
    f = faces if torch.is_tensor(faces) else torch.as_tensor(np.asarray(faces), device=z_face.device)
    if len(f) == 0:
        return f
    t0 = time.perf_counter()
    score = face_indicator(z_face, f.to(z_face.device).long(), metric, 0.0).float().cpu().numpy(); fa = f.cpu().numpy().astype(np.int64)
    t1 = time.perf_counter()
    keep = prune_core(fa, score, cap)
    out = torch.as_tensor(fa[keep], device=z_face.device)
    t2 = time.perf_counter()
    _acc('prune_score_s', t1 - t0); _acc('prune_s', t2 - t1)
    return out


def merge_core(fa, sa, fb, sb, cap=2, attached=False):
    """fa / fb (sorted rows, fb already filtered against fa), scores sa / sb -> kept faces (K, 3) int64 in acceptance order
    (= decode_policy90.merge_faces_by_capacity's `keep` list)."""
    order_a = np.argsort(-sa, kind='stable') if len(fa) else np.zeros(0, np.int64)
    order_b = np.argsort(-sb, kind='stable') if len(fb) else np.zeros(0, np.int64)
    eid, ne = _edge_ids(np.concatenate([fa, fb], 0) if len(fa) or len(fb) else np.zeros((0, 3), np.int64))
    out_a, out_b = greedy_merge(order_a, eid[:len(fa)], order_b, eid[len(fa):], ne, cap, attached, 64 if attached else 1)
    return np.concatenate([fa[out_a], fb[out_b]], 0).reshape(-1, 3)


def merge_faces_by_capacity_fast(z_face, primary, fallback, metric, cap=2, attached=False):
    """= decode_policy90.merge_faces_by_capacity (same face_indicator calls on the same arrays, same return)."""
    t0 = time.perf_counter()
    fa = np.sort((primary.cpu().numpy() if torch.is_tensor(primary) else np.asarray(primary)).astype(np.int64), axis=1)
    fb = np.sort((fallback.cpu().numpy() if torch.is_tensor(fallback) else np.asarray(fallback)).astype(np.int64), axis=1)
    if len(fa) == 0 and len(fb) == 0:
        return torch.zeros((0, 3), dtype=torch.long, device=z_face.device)
    n = max(int(fa.max()) if len(fa) else 0, int(fb.max()) if len(fb) else 0) + 1
    key = lambda t: (t[:, 0] * n + t[:, 1]) * n + t[:, 2]  # noqa: E731
    if len(fa) and len(fb):
        fb = fb[~np.isin(key(fb), key(fa))]
    t1 = time.perf_counter()
    sa = face_indicator(z_face, torch.as_tensor(fa, device=z_face.device), metric, 0.0).float().cpu().numpy() if len(fa) else np.zeros(0, np.float32)
    sb = face_indicator(z_face, torch.as_tensor(fb, device=z_face.device), metric, 0.0).float().cpu().numpy() if len(fb) else np.zeros(0, np.float32)
    t2 = time.perf_counter()
    out = merge_core(fa, sa, fb, sb, cap, attached)
    out_t = torch.as_tensor(np.asarray(out, dtype=np.int64).reshape(-1, 3), device=z_face.device)
    t3 = time.perf_counter()
    _acc('merge_score_s', t2 - t1); _acc('merge_s', (t1 - t0) + (t3 - t2))
    return out_t


# ----------------------------------------------------------------------------------------------- mesh_stats (= mesh_utils90)
def canonical_faces_fast(faces):
    """= mesh_utils90.canonical_faces (unique sorted triples, lexicographic order) through 1-D integer keys."""
    f = np.asarray(faces, dtype=np.int64).reshape(-1, 3)
    if not len(f):
        return f
    f = f[(f[:, 0] != f[:, 1]) & (f[:, 1] != f[:, 2]) & (f[:, 0] != f[:, 2])]
    if not len(f) or f.min() < 0:
        return MU.canonical_faces(f)
    fs = np.sort(f, axis=1); m = int(fs.max()) + 1
    uk = np.unique((fs[:, 0] * m + fs[:, 1]) * m + fs[:, 2])
    return np.stack([uk // (m * m), (uk // m) % m, uk % m], 1)


def mesh_stats_fast(vertices, faces):
    """= mesh_utils90.mesh_stats: the float expressions verbatim; unique edges via 1-D keys; components via scipy on the
    same unique edge list (count of distinct labels over the used vertices = the release's union-find count)."""
    t0 = time.perf_counter()
    v = np.asarray(vertices, dtype=np.float64); f = canonical_faces_fast(faces); n = len(v)
    out = dict(vertices=int(n), faces=int(len(f)))
    if not len(f):
        out.update(edges=0, boundary_edges=0, nonmanifold_edges=0, isolated_vertices=int(n), components=0, euler=int(n), faces_per_vertex=0.0, min_angle_mean_deg=None, sliver_share=None, degree_mean=None, degree_hist={})
        _acc('stats_s', time.perf_counter() - t0)
        return out
    m = max(int(f.max()) + 1, 1)
    ek = np.concatenate([f[:, 0] * m + f[:, 1], f[:, 1] * m + f[:, 2], f[:, 0] * m + f[:, 2]])   # canonical rows: every pair already sorted
    uk, cnt = np.unique(ek, return_counts=True)
    ue = np.stack([uk // m, uk % m], 1)
    used = np.zeros(n, bool); used[f.ravel()] = True
    deg = np.bincount(ue.ravel(), minlength=n)
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import connected_components
    g = csr_matrix((np.ones(len(ue), dtype=np.int8), (ue[:, 0], ue[:, 1])), shape=(n, n))
    _, labels = connected_components(g, directed=False, return_labels=True)
    comps = len(np.unique(labels[used]))
    p = v[f]; a = p[:, 1] - p[:, 0]; b = p[:, 2] - p[:, 0]; c = p[:, 2] - p[:, 1]

    def ang(u, w):
        cu = (u * w).sum(1) / np.maximum(np.linalg.norm(u, axis=1) * np.linalg.norm(w, axis=1), 1e-12)
        return np.degrees(np.arccos(np.clip(cu, -1, 1)))
    angles = np.stack([ang(a, b), ang(-a, c), ang(-b, -c)], 1); mn = angles.min(1)
    area = 0.5 * np.linalg.norm(np.cross(a, b), axis=1)
    el = np.linalg.norm(v[ue[:, 0]] - v[ue[:, 1]], axis=1)
    hist = np.bincount(np.minimum(deg[used], 12), minlength=13)
    out.update(edges=int(len(ue)), boundary_edges=int((cnt == 1).sum()), nonmanifold_edges=int((cnt > 2).sum()), isolated_vertices=int((~used).sum()),
               components=int(comps), euler=int(n - len(ue) + len(f)), faces_per_vertex=float(len(f) / max(n, 1)), used_vertices=int(used.sum()),
               min_angle_mean_deg=float(mn.mean()), sliver_share=float((mn < 15).mean()), degenerate_faces=int((area < 1e-12).sum()),
               degree_mean=float(deg[used].mean()), degree_hist={str(i): int(hist[i]) for i in range(13)}, edge_length_mean=float(el.mean()), edge_length_p95=float(np.percentile(el, 95)),
               area_total=float(area.sum()))
    _acc('stats_s', time.perf_counter() - t0)
    return out


# ----------------------------------------------------------------------------------------------- orientation (= orient_faces.py)
def _load_orient_module():
    spec = importlib.util.spec_from_file_location('orient_faces_meshaxiom_p5', MU.ORIENT_MODULE)
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    return mod


OF = _load_orient_module()
ORIENT_SHA = MU.sha256(MU.ORIENT_MODULE)


def orient_faces_fast(V, F, outward=True):
    """= orient_faces.orient_faces: the same edge table / two-face pairing, the adjacency in the same insertion order (CSR),
    the same BFS (lowest face first, FIFO) -> identical flip / piece / non-orientable sets even for non-orientable pieces; the
    per-piece outward test is the release's numpy expression on the same arrays."""
    V = np.asarray(V, np.float64); F = np.asarray(F, np.int64).copy()
    m = len(F)
    if m == 0:
        return F, dict(pieces=0, nonorientable_pieces=0, flipped_faces=0, outward_flipped_pieces=0)
    lo, hi, fwd, fid = OF._edge_table(F)
    key = lo * (int(hi.max()) + 1) + hi
    order = np.argsort(key, kind='stable'); k = key[order]
    starts = np.r_[0, np.nonzero(np.diff(k))[0] + 1]; counts = np.diff(np.r_[starts, len(k)])
    two = starts[counts == 2]
    e1, e2 = order[two], order[two + 1]
    fa, fb = fid[e1], fid[e2]; same = (fwd[e1] == fwd[e2])
    ok = fa != fb
    fa, fb, same = fa[ok], fb[ok], same[ok]
    P = len(fa)
    src = np.concatenate([fa, fb]); dst = np.concatenate([fb, fa]); s = np.concatenate([same, same])
    seq = np.concatenate([np.arange(P), np.arange(P)])
    o = np.argsort(src * (P + 1) + seq, kind='stable')           # adj[x] in insertion order = pair index ascending
    src, dst, s = src[o], dst[o], s[o]
    indptr = np.searchsorted(src, np.arange(m + 1))
    flip, piece, npieces, bad = bfs_orient(indptr, dst, s, m)
    F[flip == 1] = F[flip == 1][:, [0, 2, 1]]
    out_flips = 0
    if outward:
        T = V[F]; order_p = np.argsort(piece, kind='stable'); ps = piece[order_p]
        bounds = np.r_[0, np.nonzero(np.diff(ps))[0] + 1, len(ps)]
        for i in range(len(bounds) - 1):
            idx = order_p[bounds[i]:bounds[i + 1]]; Ti = T[idx]; c = Ti.reshape(-1, 3).mean(0)
            vol = np.einsum('ij,ij->i', Ti[:, 0] - c, np.cross(Ti[:, 1] - c, Ti[:, 2] - c)).sum()
            if vol < 0:
                F[idx] = F[idx][:, [0, 2, 1]]; out_flips += 1
    info = dict(pieces=int(npieces), nonorientable_pieces=int(bad[:npieces].sum()), flipped_faces=int(np.sum(flip == 1)),
                outward_flipped_pieces=out_flips)
    return F, info


def orient_with_report_fast(V, F, outward=True):
    before = OF.orientation_stats(F)
    G, info = orient_faces_fast(V, F, outward=outward)
    info['before'] = before; info['after'] = OF.orientation_stats(G)
    return G, info


def export_faces_fast(vertices, faces):
    """= mesh_utils90.export_faces (same report dict; module sha computed once)."""
    t0 = time.perf_counter()
    f = np.asarray(faces, dtype=np.int64).reshape(-1, 3)
    if len(f) == 0:
        return f, dict(version='meshaxiom-export-orientation-v1', faces=0, inconsistent_edges_before=0, inconsistent_edges_after=0, pieces=0, nonorientable_pieces=0)
    g, info = orient_with_report_fast(np.asarray(vertices, dtype=np.float64), f)
    assert g.shape == f.shape and np.array_equal(np.sort(g, axis=1), np.sort(f, axis=1)), 'orientation must only permute vertices inside each face'
    rep = dict(version='meshaxiom-export-orientation-v1', module='code/' + MU.ORIENT_MODULE.name, module_sha256=ORIENT_SHA, faces=int(len(f)),
               inconsistent_edges_before=info['before']['inconsistent_edges'], inconsistent_edges_after=info['after']['inconsistent_edges'],
               two_face_edges=info['after']['two_face_edges'], pieces=info['pieces'], nonorientable_pieces=info['nonorientable_pieces'],
               flipped_faces=info['flipped_faces'], outward_flipped_pieces=info['outward_flipped_pieces'], applied_to=['mesh.obj', 'mesh.json'])
    _acc('orient_s', time.perf_counter() - t0)
    return g, rep


# ----------------------------------------------------------------------------------------------- writers (= mesh_utils90)
_D9_VALUES = (np.arange(512, dtype=np.float64) + .5) / 512 * 2 - 1          # the harness's vertex coordinates: (c + 0.5) / 512 * 2 - 1
_D9_STRS = [f'{x:.9g}' for x in _D9_VALUES.tolist()]                          # the same '.9g' formatting, once per cell value


def obj_text(vertices, faces, header):
    """The text mesh_utils90.write_obj writes.  Vertices that are exactly dequantised D9 cells (checked by float equality) take their
    '.9g' strings from the 512-entry table; anything else is formatted per value as the release does."""
    v = np.asarray(vertices, dtype=np.float64); f = np.asarray(faces, dtype=np.int64).reshape(-1, 3)
    c = np.rint((v + 1.0) * 256.0 - 0.5).astype(np.int64)
    if len(v) and v.shape[1] == 3 and ((c >= 0) & (c < 512)).all() and np.array_equal(_D9_VALUES[c], v):
        S = _D9_STRS
        vl = ''.join([f'v {S[a]} {S[b]} {S[c_]}\n' for a, b, c_ in c.tolist()])
    else:
        vl = ''.join([f'v {x:.9g} {y:.9g} {z:.9g}\n' for x, y, z in v.tolist()])
    return ''.join([f'# {header}\n', vl, ''.join([f'f {a} {b} {c_}\n' for a, b, c_ in (f + 1).tolist()])])


def write_obj_fast(path, vertices, faces, header):
    """= mesh_utils90.write_obj byte for byte (the same text, built in memory, one write)."""
    t0 = time.perf_counter()
    text = obj_text(vertices, faces, header)
    with open(path, 'w') as fh:
        fh.write(text)
    _acc('write_obj_s', time.perf_counter() - t0)


# ----------------------------------------------------------------------------------------------- install / warmup
DECODE_SWAPS = (('spacetime.candidate_triangles', ST, 'candidate_triangles', candidate_triangles_fast),
                ('decode_policy90.prune_edge_capacity', DP, 'prune_edge_capacity', prune_edge_capacity_fast),
                ('decode_policy90.merge_faces_by_capacity', DP, 'merge_faces_by_capacity', merge_faces_by_capacity_fast))


def stage_swaps():
    import e2e_common as E
    return (('e2e_common.mesh_stats', E, 'mesh_stats', mesh_stats_fast),
            ('e2e_common.export_faces', E, 'export_faces', export_faces_fast),
            ('e2e_common.write_obj', E, 'write_obj', write_obj_fast))


@contextlib.contextmanager
def patched(decode=True, stage=True):
    """Swap the release / harness module attributes for the fast functions; restore on exit (also on exceptions)."""
    swaps = (list(DECODE_SWAPS) if decode else []) + (list(stage_swaps()) if stage else [])
    old = [(mod, name, getattr(mod, name)) for _, mod, name, _ in swaps]
    for _, mod, name, fn in swaps:
        setattr(mod, name, fn)
    try:
        yield
    finally:
        for mod, name, fn in old:
            setattr(mod, name, fn)


def warmup():
    """Compile / load the numba kernels and touch every path once (untimed; call before the first timed request)."""
    t0 = time.perf_counter()
    e = np.array([[0, 1], [0, 2], [1, 2], [1, 3], [2, 3], [0, 3]], np.int64)
    enumerate_3cycles(e, 5)
    f = np.array([[0, 1, 2], [1, 2, 3], [0, 1, 3], [0, 2, 3]], np.int64)
    sc = np.array([3., 2., 1., 0.5], np.float32)
    prune_core(f, sc, 2)
    merge_core(f[:2], sc[:2], f[2:], sc[2:], 2, True)
    v = np.array([[0., 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 1]])
    mesh_stats_fast(v, f)
    export_faces_fast(v, f)
    obj_text(v, f, 'warmup')
    return dict(seconds=time.perf_counter() - t0, numba=bool(USE_NUMBA and HAVE_NUMBA), numba_cache_dir=os.environ.get('NUMBA_CACHE_DIR'))


def provenance():
    return dict(version=VERSION, module=str(Path(__file__).resolve()), module_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                numba=bool(USE_NUMBA and HAVE_NUMBA), numba_version=(__import__('numba').__version__ if HAVE_NUMBA else None),
                numpy=np.__version__, scipy=__import__('scipy').__version__, torch=torch.__version__,
                swaps=[s[0] for s in DECODE_SWAPS] + ['e2e_common.mesh_stats', 'e2e_common.export_faces', 'e2e_common.write_obj'],
                orient_module_sha256=ORIENT_SHA)
