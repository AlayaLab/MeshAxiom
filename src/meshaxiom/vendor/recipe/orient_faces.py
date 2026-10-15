"""Consistent face orientation for exported meshes (no vertex is moved, no face is added or removed).

The topology decoder outputs faces as vertex triples in index order, i.e. without an orientation:
~70% of the edges shared by two faces join faces of inconsistent winding, so renderers with back-face
culling (game engines' default) show holes. orient_faces() fixes this at export time:
  1. per connected piece (faces joined by edges shared by exactly two faces), propagate one winding
     breadth-first from the lowest face index;
  2. flip each piece whose signed volume about its own centroid is negative (outward heuristic; exact
     for closed pieces, a heuristic for open ones).
Pieces that are non-orientable (Moebius-like) keep some inconsistent edges; they are reported, not hidden.
Edges with more than two faces (none in our decoder output) are ignored for propagation.

Apply ONLY to the final exported copy (mesh.obj / mesh.json / GLB for display, download, games).
Do not rewrite internal prediction files used by evaluation unless the evaluation canonicalizes faces
(sorted triples); the orientation changes the vertex order inside faces, not the face set."""
from collections import deque
import numpy as np


def _edge_table(F):
    u = np.stack([F[:, 0], F[:, 1], F[:, 2]], 1).reshape(-1)
    v = np.stack([F[:, 1], F[:, 2], F[:, 0]], 1).reshape(-1)
    fid = np.repeat(np.arange(len(F)), 3)
    lo, hi = np.minimum(u, v), np.maximum(u, v)
    fwd = u < v                                          # direction of u -> v relative to the sorted key
    return lo, hi, fwd, fid


def orientation_stats(F):
    """conflicts = edges shared by exactly two faces that both traverse the edge in the same direction."""
    F = np.asarray(F, np.int64)
    lo, hi, fwd, fid = _edge_table(F)
    key = lo * (int(hi.max()) + 1 if len(hi) else 1) + hi
    order = np.argsort(key, kind='stable'); k = key[order]
    starts = np.r_[0, np.nonzero(np.diff(k))[0] + 1]; counts = np.diff(np.r_[starts, len(k)])
    two = starts[counts == 2]
    f1, f2 = order[two], order[two + 1]
    conflicts = int(np.sum(fwd[f1] == fwd[f2]))
    return dict(faces=int(len(F)), two_face_edges=int(len(two)), nonmanifold_edges=int(np.sum(counts > 2)),
                inconsistent_edges=conflicts)


def orient_faces(V, F, outward=True):
    """Return (F_oriented, info). F_oriented has the same faces (as vertex sets) in the same order."""
    V = np.asarray(V, np.float64); F = np.asarray(F, np.int64).copy()
    m = len(F)
    if m == 0:
        return F, dict(pieces=0, nonorientable_pieces=0, flipped_faces=0, outward_flipped_pieces=0)
    lo, hi, fwd, fid = _edge_table(F)
    key = lo * (int(hi.max()) + 1) + hi
    order = np.argsort(key, kind='stable'); k = key[order]
    starts = np.r_[0, np.nonzero(np.diff(k))[0] + 1]; counts = np.diff(np.r_[starts, len(k)])
    two = starts[counts == 2]
    e1, e2 = order[two], order[two + 1]
    fa, fb = fid[e1], fid[e2]; same = (fwd[e1] == fwd[e2])      # same direction -> currently inconsistent
    adj = [[] for _ in range(m)]
    for x, y, s in zip(fa.tolist(), fb.tolist(), same.tolist()):
        if x == y: continue
        adj[x].append((y, s)); adj[y].append((x, s))
    flip = np.full(m, -1, np.int8); piece = np.full(m, -1, np.int64); npieces = 0; bad = set()
    for s0 in range(m):
        if flip[s0] >= 0: continue
        flip[s0] = 0; piece[s0] = npieces; q = deque([s0])
        while q:
            f = q.popleft()
            for g, s in adj[f]:
                want = flip[f] ^ (1 if s else 0)
                if flip[g] < 0:
                    flip[g] = want; piece[g] = npieces; q.append(g)
                elif flip[g] != want:
                    bad.add(npieces)
        npieces += 1
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
    info = dict(pieces=int(npieces), nonorientable_pieces=len(bad), flipped_faces=int(np.sum(flip == 1)),
                outward_flipped_pieces=out_flips)
    return F, info


def orient_with_report(V, F, outward=True):
    """orient_faces() plus inconsistent-edge counts before and after (info['before'], info['after'])."""
    before = orientation_stats(F)
    G, info = orient_faces(V, F, outward=outward)
    info['before'] = before; info['after'] = orientation_stats(G)
    return G, info
