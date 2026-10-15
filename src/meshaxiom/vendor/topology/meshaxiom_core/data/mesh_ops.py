"""Mesh <-> octree / topology conversions.

Everything here is numpy + pure python so it can run inside dataloader workers.

Conventions (follow MeshAxiom Sec. 3.1):
  * vertices are normalised into [-1, 1]^3 and quantised to D bits, giving
    integer coordinates in [0, 2**D - 1]   (paper: D = 9  ->  [0, 511]).
  * an octree level ``d`` holds the *unique* set of occupied node coordinates,
    each in [0, 2**d - 1]^3.  Level 0 is the single root node.
  * a node at depth ``d`` with coordinate ``c`` has parent ``c >> 1`` at depth
    ``d - 1`` and child slot ``idx = 4*(c[0]&1) + 2*(c[1]&1) + (c[2]&1)``.
"""

from __future__ import annotations

import numpy as np

CHILD_WEIGHTS = np.array([4, 2, 1], dtype=np.int64)


# --------------------------------------------------------------------------- #
# normalisation / quantisation
# --------------------------------------------------------------------------- #
def normalize_vertices(v: np.ndarray, scale: float = 0.95) -> np.ndarray:
    """Center on the bbox centre and scale the longest side to ``2*scale``."""
    v = np.asarray(v, dtype=np.float64)
    lo, hi = v.min(0), v.max(0)
    center = (lo + hi) / 2.0
    extent = float((hi - lo).max())
    if extent < 1e-12:
        extent = 1e-12
    return (v - center) * (2.0 * scale / extent)


def quantize(v: np.ndarray, depth: int) -> np.ndarray:
    """[-1, 1] floats -> integer grid coordinates in [0, 2**depth - 1]."""
    res = 1 << depth
    q = np.floor((np.asarray(v) + 1.0) * 0.5 * res)
    return np.clip(q, 0, res - 1).astype(np.int64)


def dequantize(q: np.ndarray, depth: int) -> np.ndarray:
    """Grid coordinates -> cell centres in [-1, 1] (inverse of ``quantize``)."""
    res = 1 << depth
    return (np.asarray(q, dtype=np.float64) + 0.5) / res * 2.0 - 1.0


def coord_key(c: np.ndarray, depth: int) -> np.ndarray:
    """Unique int64 key for a coordinate at ``depth`` (row-major)."""
    res = 1 << depth
    c = np.asarray(c, dtype=np.int64)
    return (c[:, 0] * res + c[:, 1]) * res + c[:, 2]


def sort_by_key(c: np.ndarray, depth: int | None = None) -> np.ndarray:
    """Canonical (lexicographic == key) ordering of node coordinates."""
    c = np.asarray(c, dtype=np.int64)
    if len(c) == 0:
        return c
    return c[np.lexsort((c[:, 2], c[:, 1], c[:, 0]))]


# --------------------------------------------------------------------------- #
# mesh cleanup after quantisation
# --------------------------------------------------------------------------- #
def quantize_mesh(vertices: np.ndarray, faces: np.ndarray, depth: int):
    """Quantise vertices, merge coincident ones, drop degenerate/duplicate faces.

    Returns ``(coords[int64, (N,3)], faces[int64, (F,3)])`` with ``coords``
    sorted by key so the representation is canonical.
    """
    q = quantize(normalize_vertices(vertices), depth)
    keys = coord_key(q, depth)
    uniq_keys, inv = np.unique(keys, return_inverse=True)
    res = 1 << depth
    coords = np.stack(
        [uniq_keys // (res * res), (uniq_keys // res) % res, uniq_keys % res], axis=1
    )  # already sorted ascending by key

    f = inv[np.asarray(faces, dtype=np.int64)]
    # degenerate: two or more corners collapsed onto the same grid cell
    ok = (f[:, 0] != f[:, 1]) & (f[:, 1] != f[:, 2]) & (f[:, 0] != f[:, 2])
    f = f[ok]
    if len(f):
        # duplicate faces (same corner set, any winding) collapse to one
        fs = np.sort(f, axis=1)
        _, keep = np.unique(fs, axis=0, return_index=True)
        f = f[np.sort(keep)]
    return coords, f


def faces_to_edges(faces: np.ndarray, num_vertices: int) -> np.ndarray:
    """Unique undirected edges (i < j) of a triangle soup, shape (E, 2)."""
    if len(faces) == 0:
        return np.zeros((0, 2), dtype=np.int64)
    f = np.asarray(faces, dtype=np.int64)
    e = np.concatenate([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]], axis=0)
    e = np.sort(e, axis=1)
    key = e[:, 0] * num_vertices + e[:, 1]
    _, keep = np.unique(key, return_index=True)
    return e[keep]


# --------------------------------------------------------------------------- #
# octree
# --------------------------------------------------------------------------- #
def build_octree(coords: np.ndarray, depth: int) -> list[np.ndarray]:
    """Occupied node coordinates for every level 0..depth.

    ``levels[depth]`` is the (deduplicated, key-sorted) leaf set, i.e. the
    vertex set; ``levels[0]`` is the root ``[[0,0,0]]``.
    """
    levels: list[np.ndarray] = [None] * (depth + 1)  # type: ignore[list-item]
    cur = np.unique(np.asarray(coords, dtype=np.int64), axis=0)
    levels[depth] = sort_by_key(cur, depth)
    for d in range(depth - 1, -1, -1):
        cur = np.unique(levels[d + 1] >> 1, axis=0)
        levels[d] = sort_by_key(cur, d)
    return levels


def child_occupancy(parents: np.ndarray, children: np.ndarray, depth_parent: int):
    """8-bit child occupancy of every parent node.

    ``parents`` are the occupied nodes at ``depth_parent`` (key-sorted),
    ``children`` the occupied nodes at ``depth_parent + 1``.
    Returns ``occ`` of shape ``(len(parents), 8)`` in {0., 1.}.
    """
    occ = np.zeros((len(parents), 8), dtype=np.float32)
    if len(children) == 0:
        return occ
    pk = coord_key(parents, depth_parent)
    ck = coord_key(children >> 1, depth_parent)
    row = np.searchsorted(pk, ck)
    # every child's parent is occupied by construction; guard anyway
    valid = (row < len(pk)) & (pk[np.clip(row, 0, len(pk) - 1)] == ck)
    slot = ((children & 1) * CHILD_WEIGHTS).sum(1)
    occ[row[valid], slot[valid]] = 1.0
    return occ


def expand_children(parents: np.ndarray, occ: np.ndarray, thresh: float = 0.5) -> np.ndarray:
    """Inverse of :func:`child_occupancy`: occupancy bits -> child coordinates.

    Output is returned in canonical (key-sorted) order so that levels stay
    consistent with :func:`build_octree` throughout coarse-to-fine sampling.
    """
    rows, slots = np.nonzero(occ > thresh)
    if len(rows) == 0:
        return np.zeros((0, 3), dtype=np.int64)
    offs = np.stack([(slots >> 2) & 1, (slots >> 1) & 1, slots & 1], axis=1)
    return sort_by_key((np.asarray(parents, dtype=np.int64)[rows] << 1) + offs)


def octree_training_pairs(levels: list[np.ndarray], depth: int):
    """``[(d, parents_at_d-1, occ_at_d), ...]`` for d = 1..depth."""
    out = []
    for d in range(1, depth + 1):
        out.append((d, levels[d - 1], child_occupancy(levels[d - 1], levels[d], d - 1)))
    return out
