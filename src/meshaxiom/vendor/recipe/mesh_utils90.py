"""Production mesh statistics and export winding; raw predictions remain untouched."""
import json, hashlib
from pathlib import Path
import numpy as np
ORIENT_MODULE = Path(__file__).resolve().parent / "orient_faces.py"

def sha256(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 24), b''): h.update(b)
    return h.hexdigest()



def canonical_faces(faces):
    f = np.asarray(faces, dtype=np.int64).reshape(-1, 3)
    if not len(f): return f
    f = f[(f[:, 0] != f[:, 1]) & (f[:, 1] != f[:, 2]) & (f[:, 0] != f[:, 2])]
    return np.unique(np.sort(f, axis=1), axis=0)



def mesh_stats(vertices, faces):
    """Topology-quality numbers shared by our meshes and the Tripo meshes (all exact, set-based)."""
    v = np.asarray(vertices, dtype=np.float64); f = canonical_faces(faces); n = len(v)
    out = dict(vertices=int(n), faces=int(len(f)))
    if not len(f):
        out.update(edges=0, boundary_edges=0, nonmanifold_edges=0, isolated_vertices=int(n), components=0, euler=int(n), faces_per_vertex=0.0, min_angle_mean_deg=None, sliver_share=None, degree_mean=None, degree_hist={})
        return out
    e = np.concatenate([f[:, [0, 1]], f[:, [1, 2]], f[:, [0, 2]]]); e = np.sort(e, axis=1)
    ue, cnt = np.unique(e, axis=0, return_counts=True)
    used = np.zeros(n, bool); used[f.ravel()] = True
    deg = np.bincount(ue.ravel(), minlength=n)
    # connected components over used vertices (union-find on unique edges)
    parent = np.arange(n)
    def find(x):
        while parent[x] != x: parent[x] = parent[parent[x]]; x = parent[x]
        return x
    for a, b in ue.tolist():
        ra, rb = find(a), find(b)
        if ra != rb: parent[ra] = rb
    comps = len({find(i) for i in np.flatnonzero(used).tolist()})
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
    return out



def write_obj(path, vertices, faces, header):
    v = np.asarray(vertices, dtype=np.float64); f = np.asarray(faces, dtype=np.int64).reshape(-1, 3)
    with open(path, 'w') as fh:
        fh.write(f'# {header}\n')
        for x, y, z in v: fh.write(f'v {x:.9g} {y:.9g} {z:.9g}\n')
        for a, b, c in f + 1: fh.write(f'f {a} {b} {c}\n')



def write_mesh_json(path, vertices, faces, extra=None):
    d = dict(vertices=np.asarray(vertices, dtype=np.float64).round(7).tolist(), faces=np.asarray(faces, dtype=np.int64).reshape(-1, 3).tolist())
    if extra: d.update(extra)
    Path(path).write_text(json.dumps(d, separators=(',', ':')))



def export_faces(vertices, faces):
    """09-30: faces for the EXPORTED copy only (mesh.obj / mesh.json): one consistent winding per connected piece, pieces
    oriented outward (signed volume; exact for closed pieces, heuristic for open ones).  No vertex moves, no face is added or
    removed, the face order is kept (only the order inside each triple changes), so mesh_stats / face_set_metrics (both
    canonicalise faces) are unchanged.  Keep the raw decoder faces in topology.npz and in every evaluation input.
    The decoder emits index-sorted triples (~70% of two-face edges inconsistent -> holes under back-face culling).
    -> (faces_oriented int64 (F,3), report dict for REPORT/STATS)."""
    import importlib.util, sys as _sys
    spec = importlib.util.spec_from_file_location('orient_faces_meshaxiom', ORIENT_MODULE); mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    f = np.asarray(faces, dtype=np.int64).reshape(-1, 3)
    if len(f) == 0:
        return f, dict(version='meshaxiom-export-orientation-v1', faces=0, inconsistent_edges_before=0, inconsistent_edges_after=0, pieces=0, nonorientable_pieces=0)
    g, info = mod.orient_with_report(np.asarray(vertices, dtype=np.float64), f)
    assert g.shape == f.shape and np.array_equal(np.sort(g, axis=1), np.sort(f, axis=1)), 'orientation must only permute vertices inside each face'
    return g, dict(version='meshaxiom-export-orientation-v1', module=str(ORIENT_MODULE), module_sha256=sha256(ORIENT_MODULE), faces=int(len(f)),
                   inconsistent_edges_before=info['before']['inconsistent_edges'], inconsistent_edges_after=info['after']['inconsistent_edges'],
                   two_face_edges=info['after']['two_face_edges'], pieces=info['pieces'], nonorientable_pieces=info['nonorientable_pieces'],
                   flipped_faces=info['flipped_faces'], outward_flipped_pieces=info['outward_flipped_pieces'], applied_to=['mesh.obj', 'mesh.json'])

