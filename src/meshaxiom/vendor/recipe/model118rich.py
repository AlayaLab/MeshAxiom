"""V118 rich-kNN variant of model113 (same API: KNN_K, F_KNN, knn_features, TopologyModel113, upgrade,
marker); used by train_topo118rich.py and decode_policy90 (marker version meshaxiom-topology-richknn-v118).

Why (09-22 night): the v115 plateau is flat on every metric (flip-free F1 0.886 at 5K and at 22.5K); the ambiguity probe shows
the DiT's error is mostly systematic given its inputs, and the v113 kNN features (+0.01, the only input-side change that ever
helped) gave the model just the log-distances of 8 neighbours and the offsets of the 2 nearest.  A quad-dominant remesh vertex
has 4 edge neighbours + diagonals, so the connectivity decision needs the full local stencil.  This variant feeds, per vertex:
  - log1p distances to the 8 nearest vertices                                   (8, unchanged)
  - offsets to ALL 8 nearest vertices (grid cells / 8)                           (24; the first 6 = the v113 two-nearest offsets)
  - log1p count of vertices within 4 cells                                       (1, unchanged)
  - local PCA of the 8-neighbourhood: n n^T of the smallest-eigenvalue direction (6, sign-free normal)
    and the two eigenvalue ratios l1/l3, l2/l3                                    (2)
F_KNN = 41.  upgrade() maps the parent's in_proj columns onto the new layout (x, sc, logd, first 6 offsets, count) and zero-inits
the rest, so step 0 reproduces a v113-family parent exactly."""
import torch, torch.nn as nn
from meshaxiom_core.models.topo_dit import TopologyModel

KNN_K = 8; F_KNN = KNN_K + 3 * KNN_K + 1 + 6 + 2   # 41
F_KNN_V113 = KNN_K + 6 + 1                          # 15 (parent layout)
VERSION = 'meshaxiom-topology-richknn-v118'


@torch.no_grad()
def knn_features(coords, lengths, k=KNN_K, chunk=4096):
    """coords (B,V,3) grid coords, lengths (B,) -> (B,V,F_KNN) float32, zeros on padding."""
    B, V, _ = coords.shape; dev = coords.device; out = torch.zeros(B, V, F_KNN, device=dev, dtype=torch.float32)
    for b in range(B):
        n = int(lengths[b]) if lengths is not None else V
        if n <= 1: continue
        p = coords[b, :n].float(); p = p - p.mean(0, keepdim=True); dists = []; idxs = []
        for s in range(0, n, chunk):
            d = torch.cdist(p[s:s + chunk], p, compute_mode='donot_use_mm_for_euclid_dist')   # exact (same as model113)
            d[torch.arange(d.shape[0], device=dev), torch.arange(s, min(s + chunk, n), device=dev)] = float('inf')
            kk = min(k, n - 1); dk, ik = d.topk(kk, dim=1, largest=False); dists.append(dk); idxs.append(ik)
        dk = torch.cat(dists); ik = torch.cat(idxs); kk = dk.shape[1]
        f = torch.zeros(n, F_KNN, device=dev)
        f[:, :kk] = torch.log1p(dk)
        off = (p[ik] - p[:, None]) / 8.0                                   # (n, kk, 3), nearest first
        f[:, k:k + 3 * kk] = off.reshape(n, -1)
        c0 = k + 3 * k; f[:, c0] = torch.log1p((dk < 4.0).sum(1).float())
        if kk >= 3:
            q = off - off.mean(1, keepdim=True); cov = q.transpose(1, 2) @ q / kk          # (n,3,3)
            ev, evec = torch.linalg.eigh(cov + 1e-6 * torch.eye(3, device=dev))             # ascending
            nrm = evec[:, :, 0]                                                            # smallest-eigenvalue direction
            nn_t = nrm[:, :, None] * nrm[:, None, :]
            iu = torch.triu_indices(3, 3, device=dev); f[:, c0 + 1:c0 + 7] = nn_t[:, iu[0], iu[1]]
            l = ev.clamp(min=1e-6); f[:, c0 + 7] = torch.log(l[:, 0] / l[:, 2]); f[:, c0 + 8] = torch.log(l[:, 1] / l[:, 2])
        out[b, :n] = f
    return out


class TopologyModel113(TopologyModel):
    """Same modules as TopologyModel plus the widened in_proj; forward takes `sc` (B,V,latent) or None."""
    v113 = True

    def forward(self, x, t, coords, lengths=None, ctx=None, sc=None, points=None, normals=None, image=None, camera=None, lod=None, drop=None):
        if ctx is None: ctx = self.cond(points=points, normals=normals, image=image, camera=camera, lod=lod, drop=drop)
        B, V, D = x.shape
        sc = torch.zeros_like(x) if sc is None else sc.to(x.dtype)
        feats = knn_features(coords, lengths).to(x.dtype) if self.v113_knn else torch.zeros(B, V, F_KNN, device=x.device, dtype=x.dtype)
        x_ext = torch.cat([x, sc, feats], dim=-1)
        return self.dit.net(x_ext, t, coords.float() + 0.5, ctx=ctx, lengths=lengths)


def _old_to_new_feature_columns():
    """parent (v113) feature column j -> rich column: logd 0..7 -> 0..7, 2-nearest offsets 8..13 -> 8..13, count 14 -> 32."""
    m = {j: j for j in range(KNN_K + 6)}; m[KNN_K + 6] = KNN_K + 3 * KNN_K; return m


def upgrade(model: TopologyModel, knn: bool = True) -> TopologyModel113:
    """Widen in_proj to lat + lat + F_KNN.  Accepts a plain TopologyModel (d_in == lat), a v113 model (lat + lat + 15) or an
    already-rich one (idempotent).  New columns start at zero."""
    net = model.dit.net; old = net.in_proj; d_in = old.in_features; lat = model.latent_dim; new_in = lat + lat + F_KNN
    if d_in != new_in:
        new = nn.Linear(new_in, old.out_features, bias=old.bias is not None).to(old.weight.device, old.weight.dtype)
        with torch.no_grad():
            new.weight.zero_()
            if d_in == lat: new.weight[:, :lat].copy_(old.weight)
            elif d_in == lat + lat + F_KNN_V113:
                new.weight[:, :2 * lat].copy_(old.weight[:, :2 * lat])
                for j, jn in _old_to_new_feature_columns().items(): new.weight[:, 2 * lat + jn].copy_(old.weight[:, 2 * lat + j])
            else: raise AssertionError(f'unexpected in_proj width {d_in} (lat {lat})')
            if old.bias is not None: new.bias.copy_(old.bias)
        net.in_proj = new
    model.__class__ = TopologyModel113; model.v113_knn = knn; return model


def marker(knn=True): return dict(version=VERSION, self_conditioning=True, knn_features=knn, F_KNN=F_KNN, KNN_K=KNN_K)
