"""Topology autoencoder of MeshAxiom (Sec. 3.2.1, PAPER.txt:426-462, 530-539).

The mesh is turned into a bipartite graph ``G``: vertex nodes (feature = vertex
position) and face nodes (feature = face centroid), connected by
vertex<->incident-face edges in both directions (PAPER.txt:437-447).  The
encoder interleaves ``SAGEConv`` graph blocks -- which carry the connectivity --
with plain transformer blocks over *all* nodes -- which bridge disconnected
components (PAPER.txt:441-446, ablation PAPER.txt:813-820).  A KL bottleneck on
the vertex nodes gives the per-vertex latent ``H_V``; the decoder is pure
attention over vertex nodes and emits two spacetime embeddings, one for the
1st-order (edge) and one for the 2nd-order (face) indicator -- the "feature
split" of PAPER.txt:821-839.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from meshaxiom_core.models.layers import RoPE3D, TransformerBlock
from meshaxiom_core.topology.spacetime import balanced_bce, balanced_bce_terms, candidate_triangles, edge_indicator, face_indicator


# NOTE(deviation): PAPER.txt:446-447 says the input node features are "the vertex
# positions and face centroids" and stops there.  A raw 3-vector makes vertices one grid
# cell apart nearly indistinguishable, so they are octave-Fourier embedded first -- the
# standard choice, and the one SPEC 3 already fixes for the point-cloud encoder.
POS_FREQS = 8


def fourier_pos(p: torch.Tensor, num_freqs: int) -> torch.Tensor:
    """(..., 3) in [-1,1] -> (..., 3 + 6*num_freqs) octave sin/cos features."""
    if num_freqs == 0:
        return p
    f = (2.0 ** torch.arange(num_freqs, device=p.device, dtype=p.dtype)) * torch.pi
    a = p.unsqueeze(-1) * f
    return torch.cat([p, a.sin().flatten(-2), a.cos().flatten(-2)], dim=-1)


class SAGEConv(nn.Module):
    """GraphSAGE mean aggregation: ``h_i' = W_self h_i + W_neigh mean_{j in N(i)} h_j``."""

    def __init__(self, dim_in: int, dim_out: int):
        super().__init__()
        self.lin_self = nn.Linear(dim_in, dim_out)
        self.lin_neigh = nn.Linear(dim_in, dim_out, bias=False)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, num_nodes: int) -> torch.Tensor:
        """``x`` (N, dim_in) flat node features, ``edge_index`` (2, E) src -> dst."""
        src, dst = edge_index[0], edge_index[1]
        agg = torch.zeros_like(x).index_add_(0, dst, x[src])
        deg = x.new_zeros(num_nodes).index_add_(0, dst, torch.ones_like(dst, dtype=x.dtype))
        return self.lin_self(x) + self.lin_neigh(agg / deg.clamp(min=1.0).unsqueeze(-1))


def bipartite_edges(faces: torch.Tensor, f_len: torch.Tensor, num_v: int) -> torch.Tensor:
    """vertex<->incident-face edges of a padded batch, in flat node indices.

    Node ``k`` of sample ``b`` lives at ``b*(Vmax+Fmax) + k``; vertex nodes come
    first, face nodes follow.  Returns ``(2, E)`` with both directions.
    """
    b, fmax, _ = faces.shape
    stride = num_v + fmax
    base = (torch.arange(b, device=faces.device) * stride).view(b, 1, 1)
    v_node = base + faces                                                  # (B, F, 3)
    f_node = (base + num_v + torch.arange(fmax, device=faces.device).view(1, fmax, 1)).expand_as(v_node)
    keep = (torch.arange(fmax, device=faces.device).view(1, fmax, 1) < f_len.view(b, 1, 1)).expand_as(v_node)
    v_node, f_node = v_node[keep], f_node[keep]
    return torch.stack([torch.cat([v_node, f_node]), torch.cat([f_node, v_node])])


class TopologyVAE(nn.Module):
    """KL-regularised autoencoder of Eq. 6 (PAPER.txt:430-436)."""

    def __init__(self, enc_dim: int = 512, enc_layers: int = 24, dec_dim: int = 1024,
                 dec_layers: int = 16, latent_dim: int = 64, heads: int = 8,
                 embed_dim_edge: int = 32, embed_dim_face: int = 32, metric: str = "spacetime",
                 decoder_use_pos: bool = True, enc_use_gcn: bool = True,
                 enc_use_attn: bool = True, split_feature: bool = True,
                 dec_use_mlp: bool = True):
        super().__init__()
        assert enc_layers % 2 == 0, "encoder interleaves graph and attention layers"
        assert enc_use_gcn or enc_use_attn, "encoder needs at least one mixing mechanism"
        self.metric, self.latent_dim = metric, latent_dim
        self.decoder_use_pos = decoder_use_pos
        # Runtime-only switch set by train_vae.  It changes activation storage, not
        # parameters/state_dict, so the pinned pilot remains architecture-compatible.
        self.grad_checkpoint = False
        # Tab. 9 ablation switches.  ``split_feature`` keeps the two heads; without it a
        # single embedding serves both the edge and the face metric, which is the only
        # reading of "w/o split feature" compatible with the rest of the architecture.
        self.enc_use_gcn, self.enc_use_attn = enc_use_gcn, enc_use_attn
        self.split_feature = split_feature
        if not split_feature and embed_dim_edge != embed_dim_face:
            raise ValueError("split_feature=false needs embed_dim_edge == embed_dim_face")
        n_pairs = enc_layers // 2

        self.pos_freqs = POS_FREQS
        self.in_proj = nn.Linear(3 + 6 * self.pos_freqs, enc_dim)
        self.type_embed = nn.Embedding(2, enc_dim)  # 0 = vertex node, 1 = face node
        # Disabled paths are not built at all: parameters that never receive gradient
        # would otherwise break DDP (find_unused_parameters is off everywhere).
        self.graph = (nn.ModuleList(SAGEConv(enc_dim, enc_dim) for _ in range(n_pairs))
                      if enc_use_gcn else [None] * n_pairs)
        self.graph_norm = (nn.ModuleList(nn.LayerNorm(enc_dim, eps=1e-6) for _ in range(n_pairs))
                           if enc_use_gcn else [None] * n_pairs)
        self.enc_attn = (nn.ModuleList(TransformerBlock(enc_dim, heads) for _ in range(n_pairs))
                         if enc_use_attn else [None] * n_pairs)
        self.enc_norm = nn.LayerNorm(enc_dim, eps=1e-6)
        self.to_latent = nn.Linear(enc_dim, 2 * latent_dim)

        self.dec_proj = nn.Linear(latent_dim, dec_dim)
        self.dec = nn.ModuleList(TransformerBlock(dec_dim, heads, use_mlp=dec_use_mlp)
                                 for _ in range(dec_layers))
        self.dec_norm = nn.LayerNorm(dec_dim, eps=1e-6)
        self.head_edge = nn.Linear(dec_dim, embed_dim_edge)
        self.head_face = nn.Linear(dec_dim, embed_dim_face) if split_feature else None
        self.rope = RoPE3D(dec_dim // heads) if decoder_use_pos else None
        # NOTE(deviation): the paper never states the RoPE units for the VAE; SPEC 1
        # fixes finest-grid units, so [-1,1] positions are mapped back to [0, 2^D].
        self.pos_scale = float(1 << 9) / 2.0

        # learnable thresholds only exist for the positive-definite metrics (Tab. 9)
        if metric in ("euclidean", "spherical"):
            init = 1.0 if metric == "euclidean" else 0.0
            self.edge_thresh = nn.Parameter(torch.full((), init))
            self.face_thresh = nn.Parameter(torch.full((), init))
        else:
            self.register_buffer("edge_thresh", torch.zeros(()), persistent=False)
            self.register_buffer("face_thresh", torch.zeros(()), persistent=False)

    def _node_mask(self, v_len: torch.Tensor, f_len: torch.Tensor, num_v: int, num_f: int):
        ar_v = torch.arange(num_v, device=v_len.device).unsqueeze(0)
        ar_f = torch.arange(num_f, device=v_len.device).unsqueeze(0)
        m = torch.cat([ar_v < v_len.unsqueeze(1), ar_f < f_len.unsqueeze(1)], dim=1)
        return m[:, None, None, :]

    def encode(self, pos: torch.Tensor, faces: torch.Tensor, v_len: torch.Tensor,
               f_len: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """``pos`` (B,V,3) in [-1,1], ``faces`` (B,F,3) -> ``mu``, ``logvar`` (B,V,latent)."""
        b, v, _ = pos.shape
        f = faces.shape[1]
        centroid = torch.gather(pos, 1, faces.reshape(b, f * 3, 1).expand(-1, -1, 3))
        centroid = centroid.reshape(b, f, 3, 3).mean(2)                    # (B, F, 3)
        x = self.in_proj(fourier_pos(torch.cat([pos, centroid], dim=1), self.pos_freqs))
        x = x + torch.cat([self.type_embed.weight[0].expand(v, -1),
                           self.type_embed.weight[1].expand(f, -1)]).unsqueeze(0)

        mask = self._node_mask(v_len, f_len, v, f)
        ei = bipartite_edges(faces, f_len, v)
        n = b * (v + f)
        for conv, gnorm, attn in zip(self.graph, self.graph_norm, self.enc_attn):
            def layer(value, conv=conv, gnorm=gnorm, attn=attn):
                if self.enc_use_gcn:
                    h = gnorm(value).reshape(n, -1)
                    value = value + conv(h, ei, n).reshape(b, v + f, -1)
                if self.enc_use_attn:
                    value = attn(value, mask=mask)
                return value

            if self.grad_checkpoint and self.training and torch.is_grad_enabled():
                x = checkpoint(layer, x, use_reentrant=False)
            else:
                x = layer(x)
        mu, logvar = self.to_latent(self.enc_norm(x[:, :v])).chunk(2, dim=-1)
        return mu, logvar.clamp(-30.0, 20.0)

    def decode(self, h: torch.Tensor, pos: torch.Tensor,
               v_len: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """``h`` (B,V,latent) -> ``z_edge``, ``z_face`` (B,V,embed_dim_*)."""
        x = self.dec_proj(h)
        rope = self.rope.cos_sin((pos + 1.0) * self.pos_scale) if self.rope is not None else None
        mask = (torch.arange(pos.shape[1], device=v_len.device).unsqueeze(0) < v_len.unsqueeze(1))
        mask = mask[:, None, None, :]
        for blk in self.dec:
            if self.grad_checkpoint and self.training and torch.is_grad_enabled():
                x = checkpoint(
                    lambda value, block=blk: block(value, rope=rope, mask=mask),
                    x, use_reentrant=False)
            else:
                x = blk(x, rope=rope, mask=mask)
        x = self.dec_norm(x)
        z = self.head_edge(x)
        return z, self.head_face(x) if self.split_feature else z

    def forward(self, pos: torch.Tensor, faces: torch.Tensor, v_len: torch.Tensor,
                f_len: torch.Tensor, sample: bool = True) -> dict:
        mu, logvar = self.encode(pos, faces, v_len, f_len)
        h = mu + torch.randn_like(mu) * (0.5 * logvar).exp() if sample else mu
        z_edge, z_face = self.decode(h, pos, v_len)
        return {"mu": mu, "logvar": logvar, "h": h, "z_edge": z_edge, "z_face": z_face,
                "metric": self.metric, "edge_thresh": self.edge_thresh,
                "face_thresh": self.face_thresh}


# --------------------------------------------------------------------------- #
# loss
# --------------------------------------------------------------------------- #
def _counts(logit: torch.Tensor, label: torch.Tensor) -> tuple[int, int, int]:
    pred, gt = logit > 0, label > 0.5
    return int((pred & gt).sum()), int((pred & ~gt).sum()), int((~pred & gt).sum())


def _prf(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    p, r = tp / max(tp + fp, 1), tp / max(tp + fn, 1)
    return p, r, 2 * p * r / max(p + r, 1e-9)


class TopologyLoss(nn.Module):
    """Balanced BCE on 1st/2nd-order indicators + KL (Eq. 7, PAPER.txt:455-476).

    ``balanced`` selects Eq. 7 (group-wise means over TP/TN/FP/FN) or the plain BCE
    of the paper's own "w/o balance loss" ablation (Tab. 9, PAPER.txt:723, 800-806).
    Eq. 7 re-partitions the pairs by the *current* hard prediction every step, so once
    the FP group is small its per-element weight explodes and, at the batch sizes
    reachable here, training oscillates instead of sharpening -- see the measured
    curves in tests/test_topo_vae.py.  Groups are pooled over the whole batch, which
    is both closer to "all vertex pairs" and 1/B of the variance of per-mesh means.
    """

    def __init__(self, kl_weight: float = 1e-6, face_weight: float = 1.0, edge_weight: float = 1.0,
                 neg_per_pos: int = 64, max_pairs: int = 8_000_000, hard_neg_ratio: float = 0.0,
                 balanced: bool = True, edge_pair_mode: str = 'legacy_sampled',
                 edge_chunk_rows: int = 512, balanced_empty_groups: str = 'skip'):
        super().__init__()
        self.kl_weight, self.face_weight, self.edge_weight = kl_weight, face_weight, edge_weight
        self.neg_per_pos, self.max_pairs, self.hard_neg_ratio = neg_per_pos, max_pairs, hard_neg_ratio
        self.balanced = balanced
        if edge_pair_mode not in ('legacy_sampled', 'all_chunked'):
            raise ValueError('unknown edge pair mode')
        if balanced_empty_groups not in ('skip', 'fixed_four'):
            raise ValueError('unknown empty-group normalization')
        if edge_chunk_rows < 1:
            raise ValueError('edge chunk must be positive')
        self.edge_pair_mode = edge_pair_mode
        self.edge_chunk_rows = edge_chunk_rows
        self.balanced_empty_groups = balanced_empty_groups
        self._cycles: dict[int, torch.Tensor] = {}

    @staticmethod
    def _dist_info(sync: bool) -> tuple[int, bool]:
        import torch.distributed as dist

        if sync and dist.is_available() and dist.is_initialized():
            return dist.get_world_size(), True
        return 1, False

    def _bce(self, logit: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        return balanced_bce(logit, label) if self.balanced else \
            nn.functional.binary_cross_entropy_with_logits(logit, label)

    def _edge_keys(self, faces: torch.Tensor, v: int) -> torch.Tensor:
        e = torch.cat([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]], 0).sort(-1).values
        return torch.unique(e[:, 0] * v + e[:, 1])

    def _edge_terms(self, z: torch.Tensor, v: int, edge_keys: torch.Tensor, metric: str, thresh):
        """Indicator + label over **all** vertex pairs (PAPER.txt:472-473).

        Above ``max_pairs`` the dense (V, V) form is unaffordable, so keep every
        positive and draw a random subset of negatives instead.
        """
        if v * (v - 1) // 2 <= self.max_pairs:  # dense: same code path as extract_edges
            keep = torch.ones(v, v, dtype=torch.bool, device=z.device).triu(1)
            adj = torch.zeros(v, v, dtype=torch.bool, device=z.device)
            adj[edge_keys // v, edge_keys % v] = True
            return edge_indicator(z, None, metric, thresh)[keep], adj[keep].float(), False
        i = torch.randint(0, v, (self.max_pairs,), device=z.device)
        j = torch.randint(0, v, (self.max_pairs,), device=z.device)
        ok = i != j
        i, j = i[ok], j[ok]
        pairs = torch.cat([torch.stack([edge_keys // v, edge_keys % v], 1),
                           torch.stack([torch.minimum(i, j), torch.maximum(i, j)], 1)])
        lbl = torch.isin(pairs[:, 0] * v + pairs[:, 1], edge_keys).float()
        return edge_indicator(z, pairs, metric, thresh), lbl, True

    def _hard_negatives(self, faces: torch.Tensor, edge_keys: torch.Tensor, v: int,
                        pos_keys: torch.Tensor) -> torch.Tensor:
        """3-cycles of the ground-truth edge graph that are not faces."""
        key = hash((v, faces.shape[0], faces.cpu().numpy().tobytes()))
        cached = self._cycles.get(key)
        if cached is None:
            edges = torch.stack([edge_keys // v, edge_keys % v], 1)
            cyc = candidate_triangles(edges, v)
            if len(cyc):
                k = (cyc[:, 0] * v + cyc[:, 1]) * v + cyc[:, 2]
                cyc = cyc[~torch.isin(k, pos_keys)]
            if len(self._cycles) > 64:
                self._cycles.clear()
            self._cycles[key] = cached = cyc
        return cached.to(faces.device)

    def _triplets(self, faces: torch.Tensor, edge_keys: torch.Tensor, v: int):
        pos = faces.sort(-1).values
        pos_keys = (pos[:, 0] * v + pos[:, 1]) * v + pos[:, 2]
        n_neg = self.neg_per_pos * len(pos)
        n_hard = int(self.hard_neg_ratio * n_neg)
        negs = []
        if n_hard:
            pool = self._hard_negatives(faces, edge_keys, v, pos_keys)
            if len(pool):
                negs.append(pool[torch.randint(0, len(pool), (n_hard,), device=faces.device)])
        rnd = torch.randint(0, v, (n_neg - sum(len(x) for x in negs), 3), device=faces.device)
        rnd = rnd.sort(-1).values
        rnd = rnd[(rnd[:, 0] != rnd[:, 1]) & (rnd[:, 1] != rnd[:, 2])]
        negs.append(rnd)
        neg = torch.cat(negs)
        k = (neg[:, 0] * v + neg[:, 1]) * v + neg[:, 2]
        neg = neg[~torch.isin(k, pos_keys)]  # a uniform triplet can hit a real face
        tri = torch.cat([pos, neg])
        lbl = torch.cat([torch.ones(len(pos), device=tri.device), torch.zeros(len(neg), device=tri.device)])
        return tri, lbl

    def forward(self, out: dict, faces: torch.Tensor, v_len: torch.Tensor, f_len: torch.Tensor,
                sync: bool = True, return_terms: bool = False):
        """Eq. 7 with the TP/TN/FP/FN groups pooled over the **global** batch.

        Under DDP each rank holds different meshes, and C(V,2) varies by up to
        ~60x across ranks, so rank-local group means averaged by DDP would make
        the gradient depend on how samples were sharded (and can even flip its
        sign).  Counts are all-reduced (grad-free) and the per-rank sums scaled
        by ``world`` so the DDP-averaged gradient equals the global pooled
        objective exactly.  ``sync=False`` computes a rank-local estimate — use
        it for any call that not every rank makes (e.g. rank-0-only logging),
        otherwise the collective deadlocks.
        """
        metric = out["metric"]
        et, ft = out["edge_thresh"], out["face_thresh"]
        z_edge, z_face = out["z_edge"].float(), out["z_face"].float()
        b = z_edge.shape[0]
        se = sf = None
        ce = cf = None
        edge_confusion = torch.zeros(3, dtype=torch.long, device=z_edge.device)
        face_confusion = torch.zeros(3, dtype=torch.long, device=z_edge.device)
        subsampled = False
        for i in range(b):
            v, f = int(v_len[i]), int(f_len[i])
            fi = faces[i, :f]
            ek = self._edge_keys(fi, v)
            if self.edge_pair_mode == 'all_chunked':
                from meshaxiom_core.topology.streamed_edges import streamed_edge_terms
                if metric != 'spacetime' or not self.balanced:
                    raise ValueError('all_chunked currently requires balanced spacetime loss')
                sei, cei = streamed_edge_terms(z_edge[i, :v], ek,
                                               self.edge_chunk_rows, et)
                edge_confusion += cei[:3]  # group order: TP, FP, FN, TN
            else:
                log_e, lbl_e, sub = self._edge_terms(z_edge[i, :v], v, ek, metric, et)
                subsampled |= sub
            tri, lbl_f = self._triplets(fi, ek, v)
            log_f = face_indicator(z_face[i, :v], tri, metric, ft)
            if self.balanced:
                if self.edge_pair_mode != 'all_chunked':
                    sei, cei = balanced_bce_terms(log_e, lbl_e)
                sfi, cfi = balanced_bce_terms(log_f, lbl_f)
            else:  # plain BCE over all pairs/triplets ("w/o balance loss" of Tab. 9)
                bce = nn.functional.binary_cross_entropy_with_logits
                sei = torch.stack([bce(log_e, lbl_e, reduction="sum")] +
                                  [log_e.new_zeros(())] * 3)
                cei = torch.stack([torch.as_tensor(log_e.numel(), device=log_e.device)] +
                                  [torch.zeros((), dtype=torch.long,
                                               device=log_e.device)] * 3)
                sfi = torch.stack([bce(log_f, lbl_f, reduction="sum")] +
                                  [log_f.new_zeros(())] * 3)
                cfi = torch.stack([torch.as_tensor(log_f.numel(), device=log_f.device)] +
                                  [torch.zeros((), dtype=torch.long,
                                               device=log_f.device)] * 3)
            se, ce = (sei, cei) if se is None else (se + sei, ce + cei)
            sf, cf = (sfi, cfi) if sf is None else (sf + sfi, cf + cfi)
            with torch.no_grad():
                confusion_terms = [(face_confusion, log_f, lbl_f)]
                if self.edge_pair_mode != 'all_chunked':
                    confusion_terms.append((edge_confusion, log_e, lbl_e))
                for target, logits, labels in confusion_terms:
                    pred, truth = logits > 0, labels > 0.5
                    target += torch.stack(((pred & truth).sum(),
                                           (pred & ~truth).sum(),
                                           (~pred & truth).sum()))
        if se is None or sf is None or ce is None or cf is None:
            raise ValueError("topology loss received an empty batch")

        mu, logvar = out["mu"].float(), out["logvar"].float()
        m = (torch.arange(mu.shape[1], device=mu.device).unsqueeze(0) < v_len.unsqueeze(1)).float()
        kl_sum = (-0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum(-1) * m).sum()

        # One compact collective supplies both the global objective denominators and
        # genuinely global logging metrics.  The previous implementation concatenated
        # every mesh's millions of logits and reported rank-0-only F1/loss.
        reduced = torch.cat([
            se.detach().double(), sf.detach().double(), kl_sum.detach().reshape(1).double(),
            ce.double(), cf.double(), m.sum().reshape(1).double(),
            edge_confusion.double(), face_confusion.double(),
            torch.as_tensor([float(subsampled)], dtype=torch.float64, device=z_edge.device),
        ])
        if return_terms:
            return se, sf, kl_sum, reduced
        return self.finish_terms((se, sf, kl_sum, reduced), sync=sync)

    def finish_terms(self, terms, sync=True, global_reduced=None, gradient_world=None):
        se, sf, kl_sum, reduced = terms
        world, ddp = self._dist_info(sync)
        if global_reduced is not None:
            reduced=global_reduced
        elif ddp:
            import torch.distributed as dist

            dist.all_reduce(reduced)
        if gradient_world is not None:world=gradient_world
        gse, gsf, gkl = reduced[:4], reduced[4:8], reduced[8]
        gce, gcf, gcm = reduced[9:13], reduced[13:17], reduced[17].clamp(min=1.0)
        global_edge_confusion = reduced[18:21]
        global_face_confusion = reduced[21:24]
        any_subsampled = reduced[24] > 0

        def pooled_gradient(sums, gcounts):
            ok = gcounts > 0
            if not ok.any():
                return sums.sum() * 0.0
            denom = 4 if self.balanced and self.balanced_empty_groups == 'fixed_four' else int(ok.sum())
            return (sums[ok] / gcounts[ok].to(sums.dtype)).sum() / denom * world

        def pooled_value(sums, counts):
            ok = counts > 0
            denom = 4 if self.balanced and self.balanced_empty_groups == 'fixed_four' else int(ok.sum())
            return ((sums[ok] / counts[ok]).sum() / denom
                    if ok.any() else sums.sum() * 0.0)

        le = pooled_gradient(se, gce)
        lf = pooled_gradient(sf, gcf)
        kl = kl_sum * world / gcm.to(kl_sum.dtype)
        loss = self.edge_weight * le + self.face_weight * lf + self.kl_weight * kl

        with torch.no_grad():
            ep, er, ef1 = _prf(*(int(x) for x in global_edge_confusion.tolist()))
            fp_, fr, ff1 = _prf(*(int(x) for x in global_face_confusion.tolist()))
            le_value, lf_value = pooled_value(gse, gce), pooled_value(gsf, gcf)
            kl_value = gkl / gcm
            loss_value = (self.edge_weight * le_value + self.face_weight * lf_value +
                          self.kl_weight * kl_value)
            edge_candidates = gce.sum().clamp(min=1)
            face_candidates = gcf.sum().clamp(min=1)
            edge_positive = global_edge_confusion[0] + global_edge_confusion[2]
            face_positive = global_face_confusion[0] + global_face_confusion[2]
            edge_predicted_positive = global_edge_confusion[0] + global_edge_confusion[1]
            face_predicted_positive = global_face_confusion[0] + global_face_confusion[1]
        metrics = {"loss": float(loss_value), "loss_edge": float(le_value),
                   "loss_face": float(lf_value), "kl": float(kl_value),
                   "edge_precision": ep, "edge_recall": er, "edge_f1": ef1,
                   "face_precision": fp_, "face_recall": fr, "face_f1": ff1,
                   "edge_candidates": float(edge_candidates),
                   "face_candidates": float(face_candidates),
                   "edge_positive_rate": float(edge_positive / edge_candidates),
                   "face_positive_rate": float(face_positive / face_candidates),
                   "edge_predicted_positive_rate": float(
                       edge_predicted_positive / edge_candidates),
                   "face_predicted_positive_rate": float(
                       face_predicted_positive / face_candidates),
                   "latent_tokens": float(gcm),
                   "pairs_subsampled": float(any_subsampled)}
        return loss, metrics


def build_vae(cfg) -> TopologyVAE:
    m = cfg["model"]
    return TopologyVAE(**{k: m[k] for k in (
        "enc_dim", "enc_layers", "dec_dim", "dec_layers", "latent_dim", "heads",
        "embed_dim_edge", "embed_dim_face", "metric", "decoder_use_pos",
        "enc_use_gcn", "enc_use_attn", "split_feature", "dec_use_mlp") if k in m})


def build_loss(cfg) -> TopologyLoss:
    l = cfg["loss"]
    return TopologyLoss(**{k: l[k] for k in (
        "kl_weight", "face_weight", "edge_weight", "neg_per_pos", "max_pairs",
        "hard_neg_ratio", "balanced", "edge_pair_mode", "edge_chunk_rows",
        "balanced_empty_groups") if k in l})


def dequantize_pos(coords: torch.Tensor, depth: int) -> torch.Tensor:
    """Quantised coords -> float positions in [-1,1] (torch mirror of ``mesh_ops.dequantize``)."""
    return (coords.float() + 0.5) / float(1 << depth) * 2.0 - 1.0
