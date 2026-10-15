"""P4 vertex stage: the shipped coarse-consensus (early-MBR) control flow of release sampler_mbr.sample_mbr over a pluggable
level solver, so that the SAME control flow (candidate seeds, per-level noise, expansion, consensus, continuation) runs
either through the release's eager DiT (reference; CPU-testable) or through the vertex study's accelerated engine
(bench/vertex_engine.Engine 'compile+pool+fine', or distill.student_engine.StudentEngine = teacher at D1-D5 + LoRA student
k=4 at D6-D9).

Control flow (= sampler_mbr.sample_mbr, read 2026-10-14):
  candidate k in 0..K-1 uses the per-level generators vertex_level_stream_seeds(seed + k, 9); candidate 0 = the ordinary sample
  levels 1..select_level: the K candidates as ONE batch (padded to max n), x0[k, :n_k] = randn(1, n_k, 8, gen=gens[k][d-1])[0],
      occ -> _expand per candidate (cap_occupancy -> expand_children -> empty fallback = argmax child)
  consensus_i = mean_j!=i Jaccard(level-select cells of i, of j); k* = argmax; levels select_level+1..9 continue k* with gens[k*]
  K=1: consensus = [1.0], k* = 0 (the ordinary single sample).

Backends
  EagerBackend(net, flow): release arithmetic (net.dit under the caller's autocast, flow.sample with cond / uncond closures,
      lengths mask for the padded coarse batch) — bit-for-bit the sampler_mbr path (CPU test: test_p4_cpu.py).
  FastBackend(engine): vertex_engine.Engine / StudentEngine.  Single-candidate levels: engine.make_vel + engine.dpmpp2m (the engine's
      own path, so K=1 reproduces the vertex study's rollouts).  Coarse K>1 levels: NEW batched path (IMPL_P4_ja.md §1.2): one CUDA
      graph slot with B = 2K rows (cond rows 0..K-1, uncond rows K..2K-1; B = K when cfg == 1), FlashAttention varlen with per-row
      seqused_k = n_k, K/V filled once per level, 20 replays; v = vu + cfg (v - vu) per candidate in bf16 then float (= the
      engine's combine = the release's RectifiedFlow.sample.vel).  Not bit-exact with the shipped padded batch (GEMM tiling).
Rules: new file of the study; vertex_engine / student_engine / release code imported read-only.
"""
from __future__ import annotations

import time

import numpy as np
import torch

from meshaxiom_core.sample import vertex_level_stream_seeds
from meshaxiom_core.models.octree_dit import node_positions, cap_occupancy
from meshaxiom_core.data.mesh_ops import expand_children


def _sync(dev):
    if dev.type == 'cuda':
        torch.cuda.synchronize(dev)


def expand(parents, occ, thr, max_nodes, info, d):
    """= sampler_mbr._expand (occ (n, 8) float on device; parents (n, 3) int64 numpy)."""
    occ, before, capped = cap_occupancy(occ, thr, max_nodes)
    if capped:
        info['cap_events'].append(dict(level=d, predicted_nodes=before, kept_nodes=max_nodes))
    child = expand_children(parents, occ.cpu().numpy(), thr)
    if len(child) == 0:
        info['empty_fallback_events'].append(dict(level=d, parents=int(len(parents))))
        slot = occ.argmax(1, keepdim=True); onehot = torch.zeros_like(occ).scatter_(1, slot, 1.0)
        child = expand_children(parents, onehot.cpu().numpy(), 0.5)
    return child


def consensus_scores(cands_last, K):
    S = [{tuple(x) for x in c.tolist()} for c in cands_last]
    return [float(np.mean([len(S[i] & S[j]) / max(len(S[i] | S[j]), 1) for j in range(K) if j != i])) if K > 1 else 1.0 for i in range(K)]


# ----------------------------------------------------------------------------------------------- backends
class EagerBackend:
    """Release arithmetic through net.dit + flow.sample (the sampler_mbr closures)."""
    name = 'eager_release'
    multi_for_k1 = True        # sampler_mbr runs levels 1..select_level through the padded (lengths-masked) batch even for K=1

    def __init__(self, net, flow):
        self.net, self.flow = net, flow
        self.forwards = 0

    def begin(self):
        pass

    def end(self):
        pass

    def multi(self, ns, pc, pos, lvl, ctx, uc, cfg, d, x0, steps, sampler):
        K = len(ns); dev = ctx.device
        lengths = torch.tensor(ns, device=dev)
        cK, uK = ctx.expand(K, -1, -1), uc.expand(K, -1, -1)
        mf = lambda x, t: self.net.dit(x, t, pos, ctx=cK, lengths=lengths, level=lvl)   # noqa: E731
        uf = lambda x, t: self.net.dit(x, t, pos, ctx=uK, lengths=lengths, level=lvl)   # noqa: E731
        return self.flow.sample(mf, x0, steps=steps, sampler=sampler, cfg_scale=cfg, uncond_fn=uf if cfg != 1.0 else None).float()

    def single(self, N, pos, lvl, ctx, uc, cfg, d, x0, steps, sampler):
        mf = lambda x, t, _p=pos, _l=lvl: self.net.dit(x, t, _p, ctx=ctx, level=_l)   # noqa: E731
        uf = lambda x, t, _p=pos, _l=lvl: self.net.dit(x, t, _p, ctx=uc, level=_l)    # noqa: E731
        return self.flow.sample(mf, x0, steps=steps, sampler=sampler, cfg_scale=cfg, uncond_fn=uf if cfg != 1.0 else None).float()


class FastBackend:
    """vertex_engine.Engine (teacher) or distill.student_engine.StudentEngine; see the module docstring."""

    def __init__(self, engine, force_multi=False):
        import vertex_engine as nf
        self.nf = nf
        self.engine = engine
        self.name = engine.name
        self.multi_for_k1 = bool(force_multi)     # debug: route K=1 coarse levels through the batched slot too (B=2 -> same slot as make_vel)
        self.multi_log = []

    @property
    def forwards(self):
        inner = getattr(self.engine, 'inner', None)
        return int(self.engine.forwards) + (int(inner.forwards) if inner is not None else 0)

    def begin(self):
        self.engine.reset_timers()
        if self.engine.w16:
            self.nf.clear_kv(self.engine.dit)
        inner = getattr(self.engine, 'inner', None)
        if inner is not None and inner.w16:
            self.nf.clear_kv(inner.dit)

    def end(self):
        if self.engine.w16:
            self.nf.clear_kv(self.engine.dit)
        if self.engine.request_slots:
            self.engine.request_slots.clear()

    def _set_depth(self, d):
        if hasattr(self.engine, '_cur_depth'):
            self.engine._cur_depth = int(d)

    def single(self, N, pos, lvl, ctx, uc, cfg, d, x0, steps, sampler):
        assert sampler == 'dpmpp2m'
        vel = self.engine.make_vel(N, pos, lvl, ctx, uc, cfg, d)      # StudentEngine dispatches student depths itself
        return self.engine.dpmpp2m(vel, x0, steps).float()            # x0 (1, N, 8) -> (1, N, 8)

    def multi(self, ns, pc, pos, lvl, ctx, uc, cfg, d, x0, steps, sampler):
        assert sampler == 'dpmpp2m'
        eng = self.engine
        self._set_depth(d)
        assert eng.graph and eng.varlen and eng.kvcache and eng.w16, 'the batched coarse path needs the graph+varlen+kvcache engine'
        K = len(ns); N = max(ns)
        use_cfg = cfg != 1.0
        B = 2 * K if use_cfg else K
        Nb = eng.bucket(N)
        if Nb is None:
            raise RuntimeError(f'{N} coarse tokens exceed the largest bucket')
        slot = eng.slot(B, Nb)                      # captured once per (B, Nb) for the process lifetime (shared pool)
        slot.x.zero_(); slot.pos.zero_()
        slot.pos[:K, :N].copy_(pos)
        if use_cfg:
            slot.pos[K:, :N].copy_(pos)
        ln = torch.tensor(ns, dtype=torch.long, device=pos.device)
        slot.lengths[:K].copy_(ln)
        if use_cfg:
            slot.lengths[K:].copy_(ln)
        if slot.varlen is not None:
            slot.varlen['seqused_k'][:K].copy_(ln.to(torch.int32))
            if use_cfg:
                slot.varlen['seqused_k'][K:].copy_(ln.to(torch.int32))
        slot.level.fill_(int(d))
        slot.ctx.copy_(torch.cat([ctx.expand(K, -1, -1), uc.expand(K, -1, -1)], 0) if use_cfg else ctx.expand(K, -1, -1))
        if eng.kvcache:
            slot.fill_kv(slot.ctx)
        self.multi_log.append(dict(level=int(d), K=K, B=B, N=N, Nb=Nb, ns=list(map(int, ns)), capture_s=slot.capture_s))
        dit_forwards = [0]

        def vel(xt, tv, i=None):
            dit_forwards[0] += 1
            slot.x[:K, :N].copy_(xt)
            if use_cfg:
                slot.x[K:, :N].copy_(xt)
            slot.t.copy_(tv.reshape(1).expand(B))
            out = slot.replay()
            v = out[:K, :N]
            if use_cfg:
                vu = out[K:, :N]
                return (vu + cfg * (v - vu)).float()
            return v.float()

        occ = eng.dpmpp2m(vel, x0, steps)
        eng.forwards += dit_forwards[0]
        return occ.float()


# ----------------------------------------------------------------------------------------------- the sampler
@torch.no_grad()
def sample_mbr(backend, vm, ctx, K=8, select_level=3, seed=2026082301, thr=0.4, cfg=5.0, steps=20, sampler='dpmpp2m', max_nodes=200000,
               timers=None):
    """-> (levels [np.int64 (n_d, 3) for d = 0..9], info dict as sampler_mbr.sample_mbr + per-level seconds in `timers`)."""
    depth_total = int(vm.depth_total)
    cfgs = list(cfg) if isinstance(cfg, (list, tuple)) else [float(cfg)] * depth_total
    assert len(cfgs) == depth_total
    dev = ctx.device
    uc = vm.cond.null_ctx(1).to(ctx.dtype)
    info = dict(cap_events=[], empty_fallback_events=[])
    T = {} if timers is None else timers
    gens = [[torch.Generator(device=dev).manual_seed(s) for s in vertex_level_stream_seeds(seed + k, depth_total)] for k in range(K)]
    cand = [[np.zeros((1, 3), dtype=np.int64)] for _ in range(K)]
    backend.begin()
    try:
        for d in range(1, select_level + 1):
            _sync(dev); t0 = time.perf_counter()
            parents = [c[-1] for c in cand]; n = [len(p) for p in parents]; N = max(n)
            pc = torch.zeros(K, N, 3, dtype=torch.long, device=dev)
            for k, p in enumerate(parents):
                pc[k, :len(p)] = torch.from_numpy(np.ascontiguousarray(p, dtype=np.int64)).to(dev)
            pos = node_positions(pc, depth_total, d - 1)
            lvl = torch.full((K,), d, dtype=torch.long, device=dev)
            x0 = torch.zeros(K, N, 8, device=dev)
            for k in range(K):
                x0[k, :n[k]] = torch.randn(1, n[k], 8, device=dev, generator=gens[k][d - 1])[0]
            c = cfgs[d - 1]
            if K > 1 or getattr(backend, 'multi_for_k1', True):
                occ = backend.multi(n, pc, pos, lvl, ctx, uc, c, d, x0, steps, sampler)
            else:                                   # FastBackend, K=1: the engine's own single path (= the vertex study's rollouts)
                occ = backend.single(N, pos, lvl, ctx, uc, c, d, x0, steps, sampler)
            _sync(dev); t1 = time.perf_counter()
            for k in range(K):
                cand[k].append(expand(parents[k], occ[k, :n[k]], thr, max_nodes, info, d))
            _sync(dev); t2 = time.perf_counter()
            T[f'D{d}'] = dict(parents=int(N), K=K, nodes=[int(len(c_[-1])) for c_ in cand], flow_s=t1 - t0, expand_s=t2 - t1, total_s=t2 - t0)
        t0 = time.perf_counter()
        cons = consensus_scores([c[-1] for c in cand], K)
        kstar = int(np.argmax(cons)); levels = cand[kstar]
        T['consensus_s'] = time.perf_counter() - t0
        for d in range(select_level + 1, depth_total + 1):
            _sync(dev); t0 = time.perf_counter()
            parents = levels[-1]
            pc = torch.from_numpy(np.ascontiguousarray(parents, dtype=np.int64)).to(dev)[None]
            pos = node_positions(pc, depth_total, d - 1)
            lvl = torch.full((1,), d, dtype=torch.long, device=dev)
            x0 = torch.randn(1, len(parents), 8, device=dev, generator=gens[kstar][d - 1])
            c = cfgs[d - 1]
            occ = backend.single(len(parents), pos, lvl, ctx, uc, c, d, x0, steps, sampler)[0]
            _sync(dev); t1 = time.perf_counter()
            levels.append(expand(parents, occ, thr, max_nodes, info, d))
            _sync(dev); t2 = time.perf_counter()
            T[f'D{d}'] = dict(parents=int(len(parents)), K=1, nodes=int(len(levels[-1])), flow_s=t1 - t0, expand_s=t2 - t1, total_s=t2 - t0)
    finally:
        backend.end()
    return levels, dict(info, picked=kstar, consensus=cons, candidate_level_nodes=[len(c[-1]) for c in cand])


def level_groups(T, select_level, student_depths, depth_total=9):
    """Seconds by phase: coarse (levels 1..select_level + consensus), mid (select_level+1 .. first student depth - 1), fine (student depths)."""
    sd = sorted(student_depths or [])
    first_fine = sd[0] if sd else depth_total + 1
    g = dict(coarse=0.0, mid=0.0, fine=0.0)
    for d in range(1, depth_total + 1):
        s = T.get(f'D{d}', {}).get('total_s', 0.0)
        if d <= select_level:
            g['coarse'] += s
        elif d < first_fine:
            g['mid'] += s
        else:
            g['fine'] += s
    g['coarse'] += T.get('consensus_s', 0.0)
    return g
