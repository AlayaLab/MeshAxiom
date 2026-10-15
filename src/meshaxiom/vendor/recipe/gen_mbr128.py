"""Production coarse-consensus MBR sampler, detached from internal evaluation paths."""
import numpy as np, torch
from meshaxiom_core.sample import vertex_level_stream_seeds
from meshaxiom_core.models.octree_dit import node_positions, cap_occupancy
from meshaxiom_core.data.mesh_ops import expand_children
SEED = 2026082301

def per_level(spec, n=9):
    vals = []
    for part in str(spec).split(','):
        if 'x' in part: v, k = part.split('x'); vals += [float(v)] * int(k)
        else: vals.append(float(part))
    if len(vals) == 1: vals = vals * n
    assert len(vals) == n, spec
    return vals



def _expand(parents, occ, thr, max_nodes, info, d):
    occ, before, capped = cap_occupancy(occ, thr, max_nodes)
    if capped: info['cap_events'].append(dict(level=d, predicted_nodes=before, kept_nodes=max_nodes))
    child = expand_children(parents, occ.cpu().numpy(), thr)
    if len(child) == 0:
        info['empty_fallback_events'].append(dict(level=d, parents=int(len(parents))))
        slot = occ.argmax(1, keepdim=True); onehot = torch.zeros_like(occ).scatter_(1, slot, 1.0)
        child = expand_children(parents, onehot.cpu().numpy(), 0.5)
    return child



@torch.no_grad()
def sample_mbr(net, ctx, flow, K=8, select_level=3, seed=SEED, thr=0.4, cfg=5.0, steps=20, sampler='dpmpp2m', max_nodes=200000):
    cfgs = list(cfg) if isinstance(cfg, (list, tuple)) else [float(cfg)] * net.depth_total   # 09-23: per-level CFG allowed
    dev = ctx.device; uc = net.cond.null_ctx(1).to(ctx.dtype); info = dict(cap_events=[], empty_fallback_events=[])
    gens = [[torch.Generator(device=dev).manual_seed(s) for s in vertex_level_stream_seeds(seed + k, 9)] for k in range(K)]
    cand = [[np.zeros((1, 3), dtype=np.int64)] for _ in range(K)]
    for d in range(1, select_level + 1):   # K candidates as one padded batch
        parents = [c[-1] for c in cand]; n = [len(p) for p in parents]; N = max(n)
        pc = torch.zeros(K, N, 3, dtype=torch.long, device=dev)
        for k, p in enumerate(parents): pc[k, :len(p)] = torch.from_numpy(p).to(dev)
        pos = node_positions(pc, net.depth_total, d - 1); lengths = torch.tensor(n, device=dev); lvl = torch.full((K,), d, dtype=torch.long, device=dev)
        x0 = torch.zeros(K, N, 8, device=dev)
        for k in range(K): x0[k, :n[k]] = torch.randn(1, n[k], 8, device=dev, generator=gens[k][d - 1])[0]
        cK, uK = ctx.expand(K, -1, -1), uc.expand(K, -1, -1)
        mf = lambda x, t: net.dit(x, t, pos, ctx=cK, lengths=lengths, level=lvl)
        uf = lambda x, t: net.dit(x, t, pos, ctx=uK, lengths=lengths, level=lvl)
        c = cfgs[d - 1]; occ = flow.sample(mf, x0, steps=steps, sampler=sampler, cfg_scale=c, uncond_fn=uf if c != 1.0 else None).float()
        for k in range(K): cand[k].append(_expand(parents[k], occ[k, :n[k]], thr, max_nodes, info, d))
    S = [{tuple(x) for x in c[-1].tolist()} for c in cand]
    cons = [float(np.mean([len(S[i] & S[j]) / max(len(S[i] | S[j]), 1) for j in range(K) if j != i])) if K > 1 else 1.0 for i in range(K)]
    kstar = int(np.argmax(cons)); levels = cand[kstar]
    for d in range(select_level + 1, net.depth_total + 1):   # continue the chosen candidate with its own streams
        parents = levels[-1]; pc = torch.from_numpy(parents).to(dev)[None]
        pos = node_positions(pc, net.depth_total, d - 1); lvl = torch.full((1,), d, dtype=torch.long, device=dev)
        mf = lambda x, t, _p=pos, _l=lvl: net.dit(x, t, _p, ctx=ctx, level=_l); uf = lambda x, t, _p=pos, _l=lvl: net.dit(x, t, _p, ctx=uc, level=_l)
        x0 = torch.randn(1, len(parents), 8, device=dev, generator=gens[kstar][d - 1])
        c = cfgs[d - 1]; occ = flow.sample(mf, x0, steps=steps, sampler=sampler, cfg_scale=c, uncond_fn=uf if c != 1.0 else None)[0].float()
        levels.append(_expand(parents, occ, thr, max_nodes, info, d))
    return levels, dict(info, picked=kstar, consensus=cons, candidate_level_nodes=[len(c[-1]) for c in cand])

