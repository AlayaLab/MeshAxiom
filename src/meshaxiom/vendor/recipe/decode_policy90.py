"""Shared topology decoding policy for the milestone evaluation (v89) and the website generation (v90).
DECODE_POLICY.json (next to this file) selects: face_thresh (absolute, VAE default 0), edge_thresh, cfg, steps, and an
optional LOD-guided adaptive ladder (raise the face threshold until faces <= adaptive x LOD). Absent file = the original
protocol (cfg 3.0, 20 steps, VAE thresholds), so the running daemons are unaffected until a policy is written."""
import json, os
from pathlib import Path
import numpy as np, torch
POLICY_PATH = Path(__file__).resolve().parent / 'DECODE_POLICY.json'
DEFAULT = dict(version='meshaxiom-topology-decode-policy-v1', cfg=3.0, steps=20, sampler='dpmpp2m', face_thresh=None, edge_thresh=None, adaptive=None, ladder=[0.0, 0.1, 0.2, 0.3, 0.45, 0.6], prune=None, note='original protocol', latent='sample', onestep_cfg=1.0, mean_k=4)


def prune_edge_capacity(z_face, faces, metric, cap=2):
    """Local fix for face explosions: keep at most `cap` faces per edge, preferring faces with the highest 2nd-order
    indicator (greedy). Vertices keep their faces wherever the edge graph is sane; only over-subscribed edges lose faces."""
    from meshaxiom_core.topology.spacetime import face_indicator
    f = faces if torch.is_tensor(faces) else torch.as_tensor(np.asarray(faces), device=z_face.device)
    if len(f) == 0: return f
    score = face_indicator(z_face, f.to(z_face.device).long(), metric, 0.0).float().cpu().numpy(); fa = f.cpu().numpy().astype(np.int64)
    order = np.argsort(-score, kind='stable'); count = {}; keep = np.zeros(len(fa), dtype=bool)
    for i in order:
        a, b, c = sorted(fa[i].tolist()); es = ((a, b), (b, c), (a, c))
        if all(count.get(e, 0) < cap for e in es):
            keep[i] = True
            for e in es: count[e] = count.get(e, 0) + 1
    return torch.as_tensor(fa[keep], device=z_face.device)


def drop_long_faces(coords, faces, k):
    """Remove faces whose longest edge is longer than k x the median edge length over all faces (edge lengths in D9 cells)."""
    f = faces.cpu().numpy() if torch.is_tensor(faces) else np.asarray(faces)
    if len(f) == 0: return faces
    v = np.asarray(coords, dtype=np.float64); e = np.linalg.norm(v[f[:, [1, 2, 0]]] - v[f], axis=2); keep = e.max(1) <= k * np.median(e)
    out = f[keep]; return torch.as_tensor(out, device=faces.device) if torch.is_tensor(faces) else out


def policy():
    if POLICY_PATH.exists():
        p = json.loads(POLICY_PATH.read_text()); assert p.get('version') == DEFAULT['version'], 'unknown decode policy version'; return {**DEFAULT, **p}
    return dict(DEFAULT)


def fill_boundary_by_score(z_face, faces, edges, num_vertices, metric, cap=2, floor=-2.0, passes=2):
    """Score-driven hole filling (09-21). Candidates = all 3-cycles of the (thresholded) edge graph that are NOT yet faces; in
    descending 2nd-order score, add a candidate iff its score >= floor, it touches at least one boundary edge (edge with exactly
    one face) and no edge of it already has `cap` faces. The conditional-mean latent ranks faces well but is under-confident,
    so faces just below 0 that close holes are recovered without opening the threshold everywhere."""
    from meshaxiom_core.topology.spacetime import candidate_triangles, face_indicator
    f = faces if torch.is_tensor(faces) else torch.as_tensor(np.asarray(faces), device=z_face.device)
    cand = candidate_triangles(edges.to(z_face.device).long(), int(num_vertices), max_triangles=20_000_000, return_info=False)
    if len(cand) == 0 or len(f) == 0: return f, 0
    score = torch.cat([face_indicator(z_face, cand[b:b + 4_000_000], metric, 0.0) for b in range(0, len(cand), 4_000_000)]).float()
    fa = np.sort(f.cpu().numpy().astype(np.int64), axis=1); ca = np.sort(cand.cpu().numpy().astype(np.int64), axis=1); sc = score.cpu().numpy()
    n = int(num_vertices); key = lambda t: (t[:, 0] * n + t[:, 1]) * n + t[:, 2]
    have = set(key(fa).tolist()); ck = key(ca); order = np.argsort(-sc, kind='stable')
    ekey = lambda a, b: a * n + b
    count = {}
    for a, b, c in fa.tolist():
        for e in ((a, b), (b, c), (a, c)): count[ekey(*e)] = count.get(ekey(*e), 0) + 1
    added = []
    for _ in range(passes):
        new = 0
        for i in order.tolist():
            if sc[i] < floor: break
            if ck[i] in have: continue
            a, b, c = ca[i].tolist(); es = (ekey(a, b), ekey(b, c), ekey(a, c)); cs = [count.get(e, 0) for e in es]
            if max(cs) >= cap or 1 not in cs: continue
            have.add(ck[i]); added.append((a, b, c)); new += 1
            for e in es: count[e] = count.get(e, 0) + 1
        if new == 0: break
    if not added: return f, 0
    out = np.concatenate([fa, np.asarray(added, dtype=np.int64)]); return torch.as_tensor(out, device=z_face.device), len(added)


def complete_boundary_loops(z_edge, z_face, faces, metric, edge_floor=-2.0, face_floor=-4.0, cap=2):
    """Close boundary loops of length 3 and 4 (09-21). Under the conditional-mean latent an ambiguous quad often has BOTH
    diagonals below the edge threshold, so no candidate triangle exists and the quad stays open (4 boundary edges). For every
    boundary 4-loop a-b-c-d pick the diagonal with the higher 1st-order score (>= edge_floor) and add its two triangles when
    both 2nd-order scores are >= face_floor; boundary 3-loops get their triangle under the same face floor. Capacity `cap` per edge."""
    from meshaxiom_core.topology.spacetime import edge_indicator, face_indicator
    f = faces if torch.is_tensor(faces) else torch.as_tensor(np.asarray(faces), device=z_face.device)
    if len(f) == 0: return f, 0
    fa = np.sort(f.cpu().numpy().astype(np.int64), axis=1); count = {}
    for a, b, c in fa.tolist():
        for e in ((a, b), (b, c), (a, c)): count[e] = count.get(e, 0) + 1
    bnd = [e for e, n in count.items() if n == 1]
    if not bnd: return f, 0
    adj = {}
    for a, b in bnd: adj.setdefault(a, set()).add(b); adj.setdefault(b, set()).add(a)
    loops3, loops4 = set(), set()
    for a, b in bnd:
        for c in adj[b]:
            if c == a: continue
            if a in adj[c]: loops3.add(tuple(sorted((a, b, c)))); continue
            for d in adj[c]:
                if d in (a, b): continue
                if d in adj[a]: loops4.add((a, b, c, d) if (a, b, c, d) < (a, d, c, b) else (a, d, c, b))
    # dedupe 4-loops by vertex set (a 4-loop has 4 rotations x 2 directions; keep one)
    seen = {}; 
    for L in loops4:
        key = frozenset(L)
        if key not in seen: seen[key] = L
    loops4 = list(seen.values()); dev = z_face.device
    added = []
    def can_add(tri):
        a, b, c = sorted(tri); es = ((a, b), (b, c), (a, c)); return all(count.get(e, 0) < cap for e in es)
    def commit(tri):
        a, b, c = sorted(tri)
        for e in ((a, b), (b, c), (a, c)): count[e] = count.get(e, 0) + 1
        added.append((a, b, c))
    if loops3:
        t3 = torch.tensor(sorted(loops3), device=dev); sc3 = face_indicator(z_face, t3, metric, 0.0).float().cpu().numpy()
        for tri, sc in zip(t3.tolist(), sc3):
            if sc >= face_floor and can_add(tri): commit(tri)
    if loops4:
        diag = torch.tensor([[min(a, c), max(a, c)] for a, b, c, d in loops4] + [[min(b, d), max(b, d)] for a, b, c, d in loops4], device=dev)
        es = edge_indicator(z_edge, diag, metric, 0.0).float().cpu().numpy(); n4 = len(loops4)
        tris = []
        for i, (a, b, c, d) in enumerate(loops4): tris += [(a, b, c), (a, c, d), (a, b, d), (b, c, d)]
        fs = face_indicator(z_face, torch.tensor([sorted(t) for t in tris], device=dev), metric, 0.0).float().cpu().numpy().reshape(n4, 4)
        for i, (a, b, c, d) in enumerate(loops4):
            if count.get((min(a, c), max(a, c)), 0) or count.get((min(b, d), max(b, d)), 0): continue   # a diagonal already carries faces: the loop is the rim of a filled patch, not a hole
            s_ac, s_bd = es[i], es[n4 + i]
            options = [(s_ac, ((a, b, c), (a, c, d)), fs[i, 0:2]), (s_bd, ((a, b, d), (b, c, d)), fs[i, 2:4])]
            options.sort(key=lambda o: -o[0])
            for s_e, pair, s_f in options:
                if s_e < edge_floor or s_f.min() < face_floor: continue
                if can_add(pair[0]) and can_add(pair[1]): commit(pair[0]); commit(pair[1]); break
    if not added: return f, 0
    return torch.as_tensor(np.concatenate([fa, np.asarray(added, dtype=np.int64)]), device=dev), len(added)


def merge_faces_by_capacity(z_face, primary, fallback, metric, cap=2, attached=False):
    """Primary faces first (in descending 2nd-order score under z_face), then fallback faces that are not present yet, in the same
    order; a face is kept only while every edge has fewer than `cap` faces. Union of two coherent tessellations without overlaps."""
    from meshaxiom_core.topology.spacetime import face_indicator
    fa = np.sort((primary.cpu().numpy() if torch.is_tensor(primary) else np.asarray(primary)).astype(np.int64), axis=1)
    fb = np.sort((fallback.cpu().numpy() if torch.is_tensor(fallback) else np.asarray(fallback)).astype(np.int64), axis=1)
    if len(fa) == 0 and len(fb) == 0: return torch.zeros((0, 3), dtype=torch.long, device=z_face.device)
    n = max(int(fa.max()) if len(fa) else 0, int(fb.max()) if len(fb) else 0) + 1; key = lambda t: (t[:, 0] * n + t[:, 1]) * n + t[:, 2]
    seen = set(key(fa).tolist()) if len(fa) else set(); fb = fb[[k not in seen for k in key(fb).tolist()]] if len(fb) else fb
    count = {}; keep = []
    for k, arr in enumerate((fa, fb)):
        if not len(arr): continue
        sc = face_indicator(z_face, torch.as_tensor(arr, device=z_face.device), metric, 0.0).float().cpu().numpy(); order = np.argsort(-sc, kind='stable').tolist()
        # `attached`: fallback faces are accepted only if they share an edge with a face already kept (grow from the rim, never
        # float islands into holes); iterate to a fixed point since acceptance opens new rims.
        for _ in range(64 if (attached and k == 1) else 1):
            rest = []; added = 0
            for i in order:
                a, b, c = arr[i].tolist(); es = ((a, b), (b, c), (a, c))
                if not all(count.get(e, 0) < cap for e in es): continue
                if attached and k == 1 and not any(count.get(e, 0) >= 1 for e in es): rest.append(i); continue
                keep.append((a, b, c)); added += 1
                for e in es: count[e] = count.get(e, 0) + 1
            order = rest
            if not added or not order: break
    return torch.as_tensor(np.asarray(keep, dtype=np.int64).reshape(-1, 3), device=z_face.device)


def build_topology_model(ck, cfg, use_ema=True):
    """Build the released v119 rich-8 architecture, then load every checkpoint tensor."""
    from meshaxiom_core.models.topo_dit import TopologyModel
    from meshaxiom_core.sample import load_weights
    from meshaxiom.vendor.recipe import model118rich
    marker = ck.get('v113') or {}
    if marker.get('version') != 'meshaxiom-topology-richknn-v118':
        raise ValueError('release supports only v119 rich-knn-v118 checkpoints')
    model = model118rich.upgrade(TopologyModel(cfg), knn=bool(marker.get('knn_features', True)))
    load_weights(model, ck, 'topo', use_ema=use_ema)
    st = ck['latent_stats']; model.dit.set_latent_stats(st['mean'], st['std'])
    return model


def _sc_fns(model, ctx, c, cfg_scale):
    """Model/uncond closures for flow.sample. V113 models get self-conditioning: the conditional call at each ODE step receives the
    previous step's x1_hat (normalised latent) and stores its own; the unconditional call uses the same input as the conditional one."""
    b = c.shape[0]; uc = model.cond.null_ctx(b).to(ctx.dtype) if cfg_scale != 1.0 else None
    if not getattr(model, 'v113', False):
        return (lambda x, t: model.forward(x, t, c, ctx=ctx)), ((lambda x, t: model.forward(x, t, c, ctx=uc)) if uc is not None else None)
    state = {'sc': None, 'used': None}
    def fn(x, t):
        state['used'] = state['sc']; v = model.forward(x, t, c, ctx=ctx, sc=state['sc']); tt = t.reshape(-1, 1, 1).to(x.dtype)
        state['sc'] = (x + (1.0 - tt) * v.to(x.dtype)).detach(); return v
    def unc(x, t): return model.forward(x, t, c, ctx=uc, sc=state['used'])
    return fn, (unc if uc is not None else None)


def sample_latents(model, ctx, c, flow, generator, pol=None):
    """Policy-selected latent estimate (denormalised per-vertex latents, (1, V, d)).
    latent='sample'       : one draw of the stochastic sampler (steps / sampler / cfg)                 -- original protocol
    latent='onestep_mean' : conditional mean.  Rectified flow with the independent coupling gives v(x0, t=0) = E[x1 | cond] - x0,
                            so x0 + v(x0, 0) is E[x1 | cond] from ONE forward pass (cfg from pol['onestep_cfg']; 1.0 = pure conditional).
                            09-21 probe v105/meanlatent: the topology is (nearly) deterministic given the vertices, so a single
                            sample carries sampling noise that flips ~20% of the GT edges below threshold; the mean does not.
    latent='mean_k'       : Monte-Carlo estimate of the same mean from pol['mean_k'] sampler draws (k x the cost of 'sample')."""
    pol = pol or policy(); mode = pol.get('latent', 'sample')
    if mode == 'sample':
        if getattr(model, 'v113', False): return sample_latents(model, ctx, c, flow, generator, {**pol, 'latent': 'sample_trunc', 'noise_scale': 1.0})
        return model.sample_latents(ctx, c, steps=int(pol['steps']), sampler=pol['sampler'], cfg_scale=float(pol['cfg']), flow=flow, generator=generator)
    if mode == 'sample_trunc':
        # low-temperature draw: the ODE starts from noise scaled by pol['noise_scale'] (< 1 pulls every draw toward the mean while
        # keeping one coherent mode per region); same sampler / cfg as 'sample'.
        b, v, _ = c.shape; x0 = torch.randn(b, v, model.latent_dim, device=c.device, generator=generator) * float(pol.get('noise_scale', 1.0))
        s_cfg = float(pol['cfg']); model_fn, unc = _sc_fns(model, ctx, c, s_cfg)
        x1 = flow.sample(model_fn, x0, steps=int(pol['steps']), sampler=pol['sampler'], cfg_scale=s_cfg, uncond_fn=unc)
        return model.dit.denormalize_latent(x1)
    if mode == 'mean_k':
        k = int(pol.get('mean_k', 4)); assert k >= 1
        return torch.stack([model.sample_latents(ctx, c, steps=int(pol['steps']), sampler=pol['sampler'], cfg_scale=float(pol['cfg']), flow=flow, generator=generator).float() for _ in range(k)]).mean(0)
    if mode == 'onestep_mean':
        b, v, _ = c.shape; x0 = torch.randn(b, v, model.latent_dim, device=c.device, generator=generator); t = torch.zeros(b, device=c.device)
        vel = model.forward(x0, t, c, ctx=ctx); s = float(pol.get('onestep_cfg', 1.0))
        if s != 1.0: vu = model.forward(x0, t, c, ctx=model.cond.null_ctx(b).to(ctx.dtype)); vel = vu + s * (vel - vu)
        if getattr(model, 'v113', False):
            for _ in range(int(pol.get('sc_passes', 2)) - 1):   # refine the estimate with itself as self-condition
                sc = (x0 + vel.float()).detach(); vel = model.forward(x0, t, c, ctx=ctx, sc=sc)
                if s != 1.0: vu = model.forward(x0, t, c, ctx=model.cond.null_ctx(b).to(ctx.dtype), sc=sc); vel = vu + s * (vel - vu)
        return model.dit.denormalize_latent(x0 + vel.float())
    if mode == 'onestep_edges_sample_faces':
        # edge graph from the conditional mean (edges are near-deterministic); faces decided on one decisive sampler draw
        h_mean = sample_latents(model, ctx, c, flow, generator, {**pol, 'latent': 'onestep_mean'})
        h_samp = model.sample_latents(ctx, c, steps=int(pol['steps']), sampler=pol['sampler'], cfg_scale=float(pol['cfg']), flow=flow, generator=generator)
        return (h_mean, h_samp)
    if mode == 'onestep_refine':
        # SDEdit-style: anchor at the conditional mean, re-noise to t = refine_t and integrate the remaining path (Heun, cfg)
        h_mean = sample_latents(model, ctx, c, flow, generator, {**pol, 'latent': pol.get('mean_estimator', 'onestep_mean')}); x1 = model.dit.normalize_latent(h_mean.float())
        tau = float(pol.get('refine_t', 0.7)); n = max(2, int(round(int(pol['steps']) * (1.0 - tau)))); b = c.shape[0]
        noise = torch.randn(x1.shape, device=x1.device, generator=generator, dtype=x1.dtype) * float(pol.get('refine_noise_scale', 1.0))   # 09-21: low-temperature warm start
        x = (1.0 - tau) * noise + tau * x1; ts = torch.linspace(tau, 1.0, n + 1, device=c.device); s_cfg = float(pol['cfg']); uc = model.cond.null_ctx(b).to(ctx.dtype) if s_cfg != 1.0 else None
        _fn, _unc = _sc_fns(model, ctx, c, s_cfg)
        def vel(xt, tv):
            tb = torch.full((b,), float(tv), device=c.device); v = _fn(xt, tb)
            if _unc is not None: vu = _unc(xt, tb); v = vu + s_cfg * (v - vu)
            return v.float()
        for i in range(n):
            dt = ts[i + 1] - ts[i]; v1 = vel(x, ts[i])
            if i == n - 1: x = x + dt * v1; break
            v2 = vel(x + dt * v1, ts[i + 1]); x = x + dt * 0.5 * (v1 + v2)
        return model.dit.denormalize_latent(x)
    if mode in ('mean_and_sample', 'mean_and_refine'):
        # two latents: the conditional mean (accurate, under-confident) and one coherent draw (sampler or mean-anchored
        # refinement). decode() merges the two tessellations according to pol['merge'] ('mean_first' | 'draw_first').
        h_mean = sample_latents(model, ctx, c, flow, generator, {**pol, 'latent': pol.get('mean_estimator', 'onestep_mean')})   # 'onestep_mean' (exact, one pass) or 'mean_k' (Monte-Carlo over mean_k draws of `steps` steps)
        draw_pol = {**pol, 'latent': 'sample_trunc', 'noise_scale': float(pol.get('noise_scale', 1.0))}   # primary draw: low-temperature when noise_scale < 1 (09-21 scan: 0.5 -> F1 +0.039, boundary -40% on dev128)
        h_draw = sample_latents(model, ctx, c, flow, generator, {**pol, 'latent': 'onestep_refine'}) if mode == 'mean_and_refine' else sample_latents(model, ctx, c, flow, generator, draw_pol)
        extra_pol = {**pol, 'latent': 'sample_trunc', 'noise_scale': float(pol.get('extra_noise_scale', pol.get('noise_scale', 1.0)))}
        extra = [sample_latents(model, ctx, c, flow, generator, extra_pol) for _ in range(int(pol.get('extra_draws', 0)))]   # further coherent draws, merged last (attached)
        return {'mean': h_mean, 'draw': h_draw, 'extra': extra}
    raise ValueError(f'unknown latent mode {mode!r}')


def decode(model, vae, coords, ctx, lod, flow, device, generator, pol=None):
    """-> (faces, edges, info). Same sampler call as meshaxiom_core.sample.sample_topology; the extraction honours the policy."""
    from meshaxiom_core.models.topo_vae import dequantize_pos
    from meshaxiom_core.topology.spacetime import extract_mesh_topology
    pol = pol or policy()
    c = torch.from_numpy(np.ascontiguousarray(coords)).to(device)[None]
    h = sample_latents(model, ctx, c, flow, generator, pol)
    pos = dequantize_pos(c, 9); v_len = torch.full((1,), len(coords), dtype=torch.long, device=device)
    if isinstance(h, tuple): z_edge = vae.decode(h[0].float(), pos, v_len)[0]; z_face = vae.decode(h[1].float(), pos, v_len)[1]   # (edge latent, face latent) from two estimators
    elif isinstance(h, dict): z_edge, z_face = vae.decode(h['mean'].float(), pos, v_len); z_edge_fb, z_face_fb = vae.decode(h['draw'].float(), pos, v_len)
    else: z_edge, z_face = vae.decode(h.float(), pos, v_len)
    et = vae.edge_thresh if pol['edge_thresh'] is None else float(pol['edge_thresh'])
    ladder = list(pol['ladder']) if pol['adaptive'] else [vae.face_thresh if pol['face_thresh'] is None else float(pol['face_thresh'])]
    used = None
    for ft in ladder:
        # 09-20 sweep v105 (dev128 tune/confirm): keep the production thresholds but cap the edge graph at edge_cap x 1.5 x LoD
        # strongest edges (extract_edges keeps the top-K by indicator); identical to the plain threshold unless the graph explodes.
        max_edges = int(float(pol['edge_cap']) * 1.5 * float(lod)) if pol.get('edge_cap') else 4_000_000
        e, f, xinfo = extract_mesh_topology(z_edge[0].float(), z_face[0].float(), vae.metric, et, float(ft), max_edges=max(max_edges, 1), return_info=True); used = float(ft); n_raw = int(len(f)); cap_hit = bool(xinfo.get('modified_by_cap', False))
        if pol.get('prune') == 'edge_capacity2': f = prune_edge_capacity(z_face[0].float(), f, vae.metric, 2)
        if isinstance(h, dict):   # two tessellations (mean / draw); see sample_latents: mean_and_*
            from meshaxiom_core.topology.spacetime import face_indicator
            e2, f2 = extract_mesh_topology(z_edge_fb[0].float(), z_face_fb[0].float(), vae.metric, et, float(ft), max_edges=max(max_edges, 1), return_info=False)
            if pol.get('prune') == 'edge_capacity2': f2 = prune_edge_capacity(z_face_fb[0].float(), f2, vae.metric, 2)
            zfm = z_face[0].float()
            if pol.get('merge', 'mean_first') == 'mean_first': f = merge_faces_by_capacity(zfm, f, f2, vae.metric, 2)                      # mean faces first, draw fills the rest
            elif pol.get('merge') == 'draw_first_attached':                                                                                # draw first (no veto), mean faces grow from the draw's rims only
                f = merge_faces_by_capacity(zfm, f2, f, vae.metric, 2, attached=True)
                for hx in h.get('extra', []):   # extra draws: their faces grow from the current rims (attached), lowest priority
                    ze_x, zf_x = vae.decode(hx.float(), pos, v_len); ex, fx = extract_mesh_topology(ze_x[0].float(), zf_x[0].float(), vae.metric, et, float(ft), max_edges=max(max_edges, 1), return_info=False)
                    if pol.get('prune') == 'edge_capacity2': fx = prune_edge_capacity(zf_x[0].float(), fx, vae.metric, 2)
                    f = merge_faces_by_capacity(zfm, f, fx, vae.metric, 2, attached=True); e = torch.unique(torch.cat([e.to(z_edge.device), ex.to(z_edge.device)]), dim=0)
            else:                                                                                                                       # draw first, minus faces the mean vetoes, mean fills the rest
                f2 = f2 if torch.is_tensor(f2) else torch.as_tensor(np.asarray(f2), device=zfm.device)
                if len(f2): f2 = f2[face_indicator(zfm, f2.to(zfm.device).long(), vae.metric, 0.0).float() >= float(pol.get('veto', -2.0))]
                f = merge_faces_by_capacity(zfm, f2, f, vae.metric, 2)
            e = torch.unique(torch.cat([e.to(z_edge.device), e2.to(z_edge.device)]), dim=0)
        if pol.get('fill') == 'boundary_score': f, n_fill = fill_boundary_by_score(z_face[0].float(), f, e, len(coords), vae.metric, 2, float(pol.get('fill_floor', -2.0)))
        if pol.get('loops'):   # close boundary 3/4-loops by indicator scores (see complete_boundary_loops)
            f, n_loops = complete_boundary_loops(z_edge[0].float(), z_face[0].float(), f, vae.metric, float(pol.get('loop_edge_floor', -2.0)), float(pol.get('loop_face_floor', -4.0)))
        if pol.get('long_face_k'):   # 09-20 probe v103: drop faces whose longest edge exceeds k x the mesh's median edge length (hair-bridging faces)
            f = drop_long_faces(coords, f, float(pol['long_face_k']))
        if not pol['adaptive'] or len(f) <= float(pol['adaptive']) * float(lod): break
    fa = f.cpu().numpy() if torch.is_tensor(f) else np.asarray(f); ea = e.cpu().numpy() if torch.is_tensor(e) else np.asarray(e)
    return fa, ea, dict(face_thresh_used=used, edge_thresh_used=float(et), modified_by_cap=cap_hit, faces_raw=n_raw, adaptive=pol['adaptive'], prune=pol.get('prune'), long_face_k=pol.get('long_face_k'), edge_cap=pol.get('edge_cap'), cfg=float(pol['cfg']), steps=int(pol['steps']), latent=pol.get('latent', 'sample'), onestep_cfg=float(pol.get('onestep_cfg', 1.0)), fill=pol.get('fill'), fill_floor=pol.get('fill_floor'), loops=pol.get('loops'), loop_edge_floor=pol.get('loop_edge_floor'), loop_face_floor=pol.get('loop_face_floor'), merge=pol.get('merge'), veto=pol.get('veto'))
