"""P1 fast topology engine for the MeshAxiom image->mesh acceleration study (meshaxiom_e2e_20261009).

The shipped topology stage (release topology_v119_e2e_20260930_v1, decode policy v4) runs the 2.0B-parameter Topology DiT
42 times per mesh in eager mode: 2 conditional-mean passes + 20 DPM++2M steps x (cond, uncond) with CFG 3, all at the SAME
token count V (the vertex count).  P0's profile (p0/profile): those forwards are 85% of the stage, the run is kernel-launch
bound (CPU self time == CUDA time), the kNN input features are recomputed inside every forward (12%), and autocast re-casts
the fp32 weights to bf16 at every Linear call (17% of CUDA time).

This module replaces ONLY the DiT part of `decode_policy90.decode` (the function `sample_latents`); the VAE decode, the
edge/face extraction, the pruning and the merge run unchanged (the release function is called with `sample_latents`
swapped in memory for the duration of the call; the release files are never edited).  The DPM++2M solver, the CFG
arithmetic and the self-conditioning are the release's own code paths (`RectifiedFlow.sample` is called with our model
closures), so the only differences are in how each DiT forward is executed.

Modes (`TopoFastEngine(mode=...)`):
  release : nothing replaced (A/B reference; `decode_policy90.decode` as shipped).
  eager   : no CUDA graphs.  bf16 pre-cast weight copy (w16), kNN features once per request, cross-attention K/V once per
            request; optional `cfgbatch` (cond + uncond in one batch-2 forward).  Diagnostic.
  graph   : one CUDA graph per request captured at the exact V (batch 1, no padding, no mask, flash attention as in the
            release), two replays per step (cond, then uncond with the null K/V swapped in) + 2 replays for the mean.
            Kernels and shapes identical to the release -> expected bit-exact (the vertex-stage engine's exact_graph2+kvcache
            was 28/28 bit-exact).  Cost: one capture (2 forwards) per request.
  fast    : V padded to a static bucket, one CUDA graph per bucket captured once per process (shared memory pool), cond +
            uncond as one batch-2 replay, FlashAttention varlen with seqused_k instead of a padding mask (padded keys never
            enter the softmax), optional torch.compile of the forward (inductor, deterministic knobs: pad_mm benchmark and
            pointwise autotuning OFF so the generated kernels do not depend on compile-time timings).  NOT bit-exact: the
            batch-2 GEMMs and the varlen kernel tile differently, and inductor fuses the fp32 elementwise chains.

Exactness levers that are safe (same arithmetic): w16 (autocast's own per-call cast, done once), kNN features once (the
same function on the same input), K/V once (the same GEMM shapes), CUDA graph at the exact V (same kernels).
Levers that change rounding: cfgbatch (GEMM M = 2V), padding + varlen attention, torch.compile.

Rules: new file of the study; release code read-only; GPU only through guardian_v3 on the assigned GPU.
"""
from __future__ import annotations

import contextlib
import copy
import gc
import hashlib
import json
import math
import os
import sys
import time
import types
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from meshaxiom.vendor.recipe import decode_policy90 as DP
from meshaxiom.vendor.recipe import model118rich as MR
from meshaxiom_core.models.layers import RoPE3D  # noqa: E402

# ----------------------------------------------------------------------------------------------- policy v4 contract
V4_REQUIRED = dict(latent='mean_and_sample', mean_estimator='onestep_mean', sampler='dpmpp2m', merge='draw_first_attached')


def check_policy_v4(pol):
    """The engine implements exactly the shipped v4 latent recipe; anything else must go through the release path."""
    for k, v in V4_REQUIRED.items():
        if pol.get(k) != v:
            raise ValueError(f'topo_fast implements decode policy v4 only: {k}={pol.get(k)!r} (want {v!r})')
    if float(pol.get('onestep_cfg', 1.0)) != 1.0:
        raise ValueError('topo_fast: onestep_cfg must be 1.0 (v4)')
    if int(pol.get('extra_draws', 0)) != 0:
        raise ValueError('topo_fast: extra_draws must be 0 (v4)')
    if int(pol.get('sc_passes', 2)) < 1:
        raise ValueError('topo_fast: sc_passes must be >= 1')


# ----------------------------------------------------------------------------------------------- buckets
def make_buckets():
    b = list(range(64, 4096 + 1, 64)) + list(range(4096 + 128, 8192 + 1, 128)) + list(range(8192 + 256, 16384 + 1, 256)) \
        + list(range(16384 + 512, 26112 + 1, 512))
    return tuple(sorted(set(b)))


DEFAULT_BUCKETS = make_buckets()


def bucket_of(n, buckets=DEFAULT_BUCKETS):
    for b in buckets:
        if b >= n:
            return b
    raise ValueError(f'{n} tokens exceed the largest bucket {buckets[-1]}')


# ----------------------------------------------------------------------------------------------- weight copy / hooks
def bf16_linear_copy(net):
    """Deep copy of the MeshAxiomDiT with every nn.Linear weight/bias cast to bf16 (round-to-nearest-even), which is exactly
    what torch.autocast(bfloat16) does per call to the fp32 weights (the release model has requires_grad=False, so autocast
    never caches the cast).  RMSNorm / LayerNorm / embedding parameters stay fp32 as under autocast."""
    d = copy.deepcopy(net)
    n = 0
    for m in d.modules():
        if isinstance(m, torch.nn.Linear):
            m.weight.data = m.weight.data.to(torch.bfloat16)
            if m.bias is not None:
                m.bias.data = m.bias.data.to(torch.bfloat16)
            n += m.weight.numel()
    return d.eval(), n


def compute_kv(cross, ctx):
    """The frozen CrossAttention K/V path (same ops / dtypes as its forward): -> (k (B,H,M,hd) qk-normed, v)."""
    b, m, _ = ctx.shape
    k, v = cross.kv(ctx).reshape(b, m, 2, cross.heads, cross.hd).permute(2, 0, 3, 1, 4)
    return cross.kn(k), v


def _flash_varlen(q, k, v, vl, cross):
    """q/k/v (B,H,L,hd) -> (B,L,H*hd) with FlashAttention-2 varlen: one sequence per batch row (= the whole bucket),
    `seqused_k` limits each row to its real keys.  Padded QUERY rows still get (finite, discarded) outputs."""
    B, H, Lq, hd = q.shape
    Lk = k.shape[2]
    qf = q.transpose(1, 2).reshape(B * Lq, H, hd)
    kf = k.transpose(1, 2).reshape(B * Lk, H, hd)
    vf = v.transpose(1, 2).reshape(B * Lk, H, hd)
    cu_k, used = (vl['cu_ctx'], None) if cross else (vl['cu_q'], vl['seqused_k'])
    out = torch.ops.aten._flash_attention_forward(qf, kf, vf, vl['cu_q'], cu_k, Lq, Lk, 0.0, False, False,
                                                  scale=1.0 / math.sqrt(hd), seqused_k=used)[0]
    return out.reshape(B, Lq, H * hd)


def _cross_forward_hooked(self, x, ctx, mask=None):
    """CrossAttention.forward replacement: K/V from `self._kv_cache` when installed; varlen when `self._varlen` is set;
    otherwise the frozen forward unchanged."""
    cache = getattr(self, '_kv_cache', None)
    vl = getattr(self, '_varlen', None)
    if cache is None and vl is None:
        return type(self).forward(self, x, ctx, mask)
    b, n, _ = x.shape
    q = self.q(x).reshape(b, n, self.heads, self.hd).transpose(1, 2)
    k, v = cache if cache is not None else compute_kv(self, ctx)
    if vl is not None:
        return self.proj(_flash_varlen(self.qn(q), k, v, vl, cross=True))
    x = F.scaled_dot_product_attention(self.qn(q), k, v, attn_mask=mask)
    return self.proj(x.transpose(1, 2).reshape(b, n, -1))


def _self_forward_hooked(self, x, rope=None, mask=None):
    vl = getattr(self, '_varlen', None)
    if vl is None:
        return type(self).forward(self, x, rope=rope, mask=mask)
    b, n, _ = x.shape
    q, k, v = self.qkv(x).reshape(b, n, 3, self.heads, self.hd).permute(2, 0, 3, 1, 4)
    q, k = self.qn(q), self.kn(k)
    if rope is not None:
        q = RoPE3D.apply(q, *rope)
        k = RoPE3D.apply(k, *rope)
    return self.proj(_flash_varlen(q, k, v, vl, cross=False))


def install_hooks(net):
    for blk in net.blocks:
        blk.cross.forward = types.MethodType(_cross_forward_hooked, blk.cross)
        blk.cross._kv_cache = None
        blk.cross._varlen = None
        blk.attn.forward = types.MethodType(_self_forward_hooked, blk.attn)
        blk.attn._varlen = None


def set_kv(net, kvs):
    for blk, kv in zip(net.blocks, kvs if kvs is not None else [None] * len(net.blocks)):
        blk.cross._kv_cache = kv


def set_varlen(net, vl):
    for blk in net.blocks:
        blk.cross._varlen = vl
        blk.attn._varlen = vl


# ----------------------------------------------------------------------------------------------- inductor
_INDUCTOR_KNOBS = ('shape_padding', 'force_shape_pad', 'max_autotune', 'max_autotune_pointwise', 'max_autotune_gemm',
                   'coordinate_descent_tuning', 'fx_graph_cache')


def inductor_setup():
    """Deterministic inductor policy (same as the vertex-stage engine, engine_check/REPORT_ja.md item 5): the pad_mm
    pass benchmarks padded vs unpadded mm at compile time and the pointwise autotuner picks block sizes by timing; both
    made the generated kernels depend on the cache state.  Off -> the first heuristic config is used deterministically."""
    import torch._inductor.config as ic
    import torch._dynamo
    ic.shape_padding = False
    ic.force_shape_pad = False
    ic.triton.autotune_pointwise = False
    torch._dynamo.config.cache_size_limit = 64
    torch._dynamo.config.capture_scalar_outputs = False
    # numerics experiments (coordinator 2026-10-09 15:1x, item 2): env knobs, recorded by inductor_provenance()
    if os.environ.get('NX_EMULATE_PRECISION_CASTS') == '1' and hasattr(ic, 'emulate_precision_casts'):
        ic.emulate_precision_casts = True        # inductor re-rounds at eager's intermediate bf16 casts inside fused kernels
    if os.environ.get('NX_EPILOGUE_FUSION') == '0':
        ic.epilogue_fusion = False
    if os.environ.get('NX_MAX_FUSION_SIZE'):
        ic.max_fusion_size = int(os.environ['NX_MAX_FUSION_SIZE'])
    if os.environ.get('NX_TF32') == '0':
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False


def inductor_provenance():
    out = {}
    try:
        out['torch'] = torch.__version__
        try:
            import triton
            out['triton'] = triton.__version__
        except Exception as e:  # noqa: BLE001
            out['triton'] = repr(e)
        import torch._inductor.config as ic
        for k in _INDUCTOR_KNOBS:
            out[k] = getattr(ic, k, None)
        out['triton.autotune_pointwise'] = ic.triton.autotune_pointwise
        out['emulate_precision_casts'] = getattr(ic, 'emulate_precision_casts', 'n/a')
        out['epilogue_fusion'] = getattr(ic, 'epilogue_fusion', None)
        out['max_fusion_size'] = getattr(ic, 'max_fusion_size', None)
        out['allow_tf32_matmul'] = torch.backends.cuda.matmul.allow_tf32
        out['env_knobs'] = {k: os.environ.get(k) for k in ('NX_EMULATE_PRECISION_CASTS', 'NX_EPILOGUE_FUSION', 'NX_MAX_FUSION_SIZE', 'NX_TF32')}
        out['TORCHINDUCTOR_CACHE_DIR'] = os.environ.get('TORCHINDUCTOR_CACHE_DIR')
        out['TRITON_CACHE_DIR'] = os.environ.get('TRITON_CACHE_DIR')
        from torch._inductor.runtime.runtime_utils import cache_dir
        out['cache_dir'] = cache_dir()
    except Exception as e:  # noqa: BLE001
        out['error'] = repr(e)
    return out


# ----------------------------------------------------------------------------------------------- static forward
class StaticForward(torch.nn.Module):
    """x (B,N,lat) fp32, sc (B,N,lat) fp32, feats (B,N,F_KNN) fp32, t (B,), pos (B,N,3), ctx (B,M,C) -> velocity (B,N,lat).
    Op for op the release's TopologyModel113.forward after its kNN call: cat -> MeshAxiomDiT(x_ext, t, pos, ctx, lengths=None)."""

    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, x, sc, feats, t, pos, ctx):
        return self.net(torch.cat([x, sc, feats], dim=-1), t, pos, ctx=ctx, lengths=None)


def _sync(dev):
    if dev.type == 'cuda':
        torch.cuda.synchronize(dev)


_CAPTURE_STREAM = []


def capture_stream():
    if not _CAPTURE_STREAM:
        _CAPTURE_STREAM.append(torch.cuda.Stream())
    return _CAPTURE_STREAM[0]


class GraphSlot:
    """Static buffers + one captured forward for (B, Nb)."""

    def __init__(self, fwd, net, B, Nb, lat, fk, M, C, device, varlen, pool, warmup, kv_static=True):
        self.B, self.Nb, self.fwd, self.net, self.pool, self.warmup = B, Nb, fwd, net, pool, warmup
        self.x = torch.zeros(B, Nb, lat, device=device)
        self.sc = torch.zeros(B, Nb, lat, device=device)
        self.feats = torch.zeros(B, Nb, fk, device=device)
        self.t = torch.zeros(B, device=device)
        self.pos = torch.zeros(B, Nb, 3, device=device)
        self.ctx = torch.zeros(B, M, C, device=device, dtype=torch.bfloat16)
        self.kv = None
        if kv_static:
            self.kv = []
            for blk in net.blocks:
                H, hd = blk.cross.heads, blk.cross.hd
                self.kv.append((torch.zeros(B, H, M, hd, dtype=torch.bfloat16, device=device),
                                torch.zeros(B, H, M, hd, dtype=torch.bfloat16, device=device)))
        self.varlen = None
        if varlen:
            self.varlen = {'cu_q': torch.arange(0, (B + 1) * Nb, Nb, dtype=torch.int32, device=device),
                           'cu_ctx': torch.arange(0, (B + 1) * M, M, dtype=torch.int32, device=device),
                           'seqused_k': torch.full((B,), Nb, dtype=torch.int32, device=device)}
        self.graph = None
        self.out = None
        self.capture_s = 0.0
        self.replays = 0

    def _run(self):
        set_kv(self.net, self.kv)
        set_varlen(self.net, self.varlen)
        try:
            with torch.autocast('cuda', dtype=torch.bfloat16, cache_enabled=False):
                return self.fwd(self.x, self.sc, self.feats, self.t, self.pos, self.ctx)
        finally:
            set_kv(self.net, None)
            set_varlen(self.net, None)

    def capture(self):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        cur = torch.cuda.current_stream()
        s = capture_stream()
        s.wait_stream(cur)
        with torch.cuda.stream(s):
            for _ in range(self.warmup):
                self._run()
            self.graph = torch.cuda.CUDAGraph()
            self.graph.capture_begin(pool=self.pool) if self.pool is not None else self.graph.capture_begin()
            try:
                self.out = self._run()
            finally:
                self.graph.capture_end()
        cur.wait_stream(s)
        torch.cuda.synchronize()
        self.capture_s = time.perf_counter() - t0

    def replay(self):
        self.graph.replay()
        self.replays += 1
        return self.out

    def free(self):
        """Destroy the captured graph (the allocator's pool use_count drops) and drop the captured output tensor."""
        g, self.graph, self.out = self.graph, None, None
        if g is not None:
            g.reset()
            del g


# ----------------------------------------------------------------------------------------------- engine
class TopoFastEngine:
    def __init__(self, model, device, mode='graph', cfgbatch=None, compile=False, compile_mode='default',
                 buckets=DEFAULT_BUCKETS, w16=True, kvcache=True, record=False, name=None,
                 graph_gc_bytes=48 * (1 << 30)):
        assert mode in ('release', 'eager', 'graph', 'fast'), mode
        self.model, self.device, self.mode = model, torch.device(device), mode
        self.cuda = self.device.type == 'cuda'
        self.cfgbatch = (mode == 'fast') if cfgbatch is None else bool(cfgbatch)
        self.compile = bool(compile)
        self.compile_mode = compile_mode
        self.buckets = tuple(sorted(buckets))
        self.kvcache = bool(kvcache)
        self.record = bool(record)
        self.rec = None
        self.name = name or mode
        self.lat = int(model.latent_dim)
        self.fk = int(MR.F_KNN)
        self.M = int(model.cond.num_tokens)
        self.C = int(model.cond.dim)
        self.graph_gc_bytes = int(graph_gc_bytes)
        self.timers = {}
        self.capture_log = []
        self.forwards = 0
        self.graph_gc_events = 0
        net = model.dit.net
        self.w16 = bool(w16) and self.cuda and mode != 'release'
        if self.w16:
            self.net, self.linear_params_bf16 = bf16_linear_copy(net)
        else:
            self.net, self.linear_params_bf16 = (copy.deepcopy(net).eval() if mode != 'release' else net), 0
        if mode != 'release':
            install_hooks(self.net)
        self.fwd = StaticForward(self.net).eval()
        self.fwd_fn = self.fwd
        if self.compile:
            assert self.cuda and mode == 'fast', 'compile is implemented for the fast (bucketed graph) mode'
            inductor_setup()
            self.fwd_fn = torch.compile(self.fwd, dynamic=True, fullgraph=False,
                                        mode=None if compile_mode == 'default' else compile_mode)
        # One memory pool shared by this engine's graphs.  The caching allocator keeps a pool alive while at least one
        # CUDAGraph captured into it exists (use_count); once the last one is destroyed the pool id is dead and the next
        # capture_begin into it asserts (it->second->use_count > 0, seen 2026-10-09 14:51 in p1_corr_t01).  So: (1) `graph`
        # mode captures the NEW request's graph before freeing the previous one (live count 1 -> 2 -> 1, the pool never
        # dies and the freed blocks are reused by the next capture), (2) whenever no graph is alive a FRESH handle is taken.
        self.pool = None
        self._live_graphs = 0
        self.slots = {}          # fast: (B, Nb) -> GraphSlot (kept for the process lifetime)
        self._req = None         # per-request state
        self._req_slot = None    # graph mode: the per-request slot

    # ---- bookkeeping -------------------------------------------------------------------------------------------
    def reset_timers(self):
        self.timers = {}
        self.forwards = 0

    def provenance(self):
        return dict(name=self.name, mode=self.mode, cfgbatch=self.cfgbatch, compile=self.compile, compile_mode=self.compile_mode,
                    w16=self.w16, kvcache=self.kvcache, linear_params_bf16=self.linear_params_bf16,
                    buckets=('fine64' if self.buckets == DEFAULT_BUCKETS else list(self.buckets)) if self.mode == 'fast' else None,
                    inductor=inductor_provenance() if self.compile else None)

    # ---- per-request setup ---------------------------------------------------------------------------------------
    def prepare(self, c, ctx):
        """c (1,V,3) int64 coords on device; ctx (1,M,C) as `encode_condition` returned it (bf16 under autocast)."""
        dev = self.device
        _sync(dev); t0 = time.perf_counter()
        V = int(c.shape[1])
        model = self.model
        pos = c.float() + 0.5
        feats = MR.knn_features(c, None).to(torch.float32)          # once per request (the release recomputes it per forward)
        uc = model.cond.null_ctx(1).to(ctx.dtype)
        kv_c = kv_u = None
        if self.kvcache and self.mode != 'release':
            with (torch.autocast('cuda', dtype=torch.bfloat16, cache_enabled=False) if self.cuda else contextlib.nullcontext()):
                kv_c = [compute_kv(blk.cross, ctx) for blk in self.net.blocks]
                kv_u = [compute_kv(blk.cross, uc) for blk in self.net.blocks]
        req = dict(V=V, c=c, pos=pos, feats=feats, ctx=ctx, uc=uc, kv_c=kv_c, kv_u=kv_u, slot=None, kv_state=None)
        _sync(dev); t1 = time.perf_counter()
        cap = 0.0
        if self.mode == 'graph':
            B = 2 if self.cfgbatch else 1
            slot = GraphSlot(self.fwd_fn, self.net, B, V, self.lat, self.fk, self.M, self.C, dev, varlen=False, pool=None, warmup=1)
            slot.pos.copy_(pos.expand(B, -1, -1))
            slot.feats.copy_(feats.expand(B, -1, -1))
            if B == 2:
                slot.ctx.copy_(torch.cat([ctx, uc], 0))
                for (k, v), (kc, vc), (ku, vu) in zip(slot.kv, kv_c, kv_u):
                    k[0:1].copy_(kc); v[0:1].copy_(vc); k[1:2].copy_(ku); v[1:2].copy_(vu)
            else:
                slot.ctx.copy_(ctx)
                for (k, v), (kc, vc) in zip(slot.kv, kv_c):
                    k.copy_(kc); v.copy_(vc)
                req['kv_state'] = 'c'
            self._capture(slot)                 # new graph first (pool stays alive) ...
            self._free_request_slot()           # ... then the previous request's graph goes
            cap = slot.capture_s
            self.capture_log.append(dict(B=B, Nb=V, exact=True, capture_s=cap))
            self._req_slot = slot
            req['slot'] = slot
        elif self.mode == 'fast':
            B = 2 if self.cfgbatch else 1
            Nb = bucket_of(V, self.buckets)
            slot = self.slots.get((B, Nb))
            if slot is None:
                slot = GraphSlot(self.fwd_fn, self.net, B, Nb, self.lat, self.fk, self.M, self.C, dev, varlen=True, pool=None,
                                 warmup=2 if self.compile else 1)
                self._capture(slot)
                cap = slot.capture_s
                self.capture_log.append(dict(B=B, Nb=Nb, exact=False, capture_s=cap))
                self.slots[(B, Nb)] = slot
            slot.x.zero_(); slot.sc.zero_(); slot.pos.zero_(); slot.feats.zero_()
            slot.pos[:, :V].copy_(pos.expand(B, -1, -1))
            slot.feats[:, :V].copy_(feats.expand(B, -1, -1))
            slot.varlen['seqused_k'].fill_(V)
            if B == 2:
                slot.ctx.copy_(torch.cat([ctx, uc], 0))
                for (k, v), (kc, vc), (ku, vu) in zip(slot.kv, kv_c, kv_u):
                    k[0:1].copy_(kc); v[0:1].copy_(vc); k[1:2].copy_(ku); v[1:2].copy_(vu)
            else:
                slot.ctx.copy_(ctx)
                for (k, v), (kc, vc) in zip(slot.kv, kv_c):
                    k.copy_(kc); v.copy_(vc)
                req['kv_state'] = 'c'
            req['slot'] = slot
        _sync(dev)
        self.timers['prepare_s'] = t1 - t0
        self.timers['capture_s'] = cap
        self.timers['V'] = V
        self.timers['bucket'] = req['slot'].Nb if req['slot'] is not None else None
        self._req = req
        return req

    def _capture(self, slot):
        """Capture `slot` into the engine's shared pool; take a fresh pool handle when no graph of ours is alive."""
        if self._live_graphs == 0 or self.pool is None:
            self.pool = torch.cuda.graph_pool_handle()
        slot.pool = self.pool
        slot.capture()
        self._live_graphs += 1

    def _free_request_slot(self):
        if self._req_slot is not None:
            self._req_slot.free()
            self._req_slot = None
            self._live_graphs -= 1
            if torch.cuda.memory_reserved() > self.graph_gc_bytes:
                gc.collect(); torch.cuda.empty_cache(); self.graph_gc_events += 1

    def finish_request(self):
        self._req = None
        if self.mode == 'graph':
            self._free_request_slot()

    # ---- the forward (one or two rows) ---------------------------------------------------------------------------
    def _forward(self, x, t, sc, which):
        """x (1,V,lat) fp32, t (1,) fp32, sc (1,V,lat) fp32 or None, which in ('c', 'u', 'cu') -> list of bf16 (1,V,lat):
        [v_cond] for 'c', [v_uncond] for 'u', [v_cond, v_uncond] for 'cu'."""
        req = self._req
        V = req['V']
        sc = torch.zeros_like(x) if sc is None else sc
        outs = []
        slot = req['slot']
        if slot is None:   # eager
            with (torch.autocast('cuda', dtype=torch.bfloat16, cache_enabled=False) if self.cuda else contextlib.nullcontext()):
                if which == 'cu' and self.cfgbatch:
                    set_kv(self.net, [(torch.cat([kc, ku], 0), torch.cat([vc, vu], 0)) for (kc, vc), (ku, vu) in zip(req['kv_c'], req['kv_u'])]
                           if req['kv_c'] is not None else None)
                    o = self.fwd_fn(x.expand(2, -1, -1).contiguous(), sc.expand(2, -1, -1).contiguous(), req['feats'].expand(2, -1, -1).contiguous(),
                                    t.expand(2), req['pos'].expand(2, -1, -1).contiguous(), torch.cat([req['ctx'], req['uc']], 0))
                    set_kv(self.net, None)
                    outs = [o[0:1], o[1:2]]
                    self.forwards += 1
                else:
                    for w in (['c', 'u'] if which == 'cu' else [which]):
                        set_kv(self.net, req['kv_c'] if w == 'c' else req['kv_u'])
                        o = self.fwd_fn(x, sc, req['feats'], t, req['pos'], req['ctx'] if w == 'c' else req['uc'])
                        set_kv(self.net, None)
                        outs.append(o)
                        self.forwards += 1
        else:
            B = slot.B
            slot.x[:, :V].copy_(x.expand(B, -1, -1))
            slot.sc[:, :V].copy_(sc.expand(B, -1, -1))
            slot.t.copy_(t.reshape(1).expand(B))
            if B == 2:
                o = slot.replay(); self.forwards += 1
                if which == 'cu':
                    outs = [o[0:1, :V].clone(), o[1:2, :V].clone()]
                elif which == 'c':
                    outs = [o[0:1, :V].clone()]          # row 1 (null ctx) is computed and discarded
                else:
                    outs = [o[1:2, :V].clone()]
            else:
                for w in (['c', 'u'] if which == 'cu' else [which]):
                    if req['kv_state'] != w:            # swap the static K/V between ctx and null ctx
                        src = req['kv_c'] if w == 'c' else req['kv_u']
                        for (k, v), (ks, vs) in zip(slot.kv, src):
                            k.copy_(ks); v.copy_(vs)
                        req['kv_state'] = w
                    o = slot.replay(); self.forwards += 1
                    outs.append(o[0:1, :V].clone())
        if self.rec is not None:
            self.rec.extend(o_.detach().clone() for o_ in outs)
        return outs

    # ---- decode policy v4 latent recipe (replaces decode_policy90.sample_latents) -----------------------------------
    def sample_latents(self, model, ctx, c, flow, generator, pol=None):
        """Same signature and the same random-draw order as decode_policy90.sample_latents for pol v4:
        1) conditional mean: x0m ~ N(0,1), v = f(x0m, t=0, sc=0); sc = x0m + v; v = f(x0m, 0, sc)  [sc_passes=2]; h = x0m + v
        2) low-temperature draw: x0 ~ noise_scale * N(0,1); DPM++2M 20 steps with CFG 3 and self-conditioning,
           through the release's RectifiedFlow.sample with our closures (release arithmetic for CFG / solver / sc).
        Returns {'mean': h_mean, 'draw': h_draw, 'extra': []} (denormalised, fp32), as the release does."""
        pol = pol or DP.policy()
        check_policy_v4(pol)
        assert model is self.model
        dev = self.device
        if self._req is None or not torch.equal(self._req['c'], c) or self._req['ctx'] is not ctx:
            self.prepare(c, ctx)
        req = self._req
        self.last_latents = None
        b, v, _ = c.shape
        assert b == 1
        self.rec = [] if self.record else None
        _sync(dev); t0 = time.perf_counter()
        # ---- (1) conditional mean (onestep_mean, onestep_cfg 1.0, sc_passes) ----
        x0m = torch.randn(b, v, self.lat, device=dev, generator=generator)
        tz = torch.zeros(b, device=dev)
        vel = self._forward(x0m, tz, None, 'c')[0]
        for _ in range(int(pol.get('sc_passes', 2)) - 1):
            sc = (x0m + vel.float()).detach()
            vel = self._forward(x0m, tz, sc, 'c')[0]
        h_mean = model.dit.denormalize_latent(x0m + vel.float())
        _sync(dev); t1 = time.perf_counter()
        # ---- (2) low-temperature draw with CFG + self-conditioning through the release solver ----
        x0 = torch.randn(b, v, self.lat, device=dev, generator=generator) * float(pol.get('noise_scale', 1.0))
        s_cfg = float(pol['cfg'])
        state = {'sc': None, 'used': None, 'stash': None}
        use_cfg = s_cfg != 1.0
        self_ = self

        def fn(x, t):
            state['used'] = state['sc']
            if use_cfg and self_.cfgbatch:
                vc, vu = self_._forward(x, t, state['sc'], 'cu')
                state['stash'] = (x, t, vu)
            else:
                vc = self_._forward(x, t, state['sc'], 'c')[0]
            tt = t.reshape(-1, 1, 1).to(x.dtype)
            state['sc'] = (x + (1.0 - tt) * vc.to(x.dtype)).detach()
            return vc

        def unc(x, t):
            st = state['stash']
            if st is not None and st[0] is x and st[1] is t:
                state['stash'] = None
                return st[2]
            return self_._forward(x, t, state['used'], 'u')[0]

        x1 = flow.sample(fn, x0, steps=int(pol['steps']), sampler=pol['sampler'], cfg_scale=s_cfg, uncond_fn=unc if use_cfg else None)
        h_draw = model.dit.denormalize_latent(x1)
        _sync(dev); t2 = time.perf_counter()
        self.timers['mean_s'] = t1 - t0
        self.timers['draw_s'] = t2 - t1
        self.timers['forwards'] = self.forwards
        self.last_latents = (h_mean.detach().clone(), h_draw.detach().clone())
        return {'mean': h_mean, 'draw': h_draw, 'extra': []}

    # ---- decode_fn for e2e_common.TopologyStage ---------------------------------------------------------------------
    def decode(self, model, vae, coords, ctx, lod, flow, device, generator, pol=None):
        """Drop-in for decode_policy90.decode: the release function runs with our sample_latents swapped in for the call."""
        self.reset_timers()
        if self.mode == 'release':
            _sync(self.device); t0 = time.perf_counter()
            out = DP.decode(model, vae, coords, ctx, lod, flow, device, generator, pol)
            _sync(self.device); self.timers['decode_s'] = time.perf_counter() - t0
            return out
        pol = pol or DP.policy()
        check_policy_v4(pol)
        c = torch.from_numpy(np.ascontiguousarray(coords)).to(self.device)[None]
        _sync(self.device); t0 = time.perf_counter()
        self.prepare(c, ctx)
        try:
            with _patched(DP, 'sample_latents', self.sample_latents):
                out = DP.decode(model, vae, coords, ctx, lod, flow, device, generator, pol)
        finally:
            self.finish_request()
        _sync(self.device); self.timers['decode_s'] = time.perf_counter() - t0
        self.timers['post_s'] = self.timers['decode_s'] - self.timers.get('prepare_s', 0) - self.timers.get('capture_s', 0) \
            - self.timers.get('mean_s', 0) - self.timers.get('draw_s', 0)
        return out


@contextlib.contextmanager
def _patched(module, name, fn):
    old = getattr(module, name)
    setattr(module, name, fn)
    try:
        yield
    finally:
        setattr(module, name, old)


# ----------------------------------------------------------------------------------------------- comparisons
def canonical_face_set(faces):
    from mesh_utils90 import canonical_faces
    f = canonical_faces(np.asarray(faces, np.int64).reshape(-1, 3))
    return set(map(tuple, f.tolist()))


def face_compare(faces_a, faces_b):
    """Jaccard of canonical face sets and the strict F1 of a against b (mesh_utils90.face_set_metrics)."""
    from mesh_utils90 import face_set_metrics
    A, B = canonical_face_set(faces_a), canonical_face_set(faces_b)
    inter, union = len(A & B), len(A | B)
    m = face_set_metrics(np.asarray(faces_a, np.int64).reshape(-1, 3), np.asarray(faces_b, np.int64).reshape(-1, 3))
    return dict(jaccard=(inter / union) if union else 1.0, inter=inter, union=union, n_a=len(A), n_b=len(B),
                f1=m['f1'], precision=m['precision'], recall=m['recall'], identical=(A == B))


def edge_compare(edges_a, edges_b):
    def s(e):
        e = np.asarray(e, np.int64).reshape(-1, 2)
        return set(map(tuple, np.unique(np.sort(e, axis=1), axis=0).tolist())) if len(e) else set()
    A, B = s(edges_a), s(edges_b)
    inter, union = len(A & B), len(A | B)
    return dict(jaccard=(inter / union) if union else 1.0, n_a=len(A), n_b=len(B), identical=(A == B))


def sha_faces(faces):
    return hashlib.sha256(np.ascontiguousarray(np.asarray(faces, np.int32).reshape(-1, 3)).tobytes()).hexdigest()
