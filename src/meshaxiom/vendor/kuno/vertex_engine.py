"""Exact CUDA-graph primitives and K/V caching, contributed by Sho Kuno.

The public wrapper fixes exact-shape graphs and separate guidance passes.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import time
import types

import numpy as np
import torch
import torch.nn.functional as F

from meshaxiom_core.models.octree_dit import cap_occupancy, node_positions
from meshaxiom_core.models.layers import padding_mask
from meshaxiom_core.models.layers import RoPE3D
from meshaxiom_core.data.mesh_ops import expand_children
from meshaxiom_core.diffusion.flow import LAMBDA_EPS, RectifiedFlow, occ_threshold

DEFAULT_BUCKETS = (8, 32, 64, 128, 256, 512, 768, 1024, 1280, 1536, 1792, 2048, 2304, 2560,
                   2816, 3072, 3328, 3584, 3840, 4096, 4608, 5120, 5632, 6144, 7168, 8192,
                   10240, 12288, 16384)
# fine buckets (lever 'fine'): padding waste <= 16/N below 1024, 64/N up to 4096; captured lazily, shared pool
FINE_BUCKETS = tuple(range(8, 1024, 16)) + tuple(range(1024, 4096, 64)) + tuple(range(4096, 16385, 256))


class StagedDiT(torch.nn.Module):
    """The frozen MeshAxiomDiT.forward, op for op, but also returning the block-stack input/output features
    (h_in, h_out) so that a step cache can replay ``h_out - h_in`` without running the 22 blocks."""

    def __init__(self, dit):
        super().__init__()
        self.dit = dit

    def forward(self, x, t, pos, ctx=None, lengths=None, level=None):
        d = self.dit
        c = d.t_embed(t)
        if c.dim() == 2:
            c = c[:, None]
        h = d.in_proj(x)
        if d.level_embed is not None:
            lv = d.level_embed(level)[:, None]
            c = c + lv
            h = h + lv
        h_in = h
        rope = d.rope.cos_sin(pos)
        mask = padding_mask(lengths, x.shape[1]) if lengths is not None else None
        for blk in d.blocks:
            h = blk(h, c, rope=rope, mask=mask, ctx=ctx)
        return d.final(h, c), h_in, h

    def skip(self, x, t, level, resid):
        """Cache replay: in_proj/level-embed/t-embed of the CURRENT step, blocks replaced by the cached residual."""
        d = self.dit
        c = d.t_embed(t)
        if c.dim() == 2:
            c = c[:, None]
        h = d.in_proj(x)
        if d.level_embed is not None:
            lv = d.level_embed(level)[:, None]
            c = c + lv
            h = h + lv
        return d.final(h + resid.to(h.dtype), c)


def sync():
    torch.cuda.synchronize()


def bf16_linear_copy(dit):
    """Deep copy of the DiT with nn.Linear weights/biases cast to bf16.

    Under ``torch.autocast(bfloat16)`` every nn.Linear casts its fp32 weight to bf16
    (round-to-nearest-even) before the GEMM, so a pre-cast bf16 weight gives the
    same GEMM inputs.  Everything that autocast keeps in fp32 (RMSNorm weights,
    embeddings, ctx_gate, LayerNorm) is left in fp32.
    """
    d = copy.deepcopy(dit)
    n = 0
    for m in d.modules():
        if isinstance(m, torch.nn.Linear):
            m.weight.data = m.weight.data.to(torch.bfloat16)
            if m.bias is not None:
                m.bias.data = m.bias.data.to(torch.bfloat16)
            n += m.weight.numel()
    return d, n


# --------------------------------------------------------------------------- #
# optional K/V cache for the cross attention (context is constant per request)
# --------------------------------------------------------------------------- #
def _cross_forward_cached(self, x, ctx, mask=None):
    """Replacement CrossAttention.forward: K/V come from ``self._kv_cache``.

    ``self._kv_cache`` = (k, v) with shape (B, H, M, hd), k already qk-normed,
    exactly what the frozen forward computes from ``ctx`` before SDPA.  When no
    cache is installed (``None``) the frozen forward runs unchanged.
    """
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


def compute_kv(cross, ctx):
    """Frozen CrossAttention K/V path (same ops/dtypes as its forward)."""
    b, m, _ = ctx.shape
    k, v = cross.kv(ctx).reshape(b, m, 2, cross.heads, cross.hd).permute(2, 0, 3, 1, 4)
    return cross.kn(k), v


# --------------------------------------------------------------------------- #
# mask-free padding: FlashAttention varlen with seqused_k (keys beyond N ignored)
# --------------------------------------------------------------------------- #
def _flash_varlen(q, k, v, vl, cross):
    """q/k/v (B, H, L, hd) -> (B, L, H*hd) via FA2 varlen: one sequence per batch
    element (= the full bucket), ``seqused_k`` limits each sequence to its real
    keys.  Padded *query* rows still produce (finite, discarded) outputs."""
    B, H, Lq, hd = q.shape
    Lk = k.shape[2]
    qf = q.transpose(1, 2).reshape(B * Lq, H, hd)
    kf = k.transpose(1, 2).reshape(B * Lk, H, hd)
    vf = v.transpose(1, 2).reshape(B * Lk, H, hd)
    cu_k, used = (vl['cu_ctx'], None) if cross else (vl['cu_q'], vl['seqused_k'])
    out = torch.ops.aten._flash_attention_forward(
        qf, kf, vf, vl['cu_q'], cu_k, Lq, Lk, 0.0, False, False,
        scale=1.0 / math.sqrt(hd), seqused_k=used)[0]
    return out.reshape(B, Lq, H * hd)


def _self_forward_varlen(self, x, rope=None, mask=None):
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


# --------------------------------------------------------------------------- #
# CUDA graph slot: static buffers + captured full DiT forward
# --------------------------------------------------------------------------- #
_SHARED_W16 = {}


def shared_w16(dit):
    """One bf16-Linear copy per process, shared by all engines (4 GB instead of 4 GB each)."""
    key = id(dit)
    if key not in _SHARED_W16:
        d, n = bf16_linear_copy(dit)
        d.eval()
        for blk in d.blocks:   # K/V-cache / varlen hooks with frozen fallback
            blk.cross.forward = types.MethodType(_cross_forward_cached, blk.cross)
            blk.cross._kv_cache = None
            blk.cross._varlen = None
            blk.attn.forward = types.MethodType(_self_forward_varlen, blk.attn)
            blk.attn._varlen = None
        _SHARED_W16[key] = (d, n)
    return _SHARED_W16[key]


_COMPILED = {}
_STAGED = {}

# Inductor knobs that decide (at compile time) which kernels/passes are chosen; recorded by inductor_provenance().
_INDUCTOR_KNOBS = ('shape_padding', 'force_shape_pad', 'max_autotune', 'max_autotune_pointwise', 'max_autotune_gemm',
                   'coordinate_descent_tuning', 'fx_graph_cache', 'autotune_local_cache')


def _inductor_setup():
    """Called before every torch.compile of this module (2026-09-18).  Inductor's pad_mm pass (config.shape_padding,
    default True) BENCHMARKS padded vs unpadded mm at compile time and pads iff ori_time > 1.1 x pad_time; the decision is
    stored in <TORCHINDUCTOR_CACHE_DIR>/cache/<system-hash> and reused on warm caches.  One cold-cache compile of this
    engine timed the M=2 (CFG batch) mm([2, K] bf16, [K, N] bf16) under noise and flipped its decision to 'pad'; a padded
    bf16 mm rounds differently and the sampler amplified that into different octrees (engine_check/REPORT_ja.md item 5),
    so outputs depended on the inductor cache state.  Every mm that reaches the benchmark here is already 8-aligned
    (K/N = 256/2048/4096/12288; token buckets are multiples of 16), so switching the pass off reproduces the all-'false'
    warm-cache decisions exactly and costs no speed.  Escape hatch NX_SHAPE_PADDING=1 restores the old default
    (validation only).  shape_padding is part of the fx-graph cache key, so caches written under the other setting are
    not reused.
    Second timing-dependent choice (found by the 2026-09-18 validation, engine_check/results/runs_padfix_*): with
    triton.autotune_pointwise=True (default) the runtime autotuner benchmarks several launch configs of the generated
    Triton kernels on first call and caches the winner (<kernel>.best_config).  For the persistent-reduction qk-RMSNorm
    kernel (reduce over the 128-wide head dim, triton_per_fused__flash_attention_forward__to_copy_mean_pow_*) the
    winner was XBLOCK=8 in some cold caches and XBLOCK=32 in others; a different block layout changes the tl.sum
    order, hence the rounding, hence the octree.  Kernel SOURCES were identical across all caches.  Autotuning off ->
    the first config of the heuristic list is used deterministically.  Escape hatch NX_AUTOTUNE_POINTWISE=1.
    inductor_provenance()['policy'] is 'deterministic' iff both knobs are off (the compile_v2 reference family)."""
    import torch._inductor.config as ic
    ic.shape_padding = os.environ.get('NX_SHAPE_PADDING', '0') == '1'
    ic.force_shape_pad = False
    ic.triton.autotune_pointwise = os.environ.get('NX_AUTOTUNE_POINTWISE', '0') == '1'


def inductor_provenance():
    """Cheap, never raises: torch/triton versions, the benchmark-related inductor knobs, the cache dirs, and a digest of the
    local pad_mm decision cache(s) (<cache_dir>/cache/<system-hash> JSON): sha256 over the sorted decision entries with the
    'base mm time' timing entries excluded, plus the number of 'true' (= pad) decisions.  Errors go into 'error'."""
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
        out['NX_SHAPE_PADDING'] = os.environ.get('NX_SHAPE_PADDING')
        out['NX_AUTOTUNE_POINTWISE'] = os.environ.get('NX_AUTOTUNE_POINTWISE')
        out['policy'] = 'deterministic' if (not ic.shape_padding and not ic.triton.autotune_pointwise) else 'legacy'
        out['TORCHINDUCTOR_CACHE_DIR'] = os.environ.get('TORCHINDUCTOR_CACHE_DIR')
        out['TRITON_CACHE_DIR'] = os.environ.get('TRITON_CACHE_DIR')
        from torch._inductor.runtime.runtime_utils import cache_dir
        cdir = os.path.join(cache_dir(), 'cache')
        files = sorted(os.listdir(cdir)) if os.path.isdir(cdir) else []
        decisions = {}
        for f in files:
            with open(os.path.join(cdir, f)) as fh:
                for k, v in (json.load(fh).get('cache') or {}).items():
                    if not k.startswith('base mm time'):
                        decisions[f'{f}:{k}'] = v
        out['pad_mm_cache_files'] = files
        out['pad_mm_decisions'] = len(decisions)
        out['pad_mm_true'] = sum(1 for v in decisions.values() if v is True)
        out['pad_mm_sha256'] = hashlib.sha256(json.dumps(sorted(decisions.items()), sort_keys=True).encode()).hexdigest() if decisions else None
    except Exception as e:  # noqa: BLE001
        out['error'] = repr(e)
    return out


def shared_staged(dit, compile_):
    """StagedDiT over the shared bf16 copy (optionally torch.compiled), one per process."""
    key = (id(dit), bool(compile_))
    if key not in _STAGED:
        st = StagedDiT(dit).eval()
        if compile_:
            import torch._dynamo
            torch._dynamo.config.cache_size_limit = 64
            torch._dynamo.config.capture_scalar_outputs = False
            _inductor_setup()
            _STAGED[key] = (st, torch.compile(st, dynamic=True, fullgraph=False))
        else:
            _STAGED[key] = (st, st)
    return _STAGED[key]


def shared_compiled(dit):
    """torch.compile (inductor, dynamic shapes) of the shared bf16 copy; captured by
    the manual CUDA graph afterwards.  Fuses the fp32 elementwise/norm/cast chains."""
    key = id(dit)
    if key not in _COMPILED:
        import torch._dynamo
        torch._dynamo.config.cache_size_limit = 64
        torch._dynamo.config.capture_scalar_outputs = False
        _inductor_setup()
        _COMPILED[key] = torch.compile(dit, dynamic=True, fullgraph=False)
    return _COMPILED[key]


def clear_kv(dit):
    for blk in dit.blocks:
        if hasattr(blk.cross, '_kv_cache'):
            blk.cross._kv_cache = None
            blk.cross._varlen = None
            blk.attn._varlen = None


def set_varlen(dit, vl):
    for blk in dit.blocks:
        blk.cross._varlen = vl
        blk.attn._varlen = vl


_CAPTURE_STREAM = []


def _capture_stream():
    if not _CAPTURE_STREAM:
        _CAPTURE_STREAM.append(torch.cuda.Stream())
    return _CAPTURE_STREAM[0]


class GraphSlot:
    def __init__(self, dit, B, Nb, ctx_shape, device, kvcache, masked=True, warmup=2, varlen=False,
                 compiled=None, pool=None):
        self.B, self.Nb, self.masked, self.warmup = B, Nb, masked, warmup
        self.compiled = compiled
        self.pool = pool                    # shared CUDA-graph memory pool (graphs never run concurrently)
        self.varlen = None
        if varlen:
            M = ctx_shape[0]
            self.varlen = {'cu_q': torch.arange(0, (B + 1) * Nb, Nb, dtype=torch.int32, device=device),
                           'cu_ctx': torch.arange(0, (B + 1) * M, M, dtype=torch.int32, device=device),
                           'seqused_k': torch.full((B,), Nb, dtype=torch.int32, device=device)}
            self.masked = False
        self.x = torch.zeros(B, Nb, 8, device=device)
        self.t = torch.zeros(B, device=device)
        self.pos = torch.zeros(B, Nb, 3, device=device)
        self.ctx = torch.zeros(B, *ctx_shape, device=device)
        self.lengths = torch.full((B,), Nb, dtype=torch.long, device=device)
        self.level = torch.ones(B, dtype=torch.long, device=device)
        self.kvcache = kvcache
        if kvcache:
            # static K/V per block, filled from ctx by ``fill_kv``
            self.kv = []
            for blk in dit.blocks:
                H, hd, M = blk.cross.heads, blk.cross.hd, ctx_shape[0]
                self.kv.append((torch.zeros(B, H, M, hd, dtype=torch.bfloat16, device=device),
                                torch.zeros(B, H, M, hd, dtype=torch.bfloat16, device=device)))
        self.dit = dit
        self.graph = None
        self.out = None
        self.capture_s = 0.0

    def _forward(self):
        clear_kv(self.dit)
        if self.kvcache:
            for blk, (k, v) in zip(self.dit.blocks, self.kv):
                blk.cross._kv_cache = (k, v)
        if self.varlen is not None:
            set_varlen(self.dit, self.varlen)
        fn = self.compiled if self.compiled is not None else self.dit
        with torch.autocast('cuda', dtype=torch.bfloat16, cache_enabled=False):
            return fn(self.x, self.t, self.pos, ctx=self.ctx,
                      lengths=self.lengths if self.masked else None, level=self.level)

    def capture(self, fast=True):
        """Warm up + capture on a persistent side stream.

        ``fast``: call ``CUDAGraph.capture_begin/capture_end`` directly.  The
        ``torch.cuda.graph`` context additionally runs ``gc.collect()`` and
        ``torch.cuda.empty_cache()`` on entry, which costs ~0.15-0.2 s per capture
        with this process's heap and forces fresh cudaMallocs for every pool.
        """
        sync()
        t0 = time.perf_counter()
        cur = torch.cuda.current_stream()
        s = _capture_stream()
        s.wait_stream(cur)
        with torch.cuda.stream(s):
            for _ in range(self.warmup):
                self._forward()
        if fast:
            with torch.cuda.stream(s):
                self.graph = torch.cuda.CUDAGraph()
                self.graph.capture_begin(pool=self.pool) if self.pool is not None else self.graph.capture_begin()
                try:
                    self.out = self._forward()
                finally:
                    self.graph.capture_end()
            cur.wait_stream(s)
        else:
            cur.wait_stream(s)
            self.graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self.graph, pool=self.pool):
                self.out = self._forward()
        sync()
        self.capture_s = time.perf_counter() - t0

    def fill_kv(self, ctx):
        """ctx (B, M, C) -> per-block static K/V (computed once per request/depth)."""
        with torch.autocast('cuda', dtype=torch.bfloat16, cache_enabled=False):
            for blk, (k, v) in zip(self.dit.blocks, self.kv):
                kk, vv = compute_kv(blk.cross, ctx)
                k.copy_(kk)
                v.copy_(vv)

    def replay(self):
        self.graph.replay()
        return self.out


# --------------------------------------------------------------------------- #
# engine
# --------------------------------------------------------------------------- #
class Engine:
    def __init__(self, model, *, cfgbatch=False, graph=False, pad=False, w16=False,
                 kvcache=False, nosync=True, steps=None, buckets=DEFAULT_BUCKETS,
                 exact_graph=False, varlen=False, compile=False, sync_debug=False, name='variant',
                 pool=False, cfg_every=1, cache=None, seed_offset=0, probe_decisions=False, early_exit=None):
        self.name = name
        # ---- levers added 2026-09-14 (each measured separately; all default off) ----
        self.pool = torch.cuda.graph_pool_handle() if pool else None   # shared graph memory pool
        # 2026-09-17: exact_graph captures one PRIVATE pool per depth per request (pool=None above is
        # not used by that path).  Clearing request_slots drops the references but does not return the
        # pools to the caching allocator, so dev 1,600 / train 19,200 runs died with CUDA OOM at ~105
        # requests with 119 GiB sitting in private pools.  Release them once the reserved footprint
        # crosses this high-water mark.  Memory management only; the arithmetic is untouched.
        self.exact_graph_gc_bytes = int(os.environ.get('NX_EXACT_GRAPH_GC_BYTES', 64 * (1 << 30)))
        self.graph_gc_events = 0
        self.cfg_every = int(cfg_every)          # >1: uncond velocity recomputed every k-th step, reused otherwise
        self.cache = dict(cache) if cache else None   # step cache: {'mode': 'easy'|'mag'|'fixed', ...}
        self.seed_offset = int(seed_offset)      # reference lane: re-seed the level generators (resampling noise)
        self.probe_decisions = bool(probe_decisions)   # log per-step thresholded-estimate agreement with the final occupancy
        self.early_exit = dict(early_exit) if early_exit else None   # {'min_steps': 8, 'stable': 3, 'depths': [1..5]}: stop the depth's ODE once the
                                                                     # thresholded denoised estimate has not changed for `stable` steps (final x = that estimate)
        self.cache_stats = []                    # per depth: full/skip decisions
        self.calib_records = []                  # residual norms per (depth, step) when cache mode == 'calib'
        self.model = model
        self.cfgbatch, self.graph, self.pad = cfgbatch, graph, pad
        self.exact_graph = exact_graph      # per-depth capture at the exact N, no mask, B=1
        self.varlen = varlen                # graph mode: FA2 varlen + seqused_k instead of bool mask
        self.compile = compile              # graph mode: inductor-fused forward inside the graph
        if varlen:
            assert graph, 'varlen is implemented for the graph path'

        if exact_graph:
            assert not (graph or pad or cfgbatch), 'exact_graph is a standalone mode'
        self.w16 = w16 or graph or pad or exact_graph
        self.kvcache = kvcache
        self.nosync = nosync
        self.steps_override = steps
        self.buckets = tuple(sorted(buckets))
        self.sync_debug = sync_debug
        self.device = next(model.parameters()).device
        self.dit = model.dit
        self.linear_params_bf16 = 0
        if self.w16:
            self.dit, self.linear_params_bf16 = shared_w16(model.dit)
        elif self.kvcache:
            raise ValueError('kvcache requires w16 (the frozen fp32 module is never patched)')
        self.slots: dict[tuple[int, int], GraphSlot] = {}
        if isinstance(buckets, str) and buckets == 'fine':
            self.buckets = FINE_BUCKETS
        if self.cache is not None:
            assert self.graph and self.cfgbatch, 'the step cache is implemented for the batched graph path'
            self.staged, self.staged_fn = shared_staged(self.dit, self.compile)
        if self.cfg_every > 1:
            assert self.graph and self.cfgbatch, 'cfg thinning is implemented for the batched graph path'
        self.request_slots = []
        self.capture_log = []
        self.timers = {}
        self.forwards = 0

    # ---- bookkeeping ---------------------------------------------------
    def reset_timers(self):
        self.timers = {}
        self.forwards = 0

    def bucket(self, n):
        for b in self.buckets:
            if b >= n:
                return b
        return None

    def slot(self, B, Nb):
        key = (B, Nb)
        if key not in self.slots:
            ctx_shape = (self.model.cond.num_tokens, self.model.cond.dim)
            fn = None
            if self.cache is not None:
                fn = self.staged_fn
            elif self.compile:
                fn = shared_compiled(self.dit)
            s = GraphSlot(self.dit, B, Nb, ctx_shape, self.device, self.kvcache, varlen=self.varlen,
                          compiled=fn, pool=self.pool)
            s.capture()
            self.slots[key] = s
            self.capture_log.append({'B': B, 'Nb': Nb, 'capture_s': s.capture_s})
        return self.slots[key]

    # ---- per-depth velocity function ------------------------------------
    def make_vel(self, N, pos, lvl, ctx, uc, cfg_scale, depth):
        """Return vel(x, t) -> float32 velocity (1, N, 8) with CFG applied.

        Arithmetic is identical to the frozen ``RectifiedFlow.sample.vel``:
        ``v = vu + cfg * (v - vu)`` in the network output dtype (bf16), then float.
        """
        dev = self.device
        B = 2 if self.cfgbatch else 1
        use_cfg = cfg_scale != 1.0
        if not use_cfg:
            B = 1
        dit = self.dit
        self_ = self

        def tb_of(tv, b):
            if self_.nosync:
                return tv.reshape(1).expand(b) if b > 1 else tv.reshape(1)
            return torch.full((b,), float(tv), device=dev)   # frozen behaviour (syncs)

        def combine(v, vu):
            if vu is None:
                return v.float()
            return (vu + cfg_scale * (v - vu)).float()

        if self.graph or self.pad:
            Nb = self.bucket(N)
        else:
            Nb = None

        # ---------- exact per-depth graph: shapes/kernels identical to the frozen path ----------
        if self.exact_graph:
            ctx_shape = (self.model.cond.num_tokens, self.model.cond.dim)
            slot = GraphSlot(dit, 1, N, ctx_shape, dev, self.kvcache, masked=False, warmup=1)
            slot.pos.copy_(pos)
            slot.level.fill_(int(depth))
            slot.ctx.copy_(ctx)
            kv_c = kv_u = None
            if self.kvcache:
                slot.fill_kv(slot.ctx)
                kv_c = [(k.clone(), v.clone()) for k, v in slot.kv]
                kv_u = []
                with torch.autocast('cuda', dtype=torch.bfloat16, cache_enabled=False):
                    for blk in dit.blocks:
                        kv_u.append(compute_kv(blk.cross, uc))
            slot.capture()
            self.request_slots.append(slot)
            self.capture_log.append({'B': 1, 'Nb': N, 'capture_s': slot.capture_s, 'exact': True})
            return self._vel_graph_b1(slot, N, ctx, uc, use_cfg, combine, kv_c, kv_u)

        # ---------- eager, no padding ----------
        if Nb is None:
            if self.kvcache:
                # eager kvcache: precompute per-block K/V for this depth
                ctxB = torch.cat([ctx, uc], 0) if (B == 2) else ctx
                kv_c = []
                with torch.autocast('cuda', dtype=torch.bfloat16, cache_enabled=False):
                    for blk in dit.blocks:
                        kv_c.append(compute_kv(blk.cross, ctxB))
                kv_u = None
                if use_cfg and B == 1:
                    kv_u = []
                    with torch.autocast('cuda', dtype=torch.bfloat16, cache_enabled=False):
                        for blk in dit.blocks:
                            kv_u.append(compute_kv(blk.cross, uc))

                def set_kv(kvs):
                    for blk, kv in zip(dit.blocks, kvs):
                        blk.cross._kv_cache = kv
            if B == 2:
                ctx2 = torch.cat([ctx, uc], 0)
                pos2 = pos.expand(2, -1, -1)
                lvl2 = lvl.expand(2)

                def vel(xt, tv, i=None):
                    self_.forwards += 1
                    if self_.kvcache:
                        set_kv(kv_c)
                    v2 = dit(xt.expand(2, -1, -1).contiguous(), tb_of(tv, 2), pos2, ctx=ctx2,
                             level=lvl2)
                    return combine(v2[0:1], v2[1:2])
                return vel

            def vel(xt, tv, i=None):
                tb = tb_of(tv, 1)
                self_.forwards += 1
                if self_.kvcache:
                    set_kv(kv_c)
                v = dit(xt, tb, pos, ctx=ctx, level=lvl)
                vu = None
                if use_cfg:
                    self_.forwards += 1
                    if self_.kvcache:
                        set_kv(kv_u)
                    vu = dit(xt, tb, pos, ctx=uc, level=lvl)
                return combine(v, vu)
            return vel

        # ---------- padded (eager or CUDA graph) ----------
        lengths = torch.full((B,), N, dtype=torch.long, device=dev)
        if self.graph:
            slot = self.slot(B, Nb)
            slot.x.zero_()
            slot.pos.zero_()
            slot.pos[:, :N].copy_(pos.expand(B, -1, -1))
            slot.lengths.fill_(N)
            if slot.varlen is not None:
                slot.varlen['seqused_k'].fill_(N)
            slot.level.fill_(int(depth))
            if B == 2:
                slot.ctx.copy_(torch.cat([ctx, uc], 0))
                if self.kvcache:
                    slot.fill_kv(slot.ctx)
            else:
                slot.ctx.copy_(ctx)
                kv_c = kv_u = None
                if self.kvcache:
                    slot.fill_kv(slot.ctx)
                    # a second slot-independent K/V set for the null ctx
                    kv_u = []
                    with torch.autocast('cuda', dtype=torch.bfloat16, cache_enabled=False):
                        for blk in dit.blocks:
                            kv_u.append(compute_kv(blk.cross, uc))
                    kv_c = [(k.clone(), v.clone()) for k, v in slot.kv]

            if B == 2:
                if self.cache is not None:
                    return self._vel_graph_cache(slot, N, depth, cfg_scale)
                if self.cfg_every > 1:
                    return self._vel_graph_cfgthin(slot, N, ctx, depth, combine)

                def vel(xt, tv, i=None):
                    self_.forwards += 1
                    slot.x[:, :N].copy_(xt.expand(2, -1, -1))
                    slot.t.copy_(tv.reshape(1).expand(2))
                    out = slot.replay()
                    return combine(out[0:1, :N], out[1:2, :N])
                return vel

            return self._vel_graph_b1(slot, N, ctx, uc, use_cfg, combine,
                                      kv_c if self.kvcache else None, kv_u if self.kvcache else None)

        # eager padded (diagnostic)
        padn = Nb - N
        posB = F.pad(pos, (0, 0, 0, padn)).expand(B, -1, -1).contiguous()
        lvlB = lvl.expand(B)
        ctxB = torch.cat([ctx, uc], 0) if B == 2 else ctx

        def vel(xt, tv, i=None):
            xB = F.pad(xt, (0, 0, 0, padn)).expand(B, -1, -1).contiguous()
            tb = tb_of(tv, B)
            self_.forwards += 1
            with torch.autocast('cuda', dtype=torch.bfloat16, cache_enabled=False):
                if B == 2:
                    v2 = dit(xB, tb, posB, ctx=ctxB, lengths=lengths, level=lvlB)
                    return combine(v2[0:1, :N], v2[1:2, :N])
                v = dit(xB, tb, posB, ctx=ctxB, lengths=lengths, level=lvlB)[:, :N]
                vu = None
                if use_cfg:
                    self_.forwards += 1
                    vu = dit(xB, tb, posB, ctx=uc, lengths=lengths, level=lvlB)[:, :N]
                return combine(v, vu)
        return vel

    def _vel_graph_cfgthin(self, slot2, N, ctx, depth, combine):
        """Lever 'cfg<k>': the unconditional velocity is recomputed only on steps i % k == 0 (and the first / last
        step); in between, the previous unconditional velocity is reused and only the conditional forward runs
        (batch-1 slot).  CFG arithmetic unchanged: v = vu + s (v - vu)."""
        self_ = self
        k = self.cfg_every
        Nb = slot2.Nb
        slot1 = self.slot(1, Nb)
        slot1.x.zero_(); slot1.pos.zero_(); slot1.pos[:, :N].copy_(slot2.pos[0:1, :N]); slot1.lengths.fill_(N)
        if slot1.varlen is not None:
            slot1.varlen['seqused_k'].fill_(N)
        slot1.level.fill_(int(depth))
        slot1.ctx.copy_(ctx)
        if self.kvcache:
            slot1.fill_kv(slot1.ctx)
        st = {'vu': None, 'steps': None, 'full': 0, 'thin': 0}

        def vel(xt, tv, i=None):
            last = (st['steps'] is not None and i is not None and i == st['steps'] - 1)
            full = i is None or st['vu'] is None or (i % k == 0) or last
            if full:
                self_.forwards += 1
                slot2.x[:, :N].copy_(xt.expand(2, -1, -1))
                slot2.t.copy_(tv.reshape(1).expand(2))
                out = slot2.replay()
                v, vu = out[0:1, :N], out[1:2, :N].clone()
                st['vu'] = vu; st['full'] += 1
            else:
                self_.forwards += 1
                slot1.x[:, :N].copy_(xt)
                slot1.t.copy_(tv.reshape(1))
                v = slot1.replay()[0:1, :N]
                vu = st['vu']; st['thin'] += 1
            return combine(v, vu)
        vel.state = st
        return vel

    def _vel_graph_cache(self, slot, N, depth, cfg_scale):
        """Lever 'easy<tau>' / 'mag<delta>:<K>' / 'fixed<k>': step cache on the whole block stack (cond+uncond
        batch together).  Full step: graph replay of the staged forward (out, h_in, h_out), residual r = h_out - h_in
        kept.  Skip step: eager in_proj / embeddings / final layer of the CURRENT step with the cached residual (no
        blocks).  Gate policies follow optimizations/tex_cache/step_cache.py of the Pixal3D study (EasyCache
        arXiv 2507.02860; MagCache with a per-depth calibrated ratio table; fixed:k)."""
        self_ = self
        cfg = self.cache
        mode = cfg['mode']
        st = {'steps': None, 'resid': None, 'E': 0.0, 'consec': 0, 'x_prev': None, 'x_prev2': None, 'v_prev': None, 'v_prev2': None,
              'decisions': [], 'resid_norms': [], 'depth': int(depth)}
        gamma = None
        if mode == 'mag':
            gamma = cfg['gamma'][str(int(depth))] if isinstance(cfg['gamma'], dict) else cfg['gamma']
        R = int(cfg.get('R', 3)); K = cfg.get('K'); tau = cfg.get('tau'); delta = cfg.get('delta'); fixed_k = cfg.get('k')
        combine_cfg = float(cfg_scale)

        def norm(a):
            return float(a.float().norm())

        def decide(i, xt):
            steps = st['steps']
            if i == 0 or (steps is not None and i == steps - 1) or st['resid'] is None:
                return True
            if mode == 'fixed':
                return i % int(fixed_k) == 0
            if mode == 'mag':
                st['E'] += abs(1.0 - float(gamma[i]))
                if st['E'] <= float(delta) and (K is None or st['consec'] < int(K)):
                    return False
                return True
            if mode == 'easy':
                if i < R:
                    return True
                xp, xp2, vp, vp2 = st['x_prev'], st['x_prev2'], st['v_prev'], st['v_prev2']
                den = norm(xp - xp2)
                kk = norm(vp - vp2) / den if den > 0 else 0.0
                nv = norm(vp)
                eps = kk * norm(xt - xp) / nv if nv > 0 else 0.0
                st['E'] += eps
                if st['E'] < float(tau) and (K is None or st['consec'] < int(K)):
                    return False
                return True
            if mode == 'calib':
                return True
            raise ValueError(mode)

        def vel(xt, tv, i=None):
            i = 0 if i is None else int(i)
            full = decide(i, xt)
            if full:
                self_.forwards += 1
                slot.x[:, :N].copy_(xt.expand(2, -1, -1))
                slot.t.copy_(tv.reshape(1).expand(2))
                out, h_in, h_out = slot.replay()
                st['resid'] = (h_out.float() - h_in.float())
                st['resid_norms'].append((i, norm(h_out.float() - h_in.float())))
                st['E'] = 0.0; st['consec'] = 0
                v2 = out
            else:
                st['consec'] += 1
                slot.x[:, :N].copy_(xt.expand(2, -1, -1))
                slot.t.copy_(tv.reshape(1).expand(2))
                with torch.autocast('cuda', dtype=torch.bfloat16, cache_enabled=False):
                    v2 = self_.staged.skip(slot.x, slot.t, slot.level, st['resid'])
            st['decisions'].append(int(full))
            v = (v2[1:2, :N] + combine_cfg * (v2[0:1, :N] - v2[1:2, :N])).float()
            # history for the easy policy (v = the CFG-combined velocity this step produced, real or replayed)
            st['x_prev2'], st['x_prev'] = st['x_prev'], xt.detach().clone()
            st['v_prev2'], st['v_prev'] = st['v_prev'], v.detach().clone()
            return v
        vel.state = st
        self.cache_stats.append(st)
        return vel

    def _vel_graph_b1(self, slot, N, ctx, uc, use_cfg, combine, kv_c, kv_u):
        """Batch-1 graph: two replays per step (cond, then uncond) with a ctx copy each."""
        self_ = self

        def vel(xt, tv, i=None):
            slot.x[:, :N].copy_(xt)
            slot.t.copy_(tv.reshape(1))
            self_.forwards += 1
            slot.ctx.copy_(ctx)
            if kv_c is not None:
                for (k, v), (kc, vc) in zip(slot.kv, kv_c):
                    k.copy_(kc); v.copy_(vc)
            v = slot.replay()[0:1, :N].clone()
            vu = None
            if use_cfg:
                self_.forwards += 1
                slot.ctx.copy_(uc)
                if kv_u is not None:
                    for (k, v_), (ku, vu_) in zip(slot.kv, kv_u):
                        k.copy_(ku); v_.copy_(vu_)
                vu = slot.replay()[0:1, :N]
            return combine(v, vu)
        return vel

    # ---- DPM++(2M): same arithmetic as frozen RectifiedFlow.sample -------
    def dpmpp2m(self, vel, x, steps, probe=None, thr=None, early=None):
        """``early`` = (min_steps, stable): return the denoised estimate d as soon as its thresholded decisions have been
        identical for `stable` consecutive steps (checked from step min_steps on).  At the last step the frozen solver's
        Euler update makes x = d anyway, so a settled estimate yields the same occupancy decisions as running on."""
        ts = torch.linspace(0.0, 1.0, steps + 1, device=x.device, dtype=torch.float32)
        lam = torch.logit(ts.clamp(LAMBDA_EPS, 1.0 - LAMBDA_EPS))
        d_prev, h_prev = None, None
        state = getattr(vel, 'state', None)
        if state is not None:
            state['steps'] = steps
        prev_dec, same = None, 0
        self.last_steps_run = steps
        for i in range(steps):
            tc, tn = ts[i], ts[i + 1]
            d = x + (1.0 - tc) * vel(x, tc, i)
            if probe is not None:
                probe.append((d[0] > thr).cpu().numpy() if thr is not None else None)
            if early is not None and thr is not None:
                dec = (d[0] > thr)
                if prev_dec is not None and bool(torch.equal(dec, prev_dec)):
                    same += 1
                else:
                    same = 0
                prev_dec = dec
                if i >= int(early[0]) and same >= int(early[1]) and i < steps - 1:
                    self.last_steps_run = i + 1
                    return d
            h = lam[i + 1] - lam[i]
            if d_prev is None or i == steps - 1:
                d_use = d
            else:
                r = h_prev / h
                d_use = (1 + 1 / (2 * r)) * d - (1 / (2 * r)) * d_prev
            emh = tc * (1.0 - tn) / ((1.0 - tc) * tn)
            x = (1.0 - tn) / (1.0 - tc) * x + tn * (1.0 - emh) * d_use
            d_prev, h_prev = d, h
        return x

    # ---- replacement for VertexModel.sample_octree ------------------------
    def sample_octree(self, vm, ctx, steps=20, sampler='dpmpp2m', cfg_scale=1.0, uncond_ctx=None,
                      max_nodes=0, flow=None, generator=None, level_generators=None,
                      verbose=False, return_info=False, occ_thr=None):
        assert ctx.shape[0] == 1 and sampler == 'dpmpp2m'
        if self.steps_override:
            steps = int(self.steps_override)
        dev = ctx.device
        thr = occ_threshold(vm.target_mode) if occ_thr is None else float(occ_thr)
        assert level_generators is not None and len(level_generators) == vm.depth_total
        levels = [np.zeros((1, 3), dtype=np.int64)]
        info = {'max_nodes': int(max_nodes), 'cap_events': [], 'empty_fallback_events': []}
        uc = None
        if cfg_scale != 1.0:
            uc = uncond_ctx if uncond_ctx is not None else vm.cond.null_ctx(1).to(ctx.dtype)
        T = self.timers
        T['steps'] = steps
        if self.seed_offset:
            for g in level_generators:
                g.manual_seed(int(g.initial_seed()) + self.seed_offset)
        if self.w16:
            clear_kv(self.dit)
        self.cache_stats = []
        T['decisions'] = {}
        T['cache'] = {}
        for d in range(1, vm.depth_total + 1):
            sync(); t_dep = time.perf_counter()
            parents = levels[-1]
            N = len(parents)
            pc = torch.from_numpy(parents).to(dev)[None]
            pos = node_positions(pc, vm.depth_total, d - 1)
            lvl = torch.full((1,), d, dtype=torch.long, device=dev)
            x0 = torch.randn(1, N, 8, device=dev, generator=level_generators[d - 1])
            vel = self.make_vel(N, pos, lvl, ctx, uc, cfg_scale, d)
            sync(); t_flow = time.perf_counter()
            if self.sync_debug:
                torch.cuda.set_sync_debug_mode('warn')
            probe = [] if self.probe_decisions else None
            early = None
            if self.early_exit and d in set(self.early_exit.get('depths', range(1, 10))):
                early = (self.early_exit.get('min_steps', 8), self.early_exit.get('stable', 3))
            occ = self.dpmpp2m(vel, x0, steps, probe=probe, thr=thr, early=early)[0].float()
            if early is not None:
                T.setdefault('early', {})[f'D{d}'] = int(self.last_steps_run)
            if probe is not None:
                final = (occ > thr).cpu().numpy()
                T['decisions'][f'D{d}'] = [int((p != final).sum()) for p in probe]   # per step: children whose thresholded estimate differs from the final decision
            cst = getattr(vel, 'state', None)
            if cst is not None and 'decisions' in cst:
                T['cache'][f'D{d}'] = {'full': int(sum(cst['decisions'])), 'skipped': int(len(cst['decisions']) - sum(cst['decisions'])),
                                      'decisions': cst['decisions'], 'resid_norms': cst['resid_norms']}
            elif cst is not None and 'full' in cst:
                T['cache'][f'D{d}'] = {'full': cst['full'], 'thin': cst['thin']}
            if self.sync_debug:
                torch.cuda.set_sync_debug_mode('default')
            sync(); t_occ = time.perf_counter()
            occ, before, capped = cap_occupancy(occ, thr, max_nodes)
            if capped:
                info['cap_events'].append({'level': d, 'predicted_nodes': before, 'kept_nodes': int(max_nodes)})
            child = expand_children(parents, occ.cpu().numpy(), thr)
            if len(child) == 0:
                info['empty_fallback_events'].append({'level': d, 'parents': int(N), 'policy': 'argmax_child_per_parent'})
                slot = occ.argmax(1, keepdim=True)
                onehot = torch.zeros_like(occ).scatter_(1, slot, 1.0)
                child = expand_children(parents, onehot.cpu().numpy(), 0.5)
            levels.append(child)
            if self.request_slots:
                self.request_slots.clear()   # per-depth exact graphs: free their pools
                if self.exact_graph and torch.cuda.memory_reserved() > self.exact_graph_gc_bytes:
                    import gc as _gc
                    _gc.collect(); torch.cuda.empty_cache()
                    self.graph_gc_events += 1
            t_end = time.perf_counter()
            T[f'D{d}'] = {'parents': int(N), 'nodes': int(len(child)),
                          'bucket': self.bucket(N) if (self.graph or self.pad) else (N if self.exact_graph else None),
                          'setup_s': t_flow - t_dep, 'flow_s': t_occ - t_flow,
                          'expand_s': t_end - t_occ, 'total_s': t_end - t_dep}
            if verbose:
                print(f'  [{self.name} level {d}] {N:6d} parents -> {len(child):6d} nodes  '
                      f'flow {t_occ - t_flow:.3f}s', flush=True)
        if self.w16:
            clear_kv(self.dit)
        return (levels, info) if return_info else levels

    def bind(self, vm):
        """Bound method for ``vm.sample_octree`` (instance-level monkeypatch)."""
        return types.MethodType(lambda _vm, *a, **k: self.sample_octree(_vm, *a, **k), vm)
