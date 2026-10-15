"""P7 (2026-10-15, decision 28 tier 1): ExactBackend — the BIT-EXACT vertex engine under the shipped coarse-consensus control flow.

Plugs into code/p4_vertex_engine.sample_mbr (the release sampler_mbr.sample_mbr control flow: candidate seeds gens[k][d-1], per-level
noise, expansion, level-3 consensus, continuation of k*) as a third backend next to EagerBackend / FastBackend:

  levels 1..select_level (K candidates, "coarse"):  THE RELEASE'S OWN ARITHMETIC — one lengths-masked eager batch of K rows for the
      conditional pass and one for the unconditional pass through release RectifiedFlow.sample (the sampler_mbr `mf` / `uf` closures,
      = EagerBackend.multi), on the bf16-Linear copy of the DiT that the exact engine runs (identical GEMM inputs to autocast's per-call
      weight cast; P7_COARSE_DIT=fp32 selects the original fp32 module for diagnosis).  Small (<= 64 parents per candidate): eager is enough.
  levels select_level+1..9 (K=1, "single"):  the vertex study's `--variant exact` path (meshaxiom_c4_200k_20260917/bench/vertex_engine.py,
      exact_graph2+kvcache: engine_check/REPORT_ja.md 28/28 bit-identical with the frozen release, 1.47x) built from its frozen pieces:
      per level one CUDA graph captured at EXACTLY N tokens (no padding, no length mask, private memory pool, warm-up 1), the cross-attention
      K/V of the request-constant context computed ONCE per level (vertex_engine.compute_kv == the frozen CrossAttention K/V ops) into static
      bf16 buffers, the two guidance passes replayed SEPARATELY (cond, then uncond), v = vu + cfg (v - vu) in bf16 then float
      (= RectifiedFlow.sample.vel), the DPM++(2M) loop = vertex_engine.Engine.dpmpp2m (same arithmetic as RectifiedFlow.sample).
      ONE DIFFERENCE IN BOOKKEEPING to the engine's make_vel: the conditional and the unconditional pass get their OWN slot (two graphs per
      level) instead of one slot with K/V copied before every replay.  Reason: with the 180K weights the shipped sampler feeds the vendor's
      duplicated 2050-token conditional context while the null context has 1025 tokens (release180k/IMPL_P6_ja.md §4.3), so the static
      K/V buffers (B, H, M, hd) of the two passes have different M and cannot share a captured graph.  Each graph records the same kernels
      the single-slot engine records (same shapes, same dtypes, same inputs), so the numbers are the same; per step the two replays only
      skip the K/V copies.  Works unchanged when both contexts have 1025 tokens (200K weights / 180K `image` mode).
  OPTION coarse='graph' (P7_COARSE=graph; default 'eager' = the brief): the SAME lengths-masked K-row forward captured in exact-shape CUDA
      graphs (B = K, Nb = N, masked=True, K/V of the K context rows once per level, one graph per guidance pass), replayed 20 times.  The
      captured graph records the kernels the eager forward launches for these shapes (the exact-variant argument), so the numbers are the
      eager batch's; what goes away is the CPU launch overhead that makes the eager coarse stage ~2.9 s (24 ms per forward for <= 64 tokens).
      Bit-exactness of this option is to be confirmed on the GPU (p7_chain_gpu.sh small_cg) before it is used.
  BETWEEN-REQUEST MAINTENANCE (16:1x, for the strict timing): maintain() = the same gc.collect() + empty_cache(), called by
      p4_pipeline.FastVertex.run BEFORE the timed span starts, when reserved memory exceeds the mark minus P7_GC_MARGIN_GIB (default 16 GiB,
      > the ~2 GiB a request adds), so that the in-level rule (the safety net, unchanged) does not fire inside a timed vertex span.
  GC rule (vertex study README_ja.md §1-0): every exact graph owns a private pool that is not returned to the caching allocator when the
      slot is dropped -> ~105 requests OOM.  After every single level the slot references are dropped (request_slots.clear()) and, when
      torch.cuda.memory_reserved() exceeds engine.exact_graph_gc_bytes (NX_EXACT_GRAPH_GC_BYTES, default 64 GiB), gc.collect() +
      torch.cuda.empty_cache() run (engine.graph_gc_events counts them).  Arithmetic untouched.
Seeds: handled by sample_mbr (gens[k][d-1]; k* continues with its own streams) — the backend never draws noise.
multi_for_k1 = True: like the release, K=1 still runs levels 1..select_level through the (trivially) masked batch path.
CPU test: test_p7_cpu.py injects a FakeSlot (eager forward on the static buffers) through `slot_cls`, so the whole single-level code path
runs on CPU and must equal EagerBackend / release sampler_mbr bit for bit.
Rules: new file of the study; vertex_engine (vertex study, frozen) and the release code are imported read-only.
"""
from __future__ import annotations

import gc
import os
import time

import torch

EXACT_FLAGS = dict(exact_graph=True, kvcache=True)      # = vertex study dagger/rollout.py ENGINES['exact'] = bench/user_samples.py 'exact_graph2+kvcache'
EXACT_NAME = 'exact_graph2+kvcache'


def build_exact_engine(vm, name=EXACT_NAME):
    """vertex_engine.Engine in the vertex study's `--variant exact` configuration (bf16-Linear shared copy, no buckets, no compile)."""
    import vertex_engine as nf
    return nf.Engine(vm, name=name, **EXACT_FLAGS)


class ExactBackend:
    """See the module docstring.  `engine`: vertex_engine.Engine(exact_graph=True, kvcache=True) (on CUDA; on CPU any eager Engine for the
    control-flow test), `flow`: the release RectifiedFlow of the vertex model (coarse levels), `coarse_dit`: module for the coarse eager
    batch (default engine.dit = the shared bf16-Linear copy), `slot_cls`: vertex_engine.GraphSlot (default) or a test double."""
    name = EXACT_NAME
    multi_for_k1 = True

    def __init__(self, engine, flow, coarse_dit=None, slot_cls=None, coarse='eager', warmup=1):
        import vertex_engine as nf
        self.nf = nf
        self.engine = engine
        self.flow = flow
        self.device = engine.device
        self.graph_path = bool(getattr(engine, 'exact_graph', False) and engine.kvcache and engine.w16)
        if self.device.type == 'cuda' and not self.graph_path:
            raise ValueError('ExactBackend needs vertex_engine.Engine(exact_graph=True, kvcache=True) (bf16 shared copy) on CUDA')
        self.slot_cls = slot_cls if slot_cls is not None else nf.GraphSlot
        self.coarse_dit = coarse_dit if coarse_dit is not None else engine.dit
        self.coarse_dit_kind = 'w16_shared' if self.coarse_dit is engine.dit else ('fp32_model' if self.coarse_dit is engine.model.dit else 'custom')
        if coarse not in ('eager', 'graph'):
            raise ValueError(coarse)
        self.coarse = coarse
        self.warmup = int(warmup)       # warm-up forwards run on the capture stream before each capture (vertex study exact: 1); 0 = capture directly
        self.multi_log = []          # one entry per coarse level (read by p4_pipeline.FastVertex.run)
        self.level_log = []          # one entry per single level: N, context tokens, capture seconds, gc
        self.gc_events_request = 0
        self.gc_between = 0              # maintenance GCs run outside the timed spans (total)
        self.gc_between_last = False     # whether maintain() ran a GC before the current request
        self.gc_margin_bytes = int(float(os.environ.get('P7_GC_MARGIN_GIB', 16)) * (1 << 30))
        self.requests = 0

    # ---- bookkeeping ------------------------------------------------------------------------------
    @property
    def forwards(self):
        return int(self.engine.forwards)

    def _reserved(self):
        return int(torch.cuda.memory_reserved()) if self.device.type == 'cuda' else 0

    def _gc_enabled(self):
        return self.device.type == 'cuda'

    def _empty_cache(self):
        gc.collect()
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()

    def _release(self):
        """= vertex_engine.Engine.sample_octree's per-depth clean-up: drop the exact slots (their graphs and private pools) and apply the
        high-water-mark GC rule.  Returns True when a GC ran."""
        eng = self.engine
        if eng.request_slots:
            eng.request_slots.clear()
        if self._gc_enabled() and self._reserved() > int(eng.exact_graph_gc_bytes):
            self._empty_cache()
            eng.graph_gc_events += 1
            self.gc_events_request += 1
            return True
        return False

    def maintain(self):
        """Untimed (call before the request's timed span): release the exact graphs' private pools when reserved memory is within
        gc_margin_bytes of the in-level mark.  Same calls as the in-level rule; arithmetic untouched.  Returns True when a GC ran."""
        eng = self.engine
        if eng.request_slots:
            eng.request_slots.clear()
        self.gc_between_last = False
        if self._gc_enabled() and self._reserved() > int(eng.exact_graph_gc_bytes) - self.gc_margin_bytes:
            self._empty_cache()
            self.gc_between += 1
            self.gc_between_last = True
        return self.gc_between_last

    def begin(self):
        self.engine.reset_timers()
        if self.engine.w16:
            self.nf.clear_kv(self.engine.dit)
        if self.engine.request_slots:
            self.engine.request_slots.clear()
        self.level_log = []
        self.gc_events_request = 0
        self.requests += 1

    def end(self):
        if self.engine.w16:
            self.nf.clear_kv(self.engine.dit)
        self._release()

    def provenance(self):
        eng = self.engine
        return dict(name=self.name, flags=dict(EXACT_FLAGS), engine_name=eng.name, graph_path=self.graph_path, slot_cls=self.slot_cls.__name__,
                    coarse_mode=self.coarse, warmup=self.warmup,
                    coarse=('release arithmetic: lengths-masked eager batch of K rows, cond and uncond passes separately, RectifiedFlow.sample' if self.coarse == 'eager'
                            else 'release arithmetic captured: lengths-masked K-row forward in exact-shape CUDA graphs (cond slot, uncond slot), K/V once per level, Engine.dpmpp2m'),
                    coarse_dit=self.coarse_dit_kind, single='exact-N CUDA graph per level and per guidance pass (cond slot M_c, uncond slot M_u), K/V once per level, Engine.dpmpp2m',
                    exact_graph_gc_bytes=int(eng.exact_graph_gc_bytes), exact_graph_gc_gib=round(int(eng.exact_graph_gc_bytes) / 2**30, 1), gc_margin_gib=round(self.gc_margin_bytes / 2**30, 1),
                    linear_params_bf16=int(getattr(eng, 'linear_params_bf16', 0)), nosync=bool(getattr(eng, 'nosync', True)))

    # ---- coarse levels: the release's own arithmetic ---------------------------------------------
    def multi(self, ns, pc, pos, lvl, ctx, uc, cfg, d, x0, steps, sampler):
        """= release sampler_mbr levels 1..select_level (= p4_vertex_engine.EagerBackend.multi) on `coarse_dit`; coarse='graph' -> _multi_graph."""
        if self.coarse == 'graph':
            return self._multi_graph(ns, pc, pos, lvl, ctx, uc, cfg, d, x0, steps, sampler)
        K = len(ns); dev = ctx.device
        dit = self.coarse_dit
        if self.engine.w16:
            self.nf.clear_kv(self.engine.dit)        # no K/V-cache / varlen hooks installed: the frozen forward runs
        lengths = torch.tensor(ns, device=dev)
        cK, uK = ctx.expand(K, -1, -1), uc.expand(K, -1, -1)
        mf = lambda x, t: dit(x, t, pos, ctx=cK, lengths=lengths, level=lvl)   # noqa: E731
        uf = lambda x, t: dit(x, t, pos, ctx=uK, lengths=lengths, level=lvl)   # noqa: E731
        use_cfg = cfg != 1.0
        t0 = time.perf_counter()
        occ = self.flow.sample(mf, x0, steps=steps, sampler=sampler, cfg_scale=cfg, uncond_fn=uf if use_cfg else None).float()
        self.engine.forwards += int(steps) * (2 if use_cfg else 1)
        self.multi_log.append(dict(level=int(d), K=K, B=K, N=int(max(ns)), Nb=None, ns=list(map(int, ns)), path='eager_release_masked_batch',
                                   ctx_tokens=int(ctx.shape[1]), uncond_tokens=int(uc.shape[1]), cfg=float(cfg), dit=self.coarse_dit_kind, seconds=time.perf_counter() - t0))
        return occ

    def _slot_batch(self, K, N, pos, ns, d, ctx):
        """One exact-shape (B = K, Nb = N) lengths-masked slot for one context (expanded to the K rows): K/V filled once, graph captured."""
        eng = self.engine
        M, Cd = int(ctx.shape[1]), int(ctx.shape[2])
        s = self.slot_cls(eng.dit, K, N, (M, Cd), self.device, kvcache=True, masked=True, warmup=self.warmup)
        s.pos.copy_(pos)
        s.level.fill_(int(d))
        s.lengths.copy_(torch.tensor(ns, dtype=torch.long, device=self.device))
        s.ctx.copy_(ctx.expand(K, -1, -1))
        s.fill_kv(s.ctx)
        s.capture()
        eng.request_slots.append(s)
        eng.capture_log.append({'B': int(K), 'Nb': int(N), 'capture_s': float(s.capture_s), 'exact': True, 'ctx_tokens': M, 'masked': True})
        return s

    def _multi_graph(self, ns, pc, pos, lvl, ctx, uc, cfg, d, x0, steps, sampler):
        assert sampler == 'dpmpp2m', sampler
        eng = self.engine
        K = len(ns); N = int(max(ns))
        use_cfg = cfg != 1.0
        t0 = time.perf_counter()
        slot_c = self._slot_batch(K, N, pos, ns, d, ctx)
        slot_u = self._slot_batch(K, N, pos, ns, d, uc) if use_cfg else None
        t_cap = time.perf_counter() - t0

        def vel(xt, tv, i=None):
            eng.forwards += 1
            slot_c.x.copy_(xt)
            slot_c.t.copy_(tv.reshape(1).expand(K))
            v = slot_c.replay()
            if not use_cfg:
                return v.float()
            eng.forwards += 1
            slot_u.x.copy_(xt)
            slot_u.t.copy_(tv.reshape(1).expand(K))
            vu = slot_u.replay()
            return (vu + cfg * (v - vu)).float()      # = RectifiedFlow.sample.vel on the K-row batch

        occ = eng.dpmpp2m(vel, x0, steps)
        del vel, slot_c, slot_u
        gc_ran = self._release()
        self.multi_log.append(dict(level=int(d), K=K, B=K, N=N, Nb=N, ns=list(map(int, ns)), path='exact_graph_masked_batch', ctx_tokens=int(ctx.shape[1]), uncond_tokens=int(uc.shape[1]),
                                   cfg=float(cfg), dit='w16_shared', capture_s=t_cap, gc=bool(gc_ran), seconds=time.perf_counter() - t0))
        return occ.float()

    # ---- single levels: exact-N graphs ------------------------------------------------------------
    def _slot(self, N, pos, d, ctx):
        """One exact-N slot for one context: static buffers, K/V filled once, graph captured (warm-up 1, private pool)."""
        eng = self.engine
        M, Cd = int(ctx.shape[1]), int(ctx.shape[2])
        s = self.slot_cls(eng.dit, 1, N, (M, Cd), self.device, kvcache=True, masked=False, warmup=self.warmup)
        s.pos.copy_(pos)
        s.level.fill_(int(d))
        s.ctx.copy_(ctx)
        s.fill_kv(s.ctx)
        s.capture()
        eng.request_slots.append(s)
        eng.capture_log.append({'B': 1, 'Nb': int(N), 'capture_s': float(s.capture_s), 'exact': True, 'ctx_tokens': M})
        return s

    def single(self, N, pos, lvl, ctx, uc, cfg, d, x0, steps, sampler):
        assert sampler == 'dpmpp2m', sampler
        eng = self.engine
        use_cfg = cfg != 1.0
        t0 = time.perf_counter()
        slot_c = self._slot(N, pos, d, ctx)
        slot_u = self._slot(N, pos, d, uc) if use_cfg else None
        t_cap = time.perf_counter() - t0

        def vel(xt, tv, i=None):
            eng.forwards += 1
            slot_c.x[:, :N].copy_(xt)
            slot_c.t.copy_(tv.reshape(1))
            v = slot_c.replay()[0:1, :N]
            if not use_cfg:
                return v.float()
            eng.forwards += 1
            slot_u.x[:, :N].copy_(xt)
            slot_u.t.copy_(tv.reshape(1))
            vu = slot_u.replay()[0:1, :N]
            return (vu + cfg * (v - vu)).float()      # = RectifiedFlow.sample.vel: bf16 combine, then float

        occ = eng.dpmpp2m(vel, x0, steps)
        del vel, slot_c, slot_u                        # the graphs and their private pools are now referenced by request_slots only
        gc_ran = self._release()
        self.level_log.append(dict(level=int(d), N=int(N), ctx_tokens=int(ctx.shape[1]), uncond_tokens=(int(uc.shape[1]) if use_cfg else None), cfg=float(cfg),
                                   slots=(2 if use_cfg else 1), capture_s=t_cap, gc=bool(gc_ran), reserved_gib=(self._reserved() / 2**30 if self.device.type == 'cuda' else None)))
        return occ.float()
