"""Load the single released EMA recipe once per isolated stage process."""
from __future__ import annotations

import gc
import json
from pathlib import Path
import sys

_STATE = None
_SIGNATURE = None


def _identity(request, stage, exact):
    geometry = request["config"]["geometry"]
    files = {}
    for key in ["vertex_checkpoint", "topology_checkpoint", "vae_checkpoint"]:
        spec = geometry[key]
        stat = Path(spec["path"]).stat()
        files[key] = [spec, stat.st_size, stat.st_mtime_ns, stat.st_ino]
    return json.dumps([stage, exact, request["device"], geometry.get("image_model"),
                       request["config"]["runtime"].get("gpu_memory_fraction"), files], sort_keys=True)


def load(request, stage, *, exact=True):
    global _STATE, _SIGNATURE
    signature = _identity(request, stage, exact)
    if _STATE is not None:
        if signature != _SIGNATURE:
            raise ValueError("Resident worker model/device changed; close it before selecting a different configuration")
        return _STATE
    from meshaxiom.vendor.recipe.runtime import activate_namespace, checkpoint, cuda_device, image_pixels
    activate_namespace(stage)
    import torch
    from meshaxiom_core.config import Cfg
    from meshaxiom_core.sample import flow_from, load_weights
    from meshaxiom.workers.offline_encoder import offline_encoder
    dev = cuda_device(request)
    torch.cuda.set_device(dev)
    if stage == "vertex":
        from meshaxiom_core.models.octree_dit import VertexModel
        ck, artifact = checkpoint(request["config"]["geometry"]["vertex_checkpoint"], "vertex")
        cfg = Cfg(ck["cfg"])
        if cfg.get_path("condition.type") != "image" or cfg.get_path("condition.fusion") is not None or cfg.get_path("depth") != 9:
            raise ValueError("Released vertex model requires D9 and image conditioning")
        # Match the original worker: override local processor/model before construction.
        image_pixels(cfg, request)
        with offline_encoder(request["config"]["geometry"]["image_model"]):
            net = load_weights(VertexModel(cfg), ck, "vertex", True).to(dev).eval().requires_grad_(False)
        del ck
        state = dict(dev=dev, cfg=cfg, net=net, flow=flow_from(cfg), artifact=artifact)
        if exact:
            root = Path(__file__).resolve().parents[1] / "vendor" / "kuno"
            sys.path.insert(0, str(root))
            from p7_exact_backend import build_exact_engine, ExactBackend
            engine = build_exact_engine(net)
            backend = ExactBackend(engine, state["flow"], coarse="graph", warmup=0)
            # Same memory-only change as Kuno's exact180k; condition tower stays fp32.
            net.dit = engine.dit
            state.update(engine=engine, backend=backend)
    else:
        from meshaxiom_core.models.topo_vae import build_vae
        from meshaxiom.vendor.recipe import decode_policy90 as decoder
        ck, artifact = checkpoint(request["config"]["geometry"]["topology_checkpoint"], "topology")
        vck, vae_artifact = checkpoint(request["config"]["geometry"]["vae_checkpoint"], "vae")
        basis = vck.get("source_checkpoint_sha256", vae_artifact["sha256"])
        if ck.get("vae_artifact", {}).get("sha256") != basis or ck["vae_artifact"].get("weights") != "ema":
            raise ValueError("Topology/VAE latent basis or weight selection mismatch")
        cfg = Cfg(ck["cfg"])
        if cfg.get_path("depth") != 9 or Cfg(vck["cfg"]).get_path("data.depth", 9) != 9:
            raise ValueError("Topology and VAE must use D9 grid coordinates")
        image_pixels(cfg, request)
        with offline_encoder(request["config"]["geometry"]["image_model"]):
            model = decoder.build_topology_model(ck, cfg, use_ema=True).to(dev).eval().requires_grad_(False)
        vae = load_weights(build_vae(Cfg(vck["cfg"])), vck, "vae", True).to(dev).eval().requires_grad_(False)
        del ck, vck
        state = dict(dev=dev, cfg=cfg, model=model, vae=vae, flow=flow_from(cfg),
                     artifact=artifact, vae_artifact=vae_artifact, basis=basis)
        if exact:
            root = Path(__file__).resolve().parents[1] / "vendor" / "kuno"
            sys.path.insert(0, str(root))
            from meshaxiom.vendor.kuno.topo_fast import TopoFastEngine
            from meshaxiom.vendor.kuno import p5_post_fast
            engine = TopoFastEngine(model, dev, name="exact", mode="graph", cfgbatch=False,
                                    graph_gc_bytes=64 * (1 << 30))
            state.update(engine=engine, post=p5_post_fast, post_warmup=p5_post_fast.warmup())
    gc.collect()
    _SIGNATURE, _STATE = signature, state
    return state
