"""Inference-only functions extracted from the frozen production sample module."""
from __future__ import annotations
import os, hashlib, inspect
import numpy as np, torch
from meshaxiom_core.config import Cfg

def load_ckpt(path: str | None, what: str) -> dict:
    if not path:
        raise SystemExit(f"no {what} checkpoint given: pass --{what}-ckpt (or set it in the config)")
    if not os.path.isfile(path):
        raise SystemExit(f"{what} checkpoint not found: {path}")
    # The formal topology launch has 64 ranks loading the same ~1.3 GiB frozen VAE.
    # mmap keeps the checkpoint read-only and lets ranks on one node share the
    # kernel page cache instead of issuing 64 independent eager CPFS reads.  This
    # changes neither tensors nor model state; all checkpoint consumers are
    # read-only and torch's standard zip checkpoints support mmap loading.
    ck = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    if not isinstance(ck, dict) or not ({"model", "ema"} & set(ck)):
        raise SystemExit(f"{path} is not a MeshAxiom checkpoint (want a dict with 'model'/'ema', got "
                         f"{sorted(ck)[:8] if isinstance(ck, dict) else type(ck).__name__})")
    if "cfg" not in ck:
        raise SystemExit(f"{path} has no 'cfg' entry: cannot rebuild the {what} model")
    return ck



def load_weights(model: torch.nn.Module, ck: dict, what: str, use_ema: bool = True):
    """Load ``ema`` (preferred) or ``model`` into ``model``; missing tensors are fatal."""
    sd = ck["ema"] if use_ema and ck.get("ema") else ck.get("model")
    if sd is None:
        raise SystemExit(f"{what} checkpoint has neither 'ema' nor 'model' weights")
    out = model.load_state_dict(sd, strict=False)
    if out.missing_keys:
        raise SystemExit(f"{what} checkpoint is missing {len(out.missing_keys)} tensors, e.g. "
                         f"{out.missing_keys[:5]}; the config in the checkpoint does not match "
                         f"the weights")
    if out.unexpected_keys:
        raise SystemExit(f"{what} checkpoint has {len(out.unexpected_keys)} unexpected tensors, "
                         f"e.g. {out.unexpected_keys[:5]}; refusing a partial/ambiguous load")
    return model



def flow_from(cfg: Cfg):
    """``cfg.diffusion`` -> :class:`RectifiedFlow` (extra keys such as ``target_mode`` ignored)."""
    from meshaxiom_core.diffusion.flow import RectifiedFlow

    ps = inspect.signature(RectifiedFlow.__init__).parameters
    return RectifiedFlow(**{k: v for k, v in (cfg.get_path("diffusion") or {}).items() if k in ps})

