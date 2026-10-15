"""Tiny YAML config helper (no omegaconf/hydra in this environment)."""

from __future__ import annotations

import copy
import json
from typing import Any

import yaml


class Cfg(dict):
    """dict with attribute access and dotted get/set."""

    def __getattr__(self, k):
        try:
            v = self[k]
        except KeyError as e:
            raise AttributeError(k) from e
        return Cfg(v) if isinstance(v, dict) and not isinstance(v, Cfg) else v

    def __setattr__(self, k, v):
        self[k] = v

    def get_path(self, path: str, default=None):
        cur: Any = self
        for part in path.split("."):
            if not isinstance(cur, dict) or part not in cur:
                return default
            cur = cur[part]
        return cur

    def set_path(self, path: str, value):
        parts = path.split(".")
        cur = self
        for part in parts[:-1]:
            cur = cur.setdefault(part, {})
        cur[parts[-1]] = value


def _merge(base: dict, over: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge(out[k], v)
        else:
            out[k] = v
    return out


def _autotype(s: str):
    for fn in (json.loads,):
        try:
            return fn(s)
        except Exception:
            pass
    return s


def load_config(path: str, overrides: list[str] | None = None) -> Cfg:
    """Load YAML; ``_base_: other.yaml`` is merged in first. ``overrides`` are
    ``dotted.key=value`` strings (values parsed as JSON, else kept as string)."""
    import os

    with open(path) as f:
        raw = yaml.safe_load(f) or {}
    base = raw.pop("_base_", None)
    if base:
        base_path = base if os.path.isabs(base) else os.path.join(os.path.dirname(path), base)
        raw = _merge(load_config(base_path), raw)
    cfg = Cfg(raw)
    for ov in overrides or []:
        k, _, v = ov.partition("=")
        cfg.set_path(k.strip(), _autotype(v.strip()))
    return cfg


def save_config(cfg: dict, path: str):
    with open(path, "w") as f:
        yaml.safe_dump(json.loads(json.dumps(cfg)), f, sort_keys=False)
