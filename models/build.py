"""Config-driven construction. Ablations flip a YAML key, never edit code."""
from __future__ import annotations

import copy
from pathlib import Path

import yaml


def _deep_update(base: dict, other: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in other.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_update(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def load_config(*paths: str | Path) -> dict:
    """Merge YAML files left to right; ``_base_`` inside a file is resolved first."""
    cfg: dict = {}
    for p in paths:
        p = Path(p)
        with open(p, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        for base in data.pop("_base_", []) or []:
            cfg = _deep_update(cfg, load_config((p.parent / base).resolve()))
        cfg = _deep_update(cfg, data)
    return cfg


def active_token_count(token_cfg: dict) -> int:
    """Tokens actually selected: the budget summed over the ACTIVE levels only.

    YAML ``_base_`` merging overrides keys but never deletes them, so an ablation
    that narrows ``levels`` (e.g. to ``[P5]``) still inherits the base config's
    ``P3``/``P4`` budget entries. The selector ignores them -- it only builds
    levels listed in ``levels`` -- but ``sum(budget.values())`` would report the
    wrong number.
    """
    if not token_cfg.get("enabled", True):
        return 0
    return sum(token_cfg["budget"][lv] for lv in token_cfg["levels"])


def count_deploy_params(model) -> int:
    """Learnable parameters that ship at inference -- the number for the paper.

    Counts ``nn.Parameter``s only, excluding the training-only EMA teacher.
    Summing ``state_dict()`` instead would also count BatchNorm running_mean /
    running_var buffers, which are not parameters: for the BN-heavy CSP baseline
    that made the "deployment" count exceed the full training-parameter count.
    """
    return sum(p.numel() for n, p in model.named_parameters() if not n.startswith("ema_router."))


def build_model(cfg: dict):
    # imported lazily so load_config() works without torch installed
    from models.detector import LiteGTR

    return LiteGTR(cfg)
