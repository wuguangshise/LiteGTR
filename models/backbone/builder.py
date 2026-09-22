"""Backbone registry.

Baselines plug in here so that ``configs/baselines/*.yaml`` reuse LiteGTR's
neck, head, losses, assigner and schedule unchanged -- the unified-protocol
requirement from docs/DESIGN.md P1-6.
"""
from __future__ import annotations

import torch.nn as nn


def build_backbone(cfg: dict, use_p2: bool) -> nn.Module:
    kind = cfg.get("type", "tinynext").lower()
    out_indices = tuple(range(4)) if use_p2 else (1, 2, 3)
    common = dict(out_indices=out_indices)

    if kind == "tinynext":
        from models.backbone.tinynext import TinyNeXt

        return TinyNeXt(channels=tuple(cfg["channels"]), depths=tuple(cfg["depths"]),
                        drop_path_rate=cfg.get("drop_path_rate", 0.0), **common)
    if kind == "csp":
        from models.baselines.csp_baseline import CSPBackbone

        return CSPBackbone(channels=tuple(cfg["channels"]), depths=tuple(cfg["depths"]), **common)
    raise ValueError(f"unknown backbone type: {kind!r} (expected 'tinynext' or 'csp')")
