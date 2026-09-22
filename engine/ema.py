"""Exponential moving average of MODEL WEIGHTS (standard detection practice).

Not to be confused with ``models/token/ema_token_router.py``, which is the EMA
*teacher for token routing* and is part of the method. This one is a generic
training-stability trick applied to the whole network, evaluated instead of the
raw weights when enabled.

Both can be on at once; they serve different purposes and are ablated separately.
"""
from __future__ import annotations

import copy
import math

import torch
import torch.nn as nn


class ModelEMA:
    def __init__(self, model: nn.Module, decay: float = 0.9998, tau: int = 2000,
                 updates: int = 0):
        self.ema = copy.deepcopy(self._bare(model)).eval()
        for p in self.ema.parameters():
            p.requires_grad_(False)
        self.updates = updates
        # ramp the decay in so early, fast-moving weights are not over-smoothed
        self.decay_fn = lambda x: decay * (1 - math.exp(-x / tau))

    @staticmethod
    def _bare(model: nn.Module) -> nn.Module:
        return model.module if hasattr(model, "module") else model

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        self.updates += 1
        d = self.decay_fn(self.updates)
        msd = self._bare(model).state_dict()
        for k, v in self.ema.state_dict().items():
            if v.dtype.is_floating_point:
                v.mul_(d).add_(msd[k].detach(), alpha=1 - d)
            else:
                v.copy_(msd[k])

    def state_dict(self) -> dict:
        return self.ema.state_dict()

    def load_state_dict(self, sd: dict) -> None:
        self.ema.load_state_dict(sd)
