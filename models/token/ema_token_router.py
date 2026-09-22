"""EMA-guided token routing (TRAINING ONLY -- removed at inference).

The gradient question, answered explicitly (docs/DESIGN.md P0-4)
----------------------------------------------------------------
Token selection is ``top-k``: discrete and non-differentiable.  We therefore do
**not** try to backpropagate through the selection.  Instead:

* the EMA teacher produces a CONTINUOUS importance score map ``s_t``;
* the student's score map ``s_s`` is aligned to it with a KL divergence over
  spatial positions (``losses/token_consistency.py``);
* ``top-k`` stays a pure forward operation with no gradient path.

No Gumbel-Softmax, no straight-through estimator: train and inference execute
the identical discrete selection, so nothing about the exported graph changes.

The teacher is a parameter-space EMA of the student's scorers only -- not the
whole network -- which keeps the memory cost negligible.  ``teacher`` is stripped
from the state dict on export, so deployment params/FLOPs are unaffected.
"""
from __future__ import annotations

import copy

import torch
import torch.nn as nn


class EMATokenRouter(nn.Module):
    def __init__(self, selector: nn.Module, momentum: float = 0.999, warmup_iters: int = 1000):
        super().__init__()
        self.momentum = momentum
        self.warmup_iters = warmup_iters
        self.register_buffer("num_updates", torch.zeros(1, dtype=torch.long))
        self.teacher = copy.deepcopy(selector)
        for p in self.teacher.parameters():
            p.requires_grad_(False)
        self.teacher.eval()

    @torch.no_grad()
    def update(self, selector: nn.Module) -> None:
        """Call once per optimiser step. Momentum ramps in over ``warmup_iters``."""
        self.num_updates += 1
        n = int(self.num_updates.item())
        m = min(self.momentum, (1 + n) / (10 + n)) if n < self.warmup_iters else self.momentum
        for tp, sp in zip(self.teacher.parameters(), selector.parameters()):
            tp.mul_(m).add_(sp.detach(), alpha=1 - m)
        for tb, sb in zip(self.teacher.buffers(), selector.buffers()):
            tb.copy_(sb)

    @torch.no_grad()
    def teacher_score_maps(self, feats: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        self.teacher.eval()
        return self.teacher(feats)["score_maps"]

    def train(self, mode: bool = True):
        super().train(mode)
        self.teacher.eval()   # teacher is never in train mode
        return self
