"""Quality Focal Loss (Generalized Focal Loss, Li et al. 2020).

Classification target for a positive is its IoU with the assigned GT, so the
score reflects localisation quality -- which is what makes NMS ranking sane on
dense small objects.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class QualityFocalLoss(nn.Module):
    def __init__(self, beta: float = 2.0, loss_weight: float = 1.0):
        super().__init__()
        self.beta = beta
        self.loss_weight = loss_weight

    def forward(self, pred: torch.Tensor, target: torch.Tensor,
                avg_factor: float | None = None) -> torch.Tensor:
        """pred: (N, C) logits; target: (N, C) soft targets in [0, 1]."""
        p = pred.sigmoid()
        scale = (p - target).abs().pow(self.beta)
        loss = F.binary_cross_entropy_with_logits(pred, target, reduction="none") * scale
        loss = loss.sum()
        if avg_factor is not None:
            loss = loss / max(avg_factor, 1.0)
        return self.loss_weight * loss
