"""Distribution Focal Loss -- linear interpolation between the two neighbouring bins."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class DistributionFocalLoss(nn.Module):
    def __init__(self, loss_weight: float = 0.25):
        super().__init__()
        self.loss_weight = loss_weight

    def forward(self, pred: torch.Tensor, target: torch.Tensor,
                weight: torch.Tensor | None = None,
                avg_factor: float | None = None) -> torch.Tensor:
        """pred: (N, reg_max+1) logits for ONE side; target: (N,) continuous in [0, reg_max]."""
        dl = target.floor().long()
        dr = dl + 1
        wl = dr.float() - target
        wr = 1.0 - wl
        loss = (F.cross_entropy(pred, dl, reduction="none") * wl
                + F.cross_entropy(pred, dr.clamp(max=pred.shape[-1] - 1), reduction="none") * wr)
        if weight is not None:
            loss = loss * weight
        loss = loss.sum()
        if avg_factor is not None:
            # a device tensor avoids a host sync; clamp keeps the same floor of 1
            loss = loss / (avg_factor.clamp_min(1.0) if torch.is_tensor(avg_factor)
                           else max(avg_factor, 1.0))
        return self.loss_weight * loss
