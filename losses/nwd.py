"""Normalized Wasserstein Distance loss (Wang et al., "A Normalized Gaussian
Wasserstein Distance for Tiny Object Detection", arXiv 2110.13389 / ISPRS 2022).

Each box is modelled as a 2-D Gaussian (centre = mean, half extents = std). The
2nd-order Wasserstein distance between two such Gaussians is

    W2^2 = (cx1 - cx2)^2 + (cy1 - cy2)^2 + ((w1 - w2)^2 + (h1 - h2)^2) / 4

and NWD = exp(-sqrt(W2^2) / C). Unlike IoU, a 2 px offset costs a 6 px box and a
60 px box the same, and non-overlapping boxes still get a gradient -- the two
failure modes of GIoU on tiny objects. ``C`` sets the pixel scale (the paper's
12.8 is AI-TOD's mean object size); boxes are in input-image pixels here.
"""
from __future__ import annotations

import torch
import torch.nn as nn


def nwd_aligned(a: torch.Tensor, b: torch.Tensor, constant: float = 12.8,
                eps: float = 1e-7) -> torch.Tensor:
    """Element-wise NWD between two (N,4) xyxy tensors -> (N,) in (0, 1]."""
    ca = (a[:, :2] + a[:, 2:]) / 2
    cb = (b[:, :2] + b[:, 2:]) / 2
    wha = (a[:, 2:] - a[:, :2]).clamp(min=0)
    whb = (b[:, 2:] - b[:, :2]).clamp(min=0)
    w2 = ((ca - cb) ** 2).sum(-1) + ((wha - whb) ** 2).sum(-1) / 4
    return torch.exp(-torch.sqrt(w2 + eps) / constant)


class NWDLoss(nn.Module):
    def __init__(self, constant: float = 12.8, loss_weight: float = 1.0):
        super().__init__()
        self.constant = constant
        self.loss_weight = loss_weight

    def forward(self, pred: torch.Tensor, target: torch.Tensor,
                weight: torch.Tensor | None = None,
                avg_factor: float | None = None) -> torch.Tensor:
        loss = 1.0 - nwd_aligned(pred, target, self.constant)
        if weight is not None:
            loss = loss * weight
        loss = loss.sum()
        if avg_factor is not None:
            loss = loss / (avg_factor.clamp_min(1.0) if torch.is_tensor(avg_factor)
                           else max(avg_factor, 1.0))
        return self.loss_weight * loss
