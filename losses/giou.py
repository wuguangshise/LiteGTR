"""IoU-family box losses (GIoU, CIoU) and the IoU helpers used by the assigner and the evaluator."""
from __future__ import annotations

import math

import torch
import torch.nn as nn


def bbox_iou(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    """Pairwise IoU between (N,4) and (M,4) xyxy boxes -> (N, M)."""
    area_a = (a[:, 2] - a[:, 0]).clamp(0) * (a[:, 3] - a[:, 1]).clamp(0)
    area_b = (b[:, 2] - b[:, 0]).clamp(0) * (b[:, 3] - b[:, 1]).clamp(0)
    lt = torch.max(a[:, None, :2], b[None, :, :2])
    rb = torch.min(a[:, None, 2:], b[None, :, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[..., 0] * wh[..., 1]
    return inter / (area_a[:, None] + area_b[None, :] - inter + eps)


def bbox_iou_aligned(a: torch.Tensor, b: torch.Tensor, giou: bool = False,
                     eps: float = 1e-7) -> torch.Tensor:
    """Element-wise IoU/GIoU between two (N,4) xyxy tensors."""
    area_a = (a[:, 2] - a[:, 0]).clamp(0) * (a[:, 3] - a[:, 1]).clamp(0)
    area_b = (b[:, 2] - b[:, 0]).clamp(0) * (b[:, 3] - b[:, 1]).clamp(0)
    lt = torch.max(a[:, :2], b[:, :2])
    rb = torch.min(a[:, 2:], b[:, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[:, 0] * wh[:, 1]
    union = area_a + area_b - inter + eps
    iou = inter / union
    if not giou:
        return iou
    lt_c = torch.min(a[:, :2], b[:, :2])
    rb_c = torch.max(a[:, 2:], b[:, 2:])
    wh_c = (rb_c - lt_c).clamp(min=0)
    area_c = wh_c[:, 0] * wh_c[:, 1] + eps
    return iou - (area_c - union) / area_c


class GIoULoss(nn.Module):
    def __init__(self, loss_weight: float = 2.0):
        super().__init__()
        self.loss_weight = loss_weight

    def forward(self, pred: torch.Tensor, target: torch.Tensor,
                weight: torch.Tensor | None = None,
                avg_factor: float | None = None) -> torch.Tensor:
        loss = 1.0 - bbox_iou_aligned(pred, target, giou=True)
        if weight is not None:
            loss = loss * weight
        loss = loss.sum()
        if avg_factor is not None:
            # a device tensor avoids a host sync; clamp keeps the same floor of 1
            loss = loss / (avg_factor.clamp_min(1.0) if torch.is_tensor(avg_factor)
                           else max(avg_factor, 1.0))
        return self.loss_weight * loss


def bbox_ciou_aligned(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    """Element-wise CIoU (Zheng et al., AAAI 2020) between (N,4) xyxy ``a`` (pred) and ``b`` (target).

    IoU - centre distance^2 / enclosing-box diagonal^2 - alpha * v, where v measures the
    aspect-ratio gap. Unlike GIoU it still pulls the centres together when one box
    contains the other, so it converges faster and localises tighter.
    """
    iou = bbox_iou_aligned(a, b, eps=eps)
    wa = (a[:, 2] - a[:, 0]).clamp(min=0)
    ha = (a[:, 3] - a[:, 1]).clamp(min=0)
    wb = (b[:, 2] - b[:, 0]).clamp(min=0)
    hb = (b[:, 3] - b[:, 1]).clamp(min=0)
    lt_c = torch.min(a[:, :2], b[:, :2])
    rb_c = torch.max(a[:, 2:], b[:, 2:])
    c2 = ((rb_c - lt_c) ** 2).sum(-1) + eps                     # enclosing diagonal^2
    rho2 = (((a[:, :2] + a[:, 2:]) - (b[:, :2] + b[:, 2:])) ** 2).sum(-1) / 4
    v = (4 / math.pi ** 2) * (torch.atan(wb / (hb + eps)) - torch.atan(wa / (ha + eps))) ** 2
    with torch.no_grad():                                        # alpha is a trade-off weight, not a target
        alpha = v / (v - iou + (1 + eps))
    return iou - rho2 / c2 - alpha * v


class CIoULoss(nn.Module):
    def __init__(self, loss_weight: float = 1.0):
        super().__init__()
        self.loss_weight = loss_weight

    def forward(self, pred: torch.Tensor, target: torch.Tensor,
                weight: torch.Tensor | None = None,
                avg_factor: float | None = None) -> torch.Tensor:
        loss = 1.0 - bbox_ciou_aligned(pred, target)
        if weight is not None:
            loss = loss * weight
        loss = loss.sum()
        if avg_factor is not None:
            loss = loss / (avg_factor.clamp_min(1.0) if torch.is_tensor(avg_factor)
                           else max(avg_factor, 1.0))
        return self.loss_weight * loss
