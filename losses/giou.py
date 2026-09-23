"""GIoU loss and the IoU helpers used by the assigner and the evaluator."""
from __future__ import annotations

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
