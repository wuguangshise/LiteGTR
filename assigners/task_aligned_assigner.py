"""Task-Aligned Assigner (TOOD).

Alignment metric ``t = s^alpha * IoU^beta`` picks positives that are good at
BOTH classification and localisation, which matters more than usual on VisDrone
where neighbouring small objects otherwise steal each other's anchors.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from losses.giou import bbox_iou


class TaskAlignedAssigner(nn.Module):
    def __init__(self, topk: int = 13, alpha: float = 1.0, beta: float = 6.0, eps: float = 1e-9,
                 enforce_reg_range: bool = True):
        super().__init__()
        self.topk = topk
        self.alpha = alpha
        self.beta = beta
        self.eps = eps
        self.enforce_reg_range = enforce_reg_range

    @torch.no_grad()
    def forward(self, pred_scores: torch.Tensor, pred_boxes: torch.Tensor,
                points: torch.Tensor, gt_boxes: torch.Tensor,
                gt_labels: torch.Tensor, point_strides: torch.Tensor | None = None,
                reg_max: int | None = None) -> dict:
        """Single image.

        pred_scores (L, C) sigmoid probabilities; pred_boxes (L, 4) xyxy;
        points (L, 2); gt_boxes (G, 4) xyxy; gt_labels (G,).

        ``point_strides`` and ``reg_max`` enable the regression-range constraint:
        DFL can only express a distance up to ``reg_max`` stride units, so a
        stride-4 point can reach just ``reg_max * 4`` px. Without this, a large
        object can be assigned to P2, its DFL target gets clamped to something it
        never asked for, and GIoU then pulls against DFL for the rest of training.
        """
        num_pts = points.shape[0]
        num_gt = gt_boxes.shape[0]
        device = points.device
        if num_gt == 0:
            return {
                "fg_mask": torch.zeros(num_pts, dtype=torch.bool, device=device),
                "assigned_gt": torch.zeros(num_pts, dtype=torch.long, device=device),
                "assigned_labels": torch.full((num_pts,), -1, dtype=torch.long, device=device),
                "assigned_ious": torch.zeros(num_pts, device=device),
            }

        # candidates must lie inside the GT box
        lt = points[:, None, :] - gt_boxes[None, :, :2]
        rb = gt_boxes[None, :, 2:] - points[:, None, :]
        ltrb = torch.cat([lt, rb], -1)                               # (L, G, 4)
        inside = ltrb.min(-1).values > 0                             # (L, G)

        if self.enforce_reg_range and point_strides is not None and reg_max is not None:
            in_range = (ltrb / point_strides[:, None, None]).max(-1).values <= reg_max
            inside = inside & in_range

        ious = bbox_iou(pred_boxes, gt_boxes)                        # (L, G)
        scores = pred_scores[:, gt_labels]                           # (L, G)
        align = scores.clamp_min(self.eps).pow(self.alpha) * ious.clamp_min(0).pow(self.beta)
        align = align * inside

        topk = min(self.topk, num_pts)
        _, cand = align.topk(topk, dim=0)                            # (topk, G)
        cand_mask = torch.zeros_like(align, dtype=torch.bool)
        cand_mask.scatter_(0, cand, True)
        mask = cand_mask & inside & (align > 0)

        # resolve a point claimed by several GTs -> highest IoU among the CLAIMING gts.
        # Using the raw argmax over all GTs would be wrong: an unclaimed GT with a
        # higher IoU would steal the point, producing a positive the assigner never
        # selected. Dense UAV scenes hit this constantly.
        overlap = mask.sum(1)
        if (overlap > 1).any():
            multi = overlap > 1
            best = (ious * mask).argmax(1)
            fix = torch.zeros_like(mask)
            fix[torch.arange(num_pts, device=device), best] = True
            mask = torch.where(multi[:, None], fix & mask, mask)

        fg = mask.any(1)
        assigned_gt = mask.float().argmax(1)
        labels = torch.full((num_pts,), -1, dtype=torch.long, device=device)
        labels[fg] = gt_labels[assigned_gt[fg]]

        # normalised alignment -> soft QFL target (TOOD eq. 8)
        align_pos = align * mask
        max_align = align_pos.max(0, keepdim=True).values             # (1, G)
        max_iou = (ious * mask).max(0, keepdim=True).values
        norm_align = align_pos / (max_align + self.eps) * max_iou
        assigned_ious = norm_align.max(1).values

        return {
            "fg_mask": fg,
            "assigned_gt": assigned_gt,
            "assigned_labels": labels,
            "assigned_ious": assigned_ious,
        }
