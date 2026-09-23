"""Foreground supervision for the token scorer.

Why the scorer needs it
-----------------------
Without this term the scorer's only training signals are

* the ``sigmoid(score)`` gate on the selected tokens -- a per-token scale that
  the mixer and write-back absorb, so it carries almost no information about
  WHERE the budget should go; and
* the EMA consistency KL, whose minimum is reached by any pair of identical
  maps -- including two flat ones.

Nothing opposes a flat map, and weight decay pulls the scorer toward it. The
first 200-epoch run showed exactly that: ``score_entropy`` rose monotonically to
0.999997 (score std ~0.01 over the map), ``ema_agreement`` read 0.999 because
both maps were flat, and ``loss_token`` sat at ~4e-6. Top-k over a flat map is
random routing.

The fix: tell the scorer where the objects are. Each level's score map is
supervised with a Gaussian centre heatmap built from the ground-truth boxes
(CenterNet's penalty-reduced focal loss). Every level sees every object -- a
tiny object on P5 simply gets the minimum radius -- so selection is driven by
object density, which on VisDrone means small objects. Once the maps are
peaked, the EMA consistency term has something non-trivial to align.

Training-only: nothing here is exported, the deployed graph is unchanged.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def center_heatmap(boxes: torch.Tensor, h: int, w: int, stride: float,
                   sigma_ratio: float = 1 / 6, sigma_min: float = 0.5) -> torch.Tensor:
    """Gaussian centre heatmap on an ``h x w`` grid of cells ``stride`` pixels wide.

    ``boxes`` are ``(N, 4)`` xyxy in input-image pixels. Cell ``(i, j)`` has its
    centre at ``((j + 0.5) * stride, (i + 0.5) * stride)`` -- the same convention
    the selector uses for token coordinates. The per-axis sigma is
    ``sigma_ratio`` of the box extent in cells, floored at ``sigma_min`` cells;
    overlapping objects take the element-wise max. The cell containing each
    centre is set to exactly 1: those are the focal loss's positives.
    """
    heat = torch.zeros(h, w, device=boxes.device, dtype=torch.float32)
    if boxes.numel() == 0:
        return heat
    b = boxes.float() / stride
    cx, cy = (b[:, 0] + b[:, 2]) * 0.5, (b[:, 1] + b[:, 3]) * 0.5
    sx = ((b[:, 2] - b[:, 0]) * sigma_ratio).clamp_min(sigma_min)
    sy = ((b[:, 3] - b[:, 1]) * sigma_ratio).clamp_min(sigma_min)
    xs = torch.arange(w, device=boxes.device, dtype=torch.float32) + 0.5
    ys = torch.arange(h, device=boxes.device, dtype=torch.float32) + 0.5
    gx = torch.exp(-((xs[None, :] - cx[:, None]) ** 2) / (2 * sx[:, None] ** 2))   # (N, W)
    gy = torch.exp(-((ys[None, :] - cy[:, None]) ** 2) / (2 * sy[:, None] ** 2))   # (N, H)
    heat = (gy[:, :, None] * gx[:, None, :]).amax(0)                               # (H, W)
    ix = cx.floor().long().clamp(0, w - 1)
    iy = cy.floor().long().clamp(0, h - 1)
    heat[iy, ix] = 1.0
    return heat


class TokenRoutingLoss(nn.Module):
    """Penalty-reduced focal loss (CenterNet) between score maps and GT centres.

    The score logits pass through a sigmoid -- the same sigmoid the score gate
    applies -- so a high score means "an object is here" in both places.
    Normalised by the number of object centres per level, averaged over levels.
    """

    def __init__(self, loss_weight: float = 0.5, alpha: float = 2.0, beta: float = 4.0,
                 sigma_ratio: float = 1 / 6, sigma_min: float = 0.5):
        super().__init__()
        self.loss_weight = loss_weight
        self.alpha = alpha
        self.beta = beta
        self.sigma_ratio = sigma_ratio
        self.sigma_min = sigma_min

    def forward(self, score_maps: dict[str, torch.Tensor], gt_boxes: list[torch.Tensor],
                strides: dict[str, float]) -> torch.Tensor:
        losses = []
        for lv, s in score_maps.items():
            # fp32: log-sigmoid on half-precision logits loses the negatives' tail
            s = s.float().squeeze(1)                                    # (B, H, W)
            b, h, w = s.shape
            heat = torch.stack([center_heatmap(gt_boxes[i], h, w, strides[lv],
                                               self.sigma_ratio, self.sigma_min)
                                for i in range(b)])
            pos = heat.eq(1.0)
            log_p, log_np = F.logsigmoid(s), F.logsigmoid(-s)
            p = s.sigmoid()
            pos_loss = -((1 - p) ** self.alpha) * log_p
            neg_loss = -((1 - heat) ** self.beta) * (p ** self.alpha) * log_np
            num_pos = pos.sum().clamp_min(1).float()
            losses.append((torch.where(pos, pos_loss, neg_loss)).sum() / num_pos)
        if not losses:
            return torch.zeros((), device=gt_boxes[0].device if gt_boxes else "cpu")
        return self.loss_weight * torch.stack(losses).mean()
