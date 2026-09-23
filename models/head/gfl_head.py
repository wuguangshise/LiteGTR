"""GFL-style anchor-free detection head (QFL + DFL + GIoU).

P2 is deliberately cheaper than the other levels (docs/DESIGN.md P0-2): at 640
input P2 is 160x160, so every 3x3 conv there costs ~1G MACs.  P2 therefore gets
a single depthwise-separable stem conv while P3-P5 get the full stack; the final
1x1 predictors are SHARED across levels, which both saves parameters and keeps
the classification scale consistent.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn

from models.neck.pyramid_projection import DWSepConv


class Integral(nn.Module):
    """Expectation of the discrete distribution -> continuous distance (DFL)."""

    def __init__(self, reg_max: int = 16):
        super().__init__()
        self.reg_max = reg_max
        self.register_buffer("project", torch.linspace(0, reg_max, reg_max + 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (..., 4*(reg_max+1)) -> (..., 4)
        shape = x.shape[:-1]
        x = x.reshape(*shape, 4, self.reg_max + 1).softmax(-1)
        return (x * self.project).sum(-1)


class GFLHead(nn.Module):
    def __init__(
        self,
        num_classes: int,
        in_channels: int = 64,
        feat_channels: int | None = None,
        strides: tuple[int, ...] = (4, 8, 16, 32),
        stacked_convs: int = 2,
        p2_stacked_convs: int = 1,
        reg_max: int = 16,
    ):
        super().__init__()
        feat_channels = feat_channels or in_channels
        self.num_classes = num_classes
        self.strides = strides
        self.reg_max = reg_max
        self.integral = Integral(reg_max)

        self.cls_stems, self.reg_stems = nn.ModuleList(), nn.ModuleList()
        for i, _ in enumerate(strides):
            n = p2_stacked_convs if (i == 0 and strides[0] == 4) else stacked_convs
            self.cls_stems.append(nn.Sequential(*[
                DWSepConv(in_channels if j == 0 else feat_channels, feat_channels) for j in range(n)]))
            self.reg_stems.append(nn.Sequential(*[
                DWSepConv(in_channels if j == 0 else feat_channels, feat_channels) for j in range(n)]))

        # shared predictors across levels
        self.cls_pred = nn.Conv2d(feat_channels, num_classes, 1)
        self.reg_pred = nn.Conv2d(feat_channels, 4 * (reg_max + 1), 1)
        # NOTE: no per-level scale here. A FCOS-style scale multiplies a *distance*;
        # multiplying DFL *logits* instead just re-tempers the softmax, which is not
        # the intended effect and makes the distribution harder to train. Each level
        # already normalises by its own stride at decode time, so no scale is needed.

        prior = 0.01
        nn.init.constant_(self.cls_pred.bias, -math.log((1 - prior) / prior))
        nn.init.normal_(self.cls_pred.weight, std=0.01)
        nn.init.normal_(self.reg_pred.weight, std=0.01)
        nn.init.constant_(self.reg_pred.bias, 0.0)

    def forward(self, feats: list[torch.Tensor]) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        cls_scores, bbox_preds = [], []
        for i, f in enumerate(feats):
            cls_scores.append(self.cls_pred(self.cls_stems[i](f)))
            bbox_preds.append(self.reg_pred(self.reg_stems[i](f)))
        return cls_scores, bbox_preds

    # ---- decoding helpers -------------------------------------------------
    @staticmethod
    def make_points(feats: list[torch.Tensor], strides: tuple[int, ...], device, dtype):
        """Anchor-free centre points, one per location, in input-image pixels."""
        pts, strd = [], []
        for f, s in zip(feats, strides):
            h, w = f.shape[-2:]
            ys = (torch.arange(h, device=device, dtype=dtype) + 0.5) * s
            xs = (torch.arange(w, device=device, dtype=dtype) + 0.5) * s
            gy, gx = torch.meshgrid(ys, xs, indexing="ij")
            pts.append(torch.stack([gx, gy], -1).reshape(-1, 2))
            strd.append(torch.full((h * w,), float(s), device=device, dtype=dtype))
        return torch.cat(pts), torch.cat(strd)

    def decode(self, cls_scores, bbox_preds, feats):
        """Flatten levels and turn distribution logits into xyxy boxes."""
        b = cls_scores[0].shape[0]
        cls = torch.cat([c.permute(0, 2, 3, 1).reshape(b, -1, self.num_classes) for c in cls_scores], 1)
        reg = torch.cat([r.permute(0, 2, 3, 1).reshape(b, -1, 4 * (self.reg_max + 1)) for r in bbox_preds], 1)
        points, strides = self.make_points(feats, self.strides, cls.device, cls.dtype)
        dist = self.integral(reg) * strides[None, :, None]      # ltrb in pixels
        x1 = points[None, :, 0] - dist[..., 0]
        y1 = points[None, :, 1] - dist[..., 1]
        x2 = points[None, :, 0] + dist[..., 2]
        y2 = points[None, :, 1] + dist[..., 3]
        boxes = torch.stack([x1, y1, x2, y2], -1)
        return cls, reg, boxes, points, strides
