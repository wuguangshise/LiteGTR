"""YOLO-class CSP baseline, deliberately sharing LiteGTR's head, losses,
assigner, augmentation and schedule.

Why this file exists (docs/DESIGN.md P1-6): the paper's fairness argument is a
UNIFIED PROTOCOL -- every method retrained under identical settings -- not a ban
on pretraining. That argument is only credible if the baselines are actually
trained here rather than quoted from other papers, whose numbers come with COCO
pretraining, different resolutions and different NMS settings.

Everything except the feature extractor is held constant, so a difference in the
main table is attributable to the backbone/neck/token design and nothing else.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from models.neck.pyramid_projection import ConvBNAct


class Bottleneck(nn.Module):
    def __init__(self, c: int, shortcut: bool = True, expansion: float = 0.5):
        super().__init__()
        hidden = int(c * expansion)
        self.cv1 = ConvBNAct(c, hidden, 1)
        self.cv2 = ConvBNAct(hidden, c, 3)
        self.add = shortcut

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.cv2(self.cv1(x))
        return x + y if self.add else y


class C2f(nn.Module):
    """CSP block with two splits and ``n`` bottlenecks (YOLOv8-style)."""

    def __init__(self, cin: int, cout: int, n: int = 1, shortcut: bool = True):
        super().__init__()
        self.c = cout // 2
        self.cv1 = ConvBNAct(cin, 2 * self.c, 1)
        self.cv2 = ConvBNAct((2 + n) * self.c, cout, 1)
        self.m = nn.ModuleList(Bottleneck(self.c, shortcut) for _ in range(n))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = list(self.cv1(x).chunk(2, 1))
        for m in self.m:
            y.append(m(y[-1]))
        return self.cv2(torch.cat(y, 1))


class SPPF(nn.Module):
    def __init__(self, cin: int, cout: int, k: int = 5):
        super().__init__()
        hidden = cin // 2
        self.cv1 = ConvBNAct(cin, hidden, 1)
        self.cv2 = ConvBNAct(hidden * 4, cout, 1)
        self.pool = nn.MaxPool2d(k, stride=1, padding=k // 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.cv1(x)
        y1 = self.pool(x)
        y2 = self.pool(y1)
        return self.cv2(torch.cat([x, y1, y2, self.pool(y2)], 1))


class CSPBackbone(nn.Module):
    """Emits C2/C3/C4/C5 at strides 4/8/16/32, matching TinyNeXt's interface."""

    def __init__(self, channels: tuple[int, ...] = (32, 64, 128, 256),
                 depths: tuple[int, ...] = (1, 2, 2, 1), in_ch: int = 3,
                 out_indices: tuple[int, ...] = (0, 1, 2, 3)):
        super().__init__()
        self.out_indices = out_indices
        self.out_channels = [channels[i] for i in out_indices]
        self.stem = nn.Sequential(ConvBNAct(in_ch, channels[0] // 2, 3, s=2),
                                  ConvBNAct(channels[0] // 2, channels[0], 3, s=2))
        self.stages = nn.ModuleList()
        self.downs = nn.ModuleList()
        for i in range(4):
            if i > 0:
                self.downs.append(ConvBNAct(channels[i - 1], channels[i], 3, s=2))
            blocks = [C2f(channels[i], channels[i], depths[i])]
            if i == 3:
                blocks.append(SPPF(channels[i], channels[i]))
            self.stages.append(nn.Sequential(*blocks))

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        outs = []
        x = self.stem(x)
        for i in range(4):
            if i > 0:
                x = self.downs[i - 1](x)
            x = self.stages[i](x)
            if i in self.out_indices:
                outs.append(x)
        return outs


def csp_n(**kw) -> CSPBackbone:
    """~YOLOv8n-scale feature extractor."""
    return CSPBackbone(channels=(32, 64, 128, 256), depths=(1, 2, 2, 1), **kw)


def csp_t(**kw) -> CSPBackbone:
    """Smaller still, for the Edge-S comparison point."""
    return CSPBackbone(channels=(24, 48, 96, 192), depths=(1, 1, 2, 1), **kw)
