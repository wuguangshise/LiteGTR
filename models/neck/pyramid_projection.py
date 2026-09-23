"""Project backbone C2-C5 to a common width, with an optional top-down path.

docs/DESIGN.md P2-12: pure 1x1 projection performs noticeably worse on small
objects because no cross-scale fusion happens.  ``use_fpn`` is therefore ON by
default and exposed as an ablation switch rather than being hard-coded away.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.backbone.tinynext import LayerNorm2d


class ConvBNAct(nn.Module):
    def __init__(self, cin: int, cout: int, k: int = 3, s: int = 1, groups: int = 1, act: bool = True):
        super().__init__()
        self.conv = nn.Conv2d(cin, cout, k, s, k // 2, groups=groups, bias=False)
        self.bn = nn.BatchNorm2d(cout)
        self.act = nn.SiLU(inplace=True) if act else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))


class DWSepConv(nn.Module):
    """Depthwise separable 3x3 -- the workhorse of the local CNN path."""

    def __init__(self, cin: int, cout: int):
        super().__init__()
        self.dw = ConvBNAct(cin, cin, 3, groups=cin)
        self.pw = ConvBNAct(cin, cout, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pw(self.dw(x))


class PyramidProjection(nn.Module):
    def __init__(self, in_channels: list[int], out_channels: int = 64, use_fpn: bool = True):
        super().__init__()
        self.out_channels = out_channels
        self.use_fpn = use_fpn
        self.lateral = nn.ModuleList(
            [nn.Sequential(nn.Conv2d(c, out_channels, 1, bias=False), LayerNorm2d(out_channels))
             for c in in_channels]
        )
        if use_fpn:
            # one depthwise-separable smoothing conv per level after top-down add
            self.smooth = nn.ModuleList([DWSepConv(out_channels, out_channels) for _ in in_channels])

    def forward(self, feats: list[torch.Tensor]) -> list[torch.Tensor]:
        outs = [lat(f) for lat, f in zip(self.lateral, feats)]
        if not self.use_fpn:
            return outs
        for i in range(len(outs) - 2, -1, -1):  # top-down: P5 -> P2
            outs[i] = outs[i] + F.interpolate(outs[i + 1], size=outs[i].shape[-2:], mode="nearest")
        return [s(o) for s, o in zip(self.smooth, outs)]


class LocalCNNPath(nn.Module):
    """Per-level local modelling: edges, texture, dense-object separation.

    Kept deliberately cheap on P2 (stride 4) -- at 640 input P2 is 160x160 and
    dominates the MAC budget (docs/DESIGN.md P0-2).
    """

    def __init__(self, channels: int, num_levels: int, strides: tuple[int, ...] | None = None,
                 blocks_per_level: tuple[int, ...] | None = None):
        super().__init__()
        if blocks_per_level is None:
            # Only a stride-4 level gets the cheap single-block treatment. Keying this
            # off position would silently starve P3 in the use_p2=False ablation, so
            # the no_p2 run would differ from the main model by more than just P2.
            strides = strides or tuple(4 * (2 ** i) for i in range(num_levels))
            blocks_per_level = tuple(1 if s == 4 else 2 for s in strides[:num_levels])
        assert len(blocks_per_level) == num_levels
        self.paths = nn.ModuleList(
            [nn.Sequential(*[DWSepConv(channels, channels) for _ in range(n)]) for n in blocks_per_level]
        )

    def forward(self, feats: list[torch.Tensor]) -> list[torch.Tensor]:
        return [p(f) for p, f in zip(self.paths, feats)]
