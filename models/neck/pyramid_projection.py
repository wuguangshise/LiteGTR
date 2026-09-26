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
    """``p2_fusion`` sets how the top-down path merges the upsampled P3 into P2:

    * ``add``      ``lat(C2) + up(P3)`` -- the plain FPN sum: the coarse P3, one value
                   over a whole tiny object, lands on C2's detail with the same weight.
    * ``weighted`` fast normalised fusion (EfficientDet / BiFPN, CVPR 2020), one pair of
                   weights per channel: ``(w1 * lat(C2) + w2 * up(P3)) / (w1 + w2 + eps)``,
                   ``w = relu(.)``, both initialised to 1. Each channel learns how much
                   P3 context it takes instead of taking all of it.
    """

    def __init__(self, in_channels: list[int], out_channels: int = 64, use_fpn: bool = True,
                 p2_fusion: str = "add"):
        super().__init__()
        if p2_fusion not in ("add", "weighted"):
            raise ValueError(f"unknown p2_fusion {p2_fusion!r} (expected 'add' or 'weighted')")
        if p2_fusion != "add" and not use_fpn:
            raise ValueError("p2_fusion needs the top-down path (use_fpn: true)")
        self.out_channels = out_channels
        self.use_fpn = use_fpn
        self.p2_fusion = p2_fusion
        if p2_fusion == "weighted":
            # two 1-D vectors, not one (2, C) tensor: 1-D parameters are exempt from weight
            # decay (engine/trainer.py), which would otherwise pull both weights to eps
            self.p2_w_lat = nn.Parameter(torch.ones(out_channels))
            self.p2_w_up = nn.Parameter(torch.ones(out_channels))
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
            up = F.interpolate(outs[i + 1], size=outs[i].shape[-2:], mode="nearest")
            if i == 0 and self.p2_fusion == "weighted":
                wl = F.relu(self.p2_w_lat).view(1, -1, 1, 1)
                wu = F.relu(self.p2_w_up).view(1, -1, 1, 1)
                outs[i] = (wl * outs[i] + wu * up) / (wl + wu + 1e-4)
            else:
                outs[i] = outs[i] + up
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
