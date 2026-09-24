"""TinyNeXt: lightweight ConvNeXt-style backbone for UAV detection.

Channel/depth presets are LOCKED from the analytic budget sweep (see
``utils/budget.py`` and docs/DESIGN.md P0-1).  Stage 4 is deliberately shallow:
in the original plan it consumed 2.16M of a 4.1-4.8M budget while contributing
least to small-object AP.

The stem is two overlapping 3x3 stride-2 convs rather than ConvNeXt's 4x4
stride-4 patchify: a patchify stem compresses each non-overlapping 4x4 block of
raw pixels into one vector, so a 4-8 px object is cut into one or two cells --
possibly split across a patch border -- before any convolution has mixed
neighbouring pixels. ``stem="patchify"`` keeps the original for comparison.

Outputs strides 4/8/16/32 -> C2/C3/C4/C5.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class LayerNorm2d(nn.Module):
    """LayerNorm over the channel dim of an NCHW tensor.

    Implemented explicitly (rather than permute + nn.LayerNorm) so the ONNX
    graph stays static-shape friendly -- see docs/DESIGN.md P1-8.
    """

    def __init__(self, num_channels: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        return x * self.weight[:, None, None] + self.bias[:, None, None]


class ConvNeXtBlock(nn.Module):
    """7x7 depthwise -> LN -> 1x1 expand(4x) -> GELU -> 1x1 project, + layer scale."""

    def __init__(self, dim: int, drop_path: float = 0.0, layer_scale_init: float = 1e-6):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, 7, padding=3, groups=dim)
        self.norm = LayerNorm2d(dim)
        self.pwconv1 = nn.Conv2d(dim, 4 * dim, 1)
        self.act = nn.GELU()
        self.pwconv2 = nn.Conv2d(4 * dim, dim, 1, bias=False)
        self.gamma = nn.Parameter(layer_scale_init * torch.ones(dim))
        self.drop_path = DropPath(drop_path) if drop_path > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = x
        x = self.dwconv(x)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        x = x * self.gamma[:, None, None]
        return shortcut + self.drop_path(x)


class DropPath(nn.Module):
    def __init__(self, p: float):
        super().__init__()
        self.p = p

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.p == 0.0 or not self.training:
            return x
        keep = 1 - self.p
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = x.new_empty(shape).bernoulli_(keep)
        return x * mask / keep


def conv_stem(in_ch: int, out_ch: int) -> nn.Sequential:
    """3x3 s2 -> BN -> GELU -> 3x3 s2 -> LN: stride 4, overlapping receptive fields.

    The hidden width is half the output, as in the CSP baseline's stem. BN after the
    first conv folds into it at export; the final LN matches the rest of the backbone.
    """
    mid = out_ch // 2
    return nn.Sequential(
        nn.Conv2d(in_ch, mid, 3, stride=2, padding=1, bias=False),
        nn.BatchNorm2d(mid),
        nn.GELU(),
        nn.Conv2d(mid, out_ch, 3, stride=2, padding=1),
        LayerNorm2d(out_ch),
    )


STEMS = ("conv", "patchify")


class TinyNeXt(nn.Module):
    """Args mirror ``utils.budget.PRESETS`` so profile numbers stay comparable."""

    def __init__(
        self,
        channels: tuple[int, ...] = (32, 64, 128, 192),
        depths: tuple[int, ...] = (2, 4, 8, 2),
        in_ch: int = 3,
        drop_path_rate: float = 0.0,
        out_indices: tuple[int, ...] = (0, 1, 2, 3),
        stem: str = "conv",
    ):
        super().__init__()
        assert len(channels) == len(depths) == 4
        if stem not in STEMS:
            raise ValueError(f"unknown stem: {stem!r} (expected one of {STEMS})")
        self.stem_type = stem
        self.channels = list(channels)
        self.out_indices = out_indices
        self.out_channels = [channels[i] for i in out_indices]
        self.out_strides = [4 * (2 ** i) for i in out_indices]

        total = sum(depths)
        dpr = [drop_path_rate * i / max(total - 1, 1) for i in range(total)]

        if stem == "conv":
            self.stem = conv_stem(in_ch, channels[0])
        else:
            self.stem = nn.Sequential(
                nn.Conv2d(in_ch, channels[0], 4, stride=4), LayerNorm2d(channels[0])
            )
        self.downsamples = nn.ModuleList()
        self.stages = nn.ModuleList()
        cur = 0
        for i in range(4):
            if i > 0:
                self.downsamples.append(
                    nn.Sequential(
                        LayerNorm2d(channels[i - 1]),
                        nn.Conv2d(channels[i - 1], channels[i], 2, stride=2),
                    )
                )
            self.stages.append(
                nn.Sequential(*[ConvNeXtBlock(channels[i], dpr[cur + j]) for j in range(depths[i])])
            )
            cur += depths[i]
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m: nn.Module) -> None:
        if isinstance(m, nn.Conv2d):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        outs = []
        x = self.stem(x)
        for i in range(4):
            if i > 0:
                x = self.downsamples[i - 1](x)
            x = self.stages[i](x)
            if i in self.out_indices:
                outs.append(x)
        return outs


def tinynext_m(**kw) -> TinyNeXt:
    return TinyNeXt(channels=(32, 64, 128, 192), depths=(2, 4, 8, 2), **kw)


def tinynext_s(**kw) -> TinyNeXt:
    return TinyNeXt(channels=(24, 48, 96, 160), depths=(2, 3, 6, 2), **kw)
