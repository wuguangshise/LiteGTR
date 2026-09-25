"""Routed detail enhancement: sharpen high-frequency detail only where objects are.

The routing score maps are supervised as GT-centre heatmaps (losses/token_routing.py),
so ``sigmoid(score)`` is a class-agnostic "an object is here" map. The same routing
that decides where global context is exchanged is reused to decide where local
detail is amplified:

    HF = F - AvgPool3x3(F)                       high-pass: edges and fine texture
    M  = MaxPool_d(Up(sigmoid(S_src)))           where objects are, dilated to their borders
    F' = F + alpha * M * HF                      alpha per channel, initialised to 0

Applied to P2 (stride 4) by default, where 2-8 px objects are detected. It runs
after the token path and before the head -- P2 only feeds the head -- so it needs
no second pass. Cost: ``dim`` parameters per level, a 3x3 average pool and a few
element-wise ops. ``alpha`` starts at 0, so the module is an identity at
initialisation and stays near it if sharpening does not help.

``mask`` selects where the enhancement applies (the ablations):
    score   soft mask from the routing score map (the design)
    global  everywhere -- plain high-frequency boosting, no routing
    token   hard mask at the selected tokens only
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

MASKS = ("score", "global", "token")


class RoutedDetailEnhance(nn.Module):
    def __init__(self, dim: int, levels: list[str], mask: str = "score",
                 source: str = "P3", dilate: int = 3):
        super().__init__()
        if mask not in MASKS:
            raise ValueError(f"unknown detail_enhance mask: {mask!r} (expected one of {MASKS})")
        assert dilate % 2 == 1, "dilate must be odd so the mask stays centred"
        self.levels = list(levels)
        self.mask = mask
        self.source = source
        self.dilate = dilate
        self.alpha = nn.ParameterDict({lv: nn.Parameter(torch.zeros(dim)) for lv in self.levels})

    @staticmethod
    def high_pass(x: torch.Tensor) -> torch.Tensor:
        return x - F.avg_pool2d(x, 3, stride=1, padding=1, count_include_pad=False)

    def _dilate(self, m: torch.Tensor) -> torch.Tensor:
        if self.dilate <= 1:
            return m
        return F.max_pool2d(m, self.dilate, stride=1, padding=self.dilate // 2)

    def _score_mask(self, score_maps: dict[str, torch.Tensor], size: tuple[int, int]) -> torch.Tensor:
        s = torch.sigmoid(score_maps[self.source])
        if tuple(s.shape[-2:]) != tuple(size):
            s = F.interpolate(s, size=size, mode="nearest")
        return self._dilate(s)

    def _token_mask(self, coords: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
        """1 at the target-level cell of every selected token, 0 elsewhere. ``coords``
        are normalised (x, y) centres; the token count is static, so is the graph."""
        h, w = size
        ix = (coords[..., 0] * w).long().clamp(0, w - 1)
        iy = (coords[..., 1] * h).long().clamp(0, h - 1)
        m = coords.new_zeros(coords.shape[0], h * w)
        m.scatter_(1, iy * w + ix, 1.0)
        # a token stands for one SOURCE-level cell, i.e. stride_src / stride_tgt target cells;
        # the dilation covers that footprint plus the object border
        return self._dilate(m.view(-1, 1, h, w))

    def forward(self, fmap: dict[str, torch.Tensor], score_maps: dict[str, torch.Tensor] | None,
                coords: torch.Tensor | None) -> dict[str, torch.Tensor]:
        out = dict(fmap)
        for lv in self.levels:
            f = fmap[lv]
            size = tuple(f.shape[-2:])
            if self.mask == "global":
                m = None
            elif self.mask == "score":
                m = self._score_mask(score_maps, size).to(f.dtype)
            else:
                m = self._token_mask(coords, size).to(f.dtype)
            hf = self.high_pass(f)
            if m is not None:
                hf = hf * m
            out[lv] = f + self.alpha[lv].view(1, -1, 1, 1).to(f.dtype) * hf
        return out
