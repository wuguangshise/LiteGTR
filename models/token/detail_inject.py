"""Routed detail injection: bring back into P2 the detail it lost, only where objects are.

tools/analyze_p2_info.py measures how much of the image detail (the high-pass behind each
stride-4 cell) each point of the P2 path still carries. Along
stem(stride 2) -> C2 -> lat(C2) -> P2 after the FPN sum -> P2 into the head the linear-probe
R^2 inside small-object boxes drops 0.99 -> 0.85 -> 0.82 -> 0.44 -> 0.34 (VisDrone val,
initialised weights): the detail is mostly lost in the stem's second stride-2 conv and in
the FPN sum, where the upsampled P3 -- one value over a whole tiny object -- is added with
the same weight as C2. Sharpening P2 itself (detail_enhance.py) amplifies what is left; it
cannot restore what is gone. So the source is taken BEFORE those losses:

    S  = stride-2 stem activation                     (before the stem's 2nd conv)
    D  = Conv1x1(SpaceToDepth2x2(S))                  lossless 2x2 -> channels, then to dim
    M  = MaxPool_d(Up(sigmoid(S_P3)))                 the routing score map (GT-centre supervised)
    P2 = P2 + gamma * M * D                           gamma per channel, initialised to 0

The routing map exists only after the neck and the selector, so the source is early and the
gate is late. The same map that decides where global context is exchanged decides where
early detail is let back in: on VisDrone the GT-centre map (M > 0.3) covers ~80% of the
small-object P2 cells with ~5% of the image, while object boxes hold 27% of the image's
high-frequency energy on 6% of its area -- a global injection would mostly add background
texture. ``mask: global`` is that control.

``gamma`` starts at 0: the module is an identity at initialisation. Cost at 640: one 1x1
conv (4*16 -> 64) at 160x160, ~0.1G MACs, ~4.2K parameters; static ONNX.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

MASKS = ("score", "global")
INJECT_AT = ("before_local", "before_head")


class RoutedDetailInject(nn.Module):
    def __init__(self, src_channels: int, dim: int, mask: str = "score", source: str = "P3",
                 dilate: int = 3, detach_mask: bool = True):
        super().__init__()
        if mask not in MASKS:
            raise ValueError(f"unknown detail_inject mask: {mask!r} (expected one of {MASKS})")
        assert dilate % 2 == 1, "dilate must be odd so the mask stays centred"
        self.mask = mask
        self.source = source
        self.dilate = dilate
        # detached: the routing map stays what its supervision makes it (where objects are),
        # the detection loss does not bend it towards "where injection helps"
        self.detach_mask = detach_mask
        self.proj = nn.Conv2d(4 * src_channels, dim, 1)
        self.gamma = nn.Parameter(torch.zeros(dim))

    def routing_mask(self, score_maps: dict[str, torch.Tensor], size: tuple[int, int]) -> torch.Tensor:
        s = score_maps[self.source]
        if self.detach_mask:
            s = s.detach()
        m = torch.sigmoid(s)
        if tuple(m.shape[-2:]) != tuple(size):
            m = F.interpolate(m, size=size, mode="nearest")
        if self.dilate > 1:
            m = F.max_pool2d(m, self.dilate, stride=1, padding=self.dilate // 2)
        return m

    def forward(self, p2: torch.Tensor, stem_s2: torch.Tensor,
                score_maps: dict[str, torch.Tensor] | None) -> torch.Tensor:
        d = self.proj(F.pixel_unshuffle(stem_s2, 2))
        if self.mask == "score":
            d = d * self.routing_mask(score_maps, tuple(p2.shape[-2:])).to(d.dtype)
        return p2 + self.gamma.view(1, -1, 1, 1).to(p2.dtype) * d.to(p2.dtype)
