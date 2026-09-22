"""Geometry-aware token writeback -- the paper's primary structural contribution.

Motivation
----------
A global token summarises a *region* of a *particular scale*.  Broadcasting it
back uniformly (or with pure content attention) destroys exactly what matters on
UAV imagery: a token describing a 12-px car must not smear across the whole
feature map, while a token describing scene context should.

Mechanism
---------
Each token ``i`` carries its geometry from the selector: a normalised source
centre ``c_i in [0,1]^2`` and its source level.  The token additionally *predicts
its own spatial extent* ``sigma_i``.  Writeback to target location ``p`` (with
normalised coordinate ``q_p``) uses an attention whose logits are the sum of a
content term and an anisotropic Gaussian geometric prior::

    C(p, i) = <W_q f_p, W_k t_i> / sqrt(D)
    G(p, i) = -0.5 * [ (qx_p - cx_i)^2 / sx_i^2 + (qy_p - cy_i)^2 / sy_i^2 ]
    A(p, .) = softmax_i( C(p, i) + G(p, i) )
    f'_p    = f_p + gamma * sum_i A(p, i) * W_v t_i

with ``sigma_i = softplus(W_s t_i) + sigma_min`` learnt per token and per axis,
and ``gamma`` a learnable scalar initialised to 0 so training starts from the
pure local-CNN solution and the global path is introduced gradually.

The residual form is what protects small-object detail: nothing is replaced,
only added.

Ablations
---------
``mode="geometric"``  full mechanism (default)
``mode="content"``    drop G  -- isolates the contribution of the geometric prior
``mode="broadcast"``  uniform attention -- the naive baseline
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn


class GeometricWriteback(nn.Module):
    def __init__(self, dim: int, num_heads: int = 1, sigma_min: float = 0.02,
                 mode: str = "geometric"):
        super().__init__()
        assert mode in ("geometric", "content", "broadcast")
        assert dim % num_heads == 0
        self.mode = mode
        self.h = num_heads
        self.dh = dim // num_heads
        self.scale = self.dh ** -0.5
        self.sigma_min = sigma_min

        self.q = nn.Conv2d(dim, dim, 1, bias=False)
        self.k = nn.Linear(dim, dim, bias=False)
        self.v = nn.Linear(dim, dim, bias=False)
        self.sigma = nn.Linear(dim, 2)            # per-token, per-axis extent
        self.proj = nn.Conv2d(dim, dim, 1, bias=False)
        self.gamma = nn.Parameter(torch.zeros(1))  # residual gate, starts closed
        nn.init.zeros_(self.sigma.weight)
        nn.init.constant_(self.sigma.bias, 0.0)

    @staticmethod
    def _grid(h: int, w: int, device, dtype) -> torch.Tensor:
        ys = (torch.arange(h, device=device, dtype=dtype) + 0.5) / h
        xs = (torch.arange(w, device=device, dtype=dtype) + 0.5) / w
        gy, gx = torch.meshgrid(ys, xs, indexing="ij")
        return torch.stack([gx, gy], dim=-1).reshape(h * w, 2)   # (HW, 2) as (x, y)

    def forward(self, feat: torch.Tensor, tokens: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
        """feat: (B,C,H,W) target level; tokens: (B,N,C); coords: (B,N,2) normalised (x,y)."""
        b, c, h, w = feat.shape
        n = tokens.shape[1]
        hw = h * w

        q = self.q(feat).reshape(b, self.h, self.dh, hw).permute(0, 1, 3, 2)   # (B,h,HW,dh)
        k = self.k(tokens).reshape(b, n, self.h, self.dh).permute(0, 2, 1, 3)  # (B,h,N,dh)
        v = self.v(tokens).reshape(b, n, self.h, self.dh).permute(0, 2, 1, 3)

        if self.mode == "broadcast":
            attn = feat.new_full((b, self.h, hw, n), 1.0 / n)
        else:
            logits = (q @ k.transpose(-2, -1)) * self.scale                    # (B,h,HW,N)
            if self.mode == "geometric":
                grid = self._grid(h, w, feat.device, feat.dtype)               # (HW,2)
                sigma = nn.functional.softplus(self.sigma(tokens)) + self.sigma_min  # (B,N,2)
                d = grid.view(1, hw, 1, 2) - coords.view(b, 1, n, 2)           # (B,HW,N,2)
                g = -0.5 * ((d / sigma.view(b, 1, n, 2)) ** 2).sum(-1)         # (B,HW,N)
                logits = logits + g.unsqueeze(1)
            attn = logits.softmax(dim=-1)

        out = (attn @ v)                                                       # (B,h,HW,dh)
        out = out.permute(0, 1, 3, 2).reshape(b, c, h, w)
        return feat + self.gamma * self.proj(out)


class MultiLevelWriteback(nn.Module):
    """Writes the shared token set back into each target level (separate weights)."""

    def __init__(self, dim: int, levels: list[str], num_heads: int = 1, mode: str = "geometric"):
        super().__init__()
        self.levels = levels
        self.blocks = nn.ModuleDict({lv: GeometricWriteback(dim, num_heads, mode=mode) for lv in levels})

    def forward(self, feats: dict[str, torch.Tensor], tokens: torch.Tensor,
                coords: torch.Tensor) -> dict[str, torch.Tensor]:
        out = dict(feats)
        for lv in self.levels:
            out[lv] = self.blocks[lv](feats[lv], tokens, coords)
        return out
