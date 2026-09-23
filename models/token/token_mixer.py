"""Global relation modelling over the (small) fixed token set.

With N = 56..512 tokens, full self-attention here is cheap -- the expense in a
CNN-Transformer hybrid comes from attending over H*W positions, which the
selector has already avoided.  A single-head attention plus a small FFN is
enough; anything heavier eats the parameter budget the writeback needs.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn


class TokenMixer(nn.Module):
    def __init__(self, dim: int, num_heads: int = 1, mlp_ratio: float = 2.0,
                 num_layers: int = 1, use_level_embed: bool = True, num_levels: int = 3):
        super().__init__()
        self.layers = nn.ModuleList([_MixerLayer(dim, num_heads, mlp_ratio) for _ in range(num_layers)])
        self.level_embed = nn.Parameter(torch.zeros(num_levels, dim)) if use_level_embed else None
        self.coord_proj = nn.Linear(2, dim)  # inject absolute position of each token
        nn.init.trunc_normal_(self.coord_proj.weight, std=0.02)

    def forward(self, tokens: torch.Tensor, coords: torch.Tensor,
                level_ids: torch.Tensor | None = None) -> torch.Tensor:
        # coords stay fp32 for precision; cast to the layer's dtype so a pure-fp16
        # model does not hit a Float/Half mismatch (autocast handles AMP already)
        x = tokens + self.coord_proj(coords.to(self.coord_proj.weight.dtype))
        if self.level_embed is not None and level_ids is not None:
            x = x + self.level_embed[level_ids.long()].unsqueeze(0)
        for layer in self.layers:
            x = layer(x)
        return x


class _MixerLayer(nn.Module):
    def __init__(self, dim: int, num_heads: int, mlp_ratio: float):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = _Attention(dim, num_heads)
        self.norm2 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        return x + self.mlp(self.norm2(x))


class _Attention(nn.Module):
    def __init__(self, dim: int, num_heads: int):
        super().__init__()
        assert dim % num_heads == 0
        self.h = num_heads
        self.scale = (dim // num_heads) ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, n, d = x.shape
        qkv = self.qkv(x).reshape(b, n, 3, self.h, d // self.h).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(b, n, d)
        return self.proj(out)
