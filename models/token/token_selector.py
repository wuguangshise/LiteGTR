"""Fixed-budget token selection with LOCAL candidate routing.

Design constraints (docs/DESIGN.md P0-3, P1-8):

* **Fixed budget, static shape.** ``k`` per region is a compile-time constant and
  no dynamic control flow is used, so the ONNX graph exports with static shapes
  and TensorRT does not fall back.  This is the deployment argument for a fixed
  token budget -- not merely "it saves FLOPs".
* **Local routing.** Instead of a global top-k over all H*W positions (which
  collapses onto a few salient blobs and starves dense regions), the map is cut
  into a GxG grid and the top-k of *each region* is kept.  Tokens therefore stay
  spatially distributed, which matters on VisDrone where a single image holds
  ~53 objects on average and 300+ in dense scenes.
* **Budget is a config value, never a constant.** ``tools/train.py`` sweeps it
  via ``configs/ablation/token_budget_*.yaml`` before any main experiment.

Returns both token features and their *geometry* (normalised centre + level),
which ``geometric_writeback`` consumes.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class TokenScorer(nn.Module):
    """Lightweight importance scorer. Shared shape across levels, separate weights.

    The raw score map is also returned for the EMA consistency loss -- the
    teacher/student alignment happens on these CONTINUOUS scores, never on the
    discrete top-k result (docs/DESIGN.md P0-4).
    """

    def __init__(self, channels: int, hidden: int | None = None):
        super().__init__()
        hidden = hidden or max(channels // 2, 16)
        self.conv = nn.Sequential(
            nn.Conv2d(channels, hidden, 3, padding=1, groups=hidden if channels == hidden else 1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden, 1, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)  # (B, 1, H, W)


class LocalTopKSelector(nn.Module):
    """Select ``grid**2 * k_per_region`` tokens from one feature level.

    Score gating -- why the scorer needs it
    ---------------------------------------
    ``top-k`` contributes only *indices*, and indices are discrete. Without a
    gate the selected tokens are plain ``gather``s of the feature map, so the
    detection loss sends gradient into the features and **none into the scorer**.
    The routing would then never learn what is worth selecting; its only training
    signal would be whatever auxiliary loss happens to touch the score map.

    With ``score_gate`` each selected token is scaled by ``sigmoid(score)``. The
    detection loss can now raise the score of tokens that help and lower the
    score of those that do not. ``top-k`` itself stays a pure forward op with a
    static ``k``, so the exported graph is unchanged apart from one sigmoid and a
    multiply. ``score_gate=False`` is kept as the random-routing ablation.
    """

    def __init__(self, channels: int, num_tokens: int, grid: int, level_id: int,
                 score_gate: bool = True):
        super().__init__()
        assert num_tokens % (grid * grid) == 0, (
            f"num_tokens={num_tokens} must be divisible by grid^2={grid * grid}"
        )
        self.num_tokens = num_tokens
        self.grid = grid
        self.k = num_tokens // (grid * grid)
        self.level_id = level_id
        self.score_gate = score_gate
        self.scorer = TokenScorer(channels)

    def forward(self, x: torch.Tensor) -> dict:
        b, c, h, w = x.shape
        g = self.grid
        ph, pw = (-h) % g, (-w) % g
        if ph or pw:  # keep region sizes uniform; padded scores are masked out
            x = F.pad(x, (0, pw, 0, ph))
        hh, ww = x.shape[-2:]
        rh, rw = hh // g, ww // g

        score = self.scorer(x)                                     # (B,1,hh,ww)
        if ph or pw:
            mask = torch.zeros(1, 1, hh, ww, device=x.device, dtype=x.dtype)
            mask[..., :h, :w] = 1.0
            score = score.masked_fill(mask == 0, float("-inf"))

        # (B, C, hh, ww) -> (B, G*G, rh*rw, C)
        def to_regions(t: torch.Tensor) -> torch.Tensor:
            bb, cc = t.shape[0], t.shape[1]
            t = t.view(bb, cc, g, rh, g, rw).permute(0, 2, 4, 3, 5, 1)
            return t.reshape(bb, g * g, rh * rw, cc)

        feat_r = to_regions(x)                                     # (B, G^2, rh*rw, C)
        score_r = to_regions(score).squeeze(-1)                    # (B, G^2, rh*rw)

        topv, topi = score_r.topk(self.k, dim=-1)                  # (B, G^2, k) -- static k
        idx = topi.unsqueeze(-1).expand(-1, -1, -1, c)
        tokens = feat_r.gather(2, idx).reshape(b, self.num_tokens, c)
        scores = topv.reshape(b, self.num_tokens)
        if self.score_gate:
            # the gradient path into the scorer -- see the class docstring
            tokens = tokens * torch.sigmoid(scores).unsqueeze(-1)

        # --- geometry: normalised centre of each selected position, in [0,1] ---
        flat = topi                                                # index inside region
        iy = flat // rw
        ix = flat % rw
        ridx = torch.arange(g * g, device=x.device)
        ry, rx = (ridx // g).view(1, -1, 1), (ridx % g).view(1, -1, 1)
        abs_y = (ry * rh + iy).float() + 0.5
        abs_x = (rx * rw + ix).float() + 0.5
        coords = torch.stack([abs_x / ww, abs_y / hh], dim=-1)     # (B, G^2, k, 2)
        coords = coords.reshape(b, self.num_tokens, 2)

        return {
            "tokens": tokens,            # (B, N, C)
            "coords": coords,            # (B, N, 2) normalised (x, y)
            "scores": scores,            # (B, N)
            "score_map": score[..., :h, :w],  # (B,1,H,W) for the EMA loss
            "level": self.level_id,
        }


class MultiLevelTokenSelector(nn.Module):
    """Runs one :class:`LocalTopKSelector` per participating level and concatenates."""

    def __init__(self, channels: int, token_budget: dict[str, int], grids: dict[str, int],
                 levels: list[str], score_gate: bool = True):
        super().__init__()
        self.levels = levels
        self.selectors = nn.ModuleDict(
            {lv: LocalTopKSelector(channels, token_budget[lv], grids[lv], i, score_gate=score_gate)
             for i, lv in enumerate(levels)}
        )
        self.num_tokens = sum(token_budget[lv] for lv in levels)

    def forward(self, feats: dict[str, torch.Tensor]) -> dict:
        outs = [self.selectors[lv](feats[lv]) for lv in self.levels]
        return {
            "tokens": torch.cat([o["tokens"] for o in outs], dim=1),
            "coords": torch.cat([o["coords"] for o in outs], dim=1),
            "scores": torch.cat([o["scores"] for o in outs], dim=1),
            "level_ids": torch.cat(
                [torch.full((o["tokens"].shape[1],), float(o["level"]), device=o["tokens"].device)
                 for o in outs]
            ),
            "score_maps": {lv: o["score_map"] for lv, o in zip(self.levels, outs)},
        }
