"""Analytic parameter/MAC budget calculator -- pure Python, no torch required.

Used by ``tools/profile_model.py --search`` to sweep backbone configurations without
instantiating models.  Real (authoritative) numbers come from the torch path in
``tools/profile_model.py``; this module exists so that config search is instant and
runnable in environments without torch.

See docs/DESIGN.md section "P0-1 / P0-2 parameter budget".
"""
from __future__ import annotations


def convnext_block_params(c: int) -> int:
    """7x7 depthwise + LayerNorm + pointwise c->4c->c + layer-scale gamma."""
    dw = 49 * c + c              # depthwise conv weight + bias
    ln = 2 * c                   # LayerNorm weight + bias
    pw1 = 4 * c * c + 4 * c      # c -> 4c
    pw2 = 4 * c * c + c          # 4c -> c
    gamma = c                    # layer scale
    return dw + ln + pw1 + pw2 + gamma


def backbone_params(channels: list[int], depths: list[int], in_ch: int = 3) -> dict:
    """Parameters of a TinyNeXt-style backbone, broken down per stage."""
    assert len(channels) == len(depths) == 4
    stem = in_ch * channels[0] * 16 + channels[0] + 2 * channels[0]  # 4x4 s4 conv + LN
    total = stem
    per_stage, downsamples = [], []
    for i, (c, d) in enumerate(zip(channels, depths)):
        if i > 0:
            ds = 2 * channels[i - 1] + channels[i - 1] * c * 4 + c   # LN + 2x2 s2 conv
            downsamples.append(ds)
            total += ds
        blocks = d * convnext_block_params(c)
        per_stage.append(blocks)
        total += blocks
    return {
        "total": total,
        "stem": stem,
        "per_stage": per_stage,
        "downsamples": downsamples,
    }


def conv_macs(h: int, w: int, cin: int, cout: int, k: int = 1, groups: int = 1) -> int:
    return h * w * (cin // groups) * cout * k * k


def backbone_macs(channels: list[int], depths: list[int], img: int = 640) -> int:
    """MACs of the backbone at a given square input resolution."""
    macs = conv_macs(img // 4, img // 4, 3, channels[0], 4)
    for i, (c, d) in enumerate(zip(channels, depths)):
        s = 4 * (2 ** i)
        h = w = img // s
        if i > 0:
            macs += conv_macs(h, w, channels[i - 1], c, 2)
        per_block = (
            conv_macs(h, w, c, c, 7, groups=c)   # depthwise
            + conv_macs(h, w, c, 4 * c)          # pointwise expand
            + conv_macs(h, w, 4 * c, c)          # pointwise project
        )
        macs += d * per_block
    return macs


PRESETS = {
    # name: (channels, depths)
    "tinynext_m_orig": ([32, 64, 128, 256], [4, 4, 9, 4]),   # original plan (over budget)
    "tinynext_m": ([32, 64, 128, 192], [2, 4, 8, 2]),        # LOCKED: Main
    "tinynext_s": ([24, 48, 96, 160], [2, 3, 6, 2]),         # LOCKED: Edge-S
    "tinynext_a": ([24, 48, 96, 192], [2, 4, 8, 3]),         # alternative
}
