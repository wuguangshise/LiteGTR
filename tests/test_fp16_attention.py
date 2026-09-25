"""Attention logits must not overflow under fp16 autocast.

The first run with the conv stem went NaN in epoch 1 and never recovered. Cause:
the geometric writeback computed q.k as a half matmul -- q from an un-normalised
feature map, k from the un-normalised mixer output. On VisDrone the raw product
passed fp16's 65504 within the first epoch; softmax(inf) is NaN, the NaN reached
the head, and the AMP loss scale collapsed to zero (tests/test_trainer_nonfinite.py
covers what followed). Inputs of magnitude ~100 reproduce the overflow on CPU.
"""
import pytest

torch = pytest.importorskip("torch")

from models.token.geometric_writeback import GeometricWriteback  # noqa: E402
from models.token.token_mixer import _Attention  # noqa: E402


def _fp16_autocast():
    return torch.autocast(device_type="cpu", dtype=torch.float16)


@pytest.mark.parametrize("mode", ["geometric", "content"])
def test_writeback_logits_do_not_overflow_in_fp16(mode):
    torch.manual_seed(0)
    wb = GeometricWriteback(64, mode=mode)
    with torch.no_grad():
        wb.gamma.fill_(1.0)
        for w in (wb.q.weight, wb.k.weight):
            w.fill_(0.05)
    feat = torch.full((1, 64, 8, 8), 100.0)
    tokens = torch.full((1, 6, 64), 100.0)
    coords = torch.rand(1, 6, 2)
    raw = (wb.q(feat).flatten(2).transpose(1, 2) @ wb.k(tokens).transpose(1, 2)).abs().max()
    assert raw > 65504, "inputs must overflow a half matmul, or the test proves nothing"
    with _fp16_autocast():
        out = wb(feat, tokens, coords)
    assert torch.isfinite(out).all()


def test_mixer_logits_do_not_overflow_in_fp16():
    attn = _Attention(64, num_heads=1)
    with torch.no_grad():
        attn.qkv.weight.fill_(0.05)
    x = torch.full((1, 6, 64), 100.0)
    with _fp16_autocast():
        out = attn(x)
    assert torch.isfinite(out).all()
