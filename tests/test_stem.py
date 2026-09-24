"""The overlapping conv stem (models/backbone/tinynext.py).

Guards what the switch from ConvNeXt's 4x4 s4 patchify stem must not change --
strides, output widths, the parameter budget, the analytic budget model -- and
that a checkpoint trained with the old stem can never be evaluated with the new
one by accident.
"""
import pytest

torch = pytest.importorskip("torch")

from engine.checkpoint import CheckpointManager  # noqa: E402
from models.backbone.tinynext import TinyNeXt  # noqa: E402
from models.build import build_model  # noqa: E402
from tests._variants import variant_cfg  # noqa: E402
from utils.budget import PRESETS, backbone_macs, backbone_params  # noqa: E402


def test_main_config_uses_conv_stem():
    model = build_model(variant_cfg("main"))
    assert model.backbone.stem_type == "conv"


@pytest.mark.parametrize("stem", ["conv", "patchify"])
def test_stem_keeps_strides_and_widths(stem):
    bb = TinyNeXt(stem=stem).eval()
    with torch.no_grad():
        outs = bb(torch.randn(1, 3, 256, 256))
    assert [o.shape[1] for o in outs] == bb.out_channels
    assert [o.shape[-1] for o in outs] == [256 // s for s in bb.out_strides]


def test_conv_stem_overlaps_neighbouring_patches():
    """One output cell of the conv stem sees pixels outside its own 4x4 patch --
    the property the patchify stem lacks."""
    for stem, overlaps in (("patchify", False), ("conv", True)):
        torch.manual_seed(0)
        bb = TinyNeXt(stem=stem).eval()
        x = torch.randn(1, 3, 32, 32, requires_grad=True)
        y = bb.stem(x)[0, :, 3, 3]                     # cell (3, 3) <- pixels 12..15
        # a random projection: the channel SUM of a LayerNorm output is constant
        (y * torch.randn_like(y)).sum().backward()
        seen = x.grad.abs().sum((0, 1)) > 0
        outside = seen.clone()
        outside[12:16, 12:16] = False
        assert bool(outside.any()) == overlaps, stem


@pytest.mark.parametrize("name", ["tinynext_m", "tinynext_s"])
@pytest.mark.parametrize("stem", ["conv", "patchify"])
def test_analytic_stem_matches_torch(name, stem):
    """The stem term is exact. (The rest of the analytic model is an approximation
    that overcounts by a few K -- it is for the config search, not for the paper.)"""
    ch, dp = PRESETS[name]
    bb = TinyNeXt(channels=tuple(ch), depths=tuple(dp), stem=stem)
    assert backbone_params(ch, dp, stem=stem)["stem"] == sum(p.numel() for p in bb.stem.parameters())


def test_conv_stem_cost_is_marginal():
    ch, dp = PRESETS["tinynext_m"]
    dp_params = backbone_params(ch, dp)["total"] - backbone_params(ch, dp, stem="patchify")["total"]
    d_macs = backbone_macs(ch, dp) - backbone_macs(ch, dp, stem="patchify")
    assert 0 < dp_params < 10_000
    assert 0 < d_macs < 0.2e9


def test_patchify_checkpoint_does_not_load_silently(tmp_path):
    """Evaluation loads with strict=False; the stems' weights differ in shape, so an
    old-stem checkpoint must still fail loudly instead of leaving a random stem."""
    old = build_model(variant_cfg("patchify_stem"))
    path = tmp_path / "old.pt"
    torch.save({"model": old.state_dict(), "model_ema": None}, path)
    with pytest.raises(RuntimeError, match="stem"):
        CheckpointManager.load(path, build_model(variant_cfg("main")))
