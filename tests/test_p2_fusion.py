"""Weighted P3 -> P2 merge in the neck (models/neck/pyramid_projection.py, p2_fusion)."""
import pytest

torch = pytest.importorskip("torch")

from models.build import build_model  # noqa: E402
from models.neck.pyramid_projection import PyramidProjection  # noqa: E402
from tests._variants import variant_cfg  # noqa: E402
from utils.export import onnx_export  # noqa: E402

CH = [8, 16, 32, 64]


def _feats():
    torch.manual_seed(0)
    return [torch.randn(2, c, 32 // 2 ** i, 32 // 2 ** i) for i, c in enumerate(CH)]


def _pair():
    torch.manual_seed(0)
    add = PyramidProjection(CH, 16, p2_fusion="add").eval()
    wtd = PyramidProjection(CH, 16, p2_fusion="weighted").eval()
    wtd.load_state_dict(add.state_dict(), strict=False)
    return add, wtd


def test_starts_as_the_mean_and_touches_p2_only():
    """w1 = w2 = 1: P2 gets (lat + up) / 2, every coarser level is the plain FPN sum."""
    add, wtd = _pair()
    f = _feats()
    with torch.no_grad():
        a, b = add(f), wtd(f)
        lat = wtd.lateral[0](f[0])
        s = add.smooth
        # rebuild the P3 that reached P2, from the plain path
        outs = [l(x) for l, x in zip(add.lateral, f)]
        for i in range(len(outs) - 2, 0, -1):
            outs[i] = outs[i] + torch.nn.functional.interpolate(outs[i + 1], size=outs[i].shape[-2:])
        up = torch.nn.functional.interpolate(outs[1], size=lat.shape[-2:])
        want = s[0]((lat + up) / (2 + 1e-4))
    assert torch.allclose(b[0], want, atol=1e-5)
    for u, v in zip(a[1:], b[1:]):
        assert torch.equal(u, v)


def test_weights_decide_how_much_p3_reaches_p2():
    _, wtd = _pair()
    f = _feats()
    with torch.no_grad():
        wtd.p2_w_up.zero_()                       # no P3 at all
        only_lat = wtd(f)[0]
        want = wtd.smooth[0](wtd.lateral[0](f[0]) * (1 / (1 + 1e-4)))
    assert torch.allclose(only_lat, want, atol=1e-5)


def test_weights_learn_and_skip_weight_decay():
    model = build_model(variant_cfg("p2_weighted_fusion")).train()
    targets = [{"boxes": torch.tensor([[20.0, 20.0, 40.0, 44.0]]), "labels": torch.tensor([1])}]
    sum(model.loss(torch.randn(1, 3, 256, 256), targets).values()).backward()
    for p in (model.neck.p2_w_lat, model.neck.p2_w_up):
        assert p.ndim == 1 and p.grad is not None and p.grad.abs().sum() > 0


def test_bad_settings_are_rejected():
    with pytest.raises(ValueError):
        PyramidProjection(CH, 16, p2_fusion="concat")
    with pytest.raises(ValueError):
        PyramidProjection(CH, 16, use_fpn=False, p2_fusion="weighted")
    cfg = variant_cfg("no_p2")
    cfg["model"]["neck"]["p2_fusion"] = "weighted"
    with pytest.raises(ValueError):
        build_model(cfg)


def test_export_is_static(tmp_path):
    onnx = pytest.importorskip("onnx")
    model = build_model(variant_cfg("p2_weighted_fusion")).eval()
    model.ema_router = None

    class W(torch.nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m

        def forward(self, t):
            cls, reg, feats = self.m(t)
            c, _, boxes, _, _ = self.m.head.decode(cls, reg, feats)
            return c.sigmoid(), boxes

    path = tmp_path / "m.onnx"
    onnx_export(W(model), torch.randn(1, 3, 256, 256), str(path), input_names=["images"],
                output_names=["scores", "boxes"], opset_version=13, do_constant_folding=True,
                dynamic_axes=None)
    m = onnx.load(str(path))
    onnx.checker.check_model(m)
