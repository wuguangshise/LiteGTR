"""Routed detail enhancement (models/token/detail_enhance.py)."""
import pytest

torch = pytest.importorskip("torch")

from models.build import build_model, count_deploy_params, load_config  # noqa: E402
from models.token.detail_enhance import RoutedDetailEnhance  # noqa: E402
from tests._variants import variant_cfg  # noqa: E402
from utils.export import onnx_export  # noqa: E402

MASKS = ["score", "global", "token"]


def _model(mask):
    torch.manual_seed(0)
    return build_model(variant_cfg(f"detail_enhance{'' if mask == 'score' else '_' + mask}"))


@pytest.mark.parametrize("mask", MASKS)
def test_identity_at_initialisation(mask):
    """alpha starts at 0: switching the module on must not change a fresh model."""
    torch.manual_seed(0)
    base = build_model(variant_cfg("main")).eval()
    enh = _model(mask).eval()
    enh.load_state_dict(base.state_dict(), strict=False)
    x = torch.randn(1, 3, 256, 256)
    with torch.no_grad():
        a, b = base(x), enh(x)
    for u, v in zip(a[0] + a[1], b[0] + b[1]):
        assert torch.equal(u, v)


def test_costs_one_scalar_per_channel():
    main = count_deploy_params(build_model(variant_cfg("main")))
    enh = count_deploy_params(_model("score"))
    assert enh - main == load_config("configs/models/model_main.yaml")["model"]["neck"]["channels"]


@pytest.mark.parametrize("mask", MASKS)
def test_gradients_reach_alpha(mask):
    model = _model(mask).train()
    with torch.no_grad():
        for a in model.detail_enhance.alpha.values():
            a.fill_(0.1)
    targets = [{"boxes": torch.tensor([[20.0, 20.0, 40.0, 44.0]]), "labels": torch.tensor([1])}]
    sum(model.loss(torch.randn(1, 3, 256, 256), targets).values()).backward()
    g = model.detail_enhance.alpha["P2"].grad
    assert g is not None and torch.isfinite(g).all() and g.abs().sum() > 0


def test_score_mask_passes_gradient_to_the_scorer():
    """The soft mask is differentiable: the detection loss can shape WHERE to sharpen."""
    model = _model("score").train()
    with torch.no_grad():
        model.detail_enhance.alpha["P2"].fill_(0.1)
    model.routing_loss = None                 # isolate the path through the mask
    model.ema_router = None
    feats = torch.randn(1, 64, 64, 64)
    sel = model.selector({"P3": feats[:, :, ::2, ::2], "P4": feats[:, :, ::4, ::4],
                          "P5": feats[:, :, ::8, ::8]})
    out = model.detail_enhance({"P2": torch.randn(1, 64, 64, 64)}, sel["score_maps"], sel["coords"])
    out["P2"].sum().backward()
    w = model.selector.selectors["P3"].scorer.conv[-1].weight.grad
    assert w is not None and w.abs().sum() > 0


def test_mask_is_local():
    """With the score mask, detail is amplified where the score is high and left alone
    where it is low."""
    de = RoutedDetailEnhance(4, ["P2"], mask="score", source="P3", dilate=1)
    with torch.no_grad():
        de.alpha["P2"].fill_(1.0)
    f = torch.randn(1, 4, 8, 8)
    score = torch.full((1, 1, 4, 4), -30.0)
    score[..., 0, 0] = 30.0                   # an object in the top-left P3 cell only
    out = de({"P2": f}, {"P3": score}, None)["P2"]
    diff = (out - f).abs().sum(1)[0]
    assert diff[:2, :2].sum() > 0
    assert diff[2:, :].abs().max() < 1e-6 and diff[:, 2:].abs().max() < 1e-6


def test_unknown_mask_is_rejected():
    with pytest.raises(ValueError):
        RoutedDetailEnhance(4, ["P2"], mask="edges")


@pytest.mark.parametrize("mask", MASKS)
def test_export_is_static(tmp_path, mask):
    onnx = pytest.importorskip("onnx")
    model = _model(mask).eval()
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
    assert not [d.dim_param for vi in list(m.graph.input) + list(m.graph.output)
                for d in vi.type.tensor_type.shape.dim if d.dim_param]


def test_finite_under_fp16_autocast():
    model = _model("score").train()
    with torch.no_grad():
        model.detail_enhance.alpha["P2"].fill_(0.5)
    targets = [{"boxes": torch.tensor([[20.0, 20.0, 40.0, 44.0]]), "labels": torch.tensor([1])}]
    with torch.autocast("cpu", dtype=torch.float16):
        losses = model.loss(torch.randn(1, 3, 256, 256), targets)
    assert all(torch.isfinite(v) for v in losses.values()), losses
