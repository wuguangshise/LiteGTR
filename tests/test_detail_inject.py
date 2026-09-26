"""Routed detail injection (models/token/detail_inject.py)."""
import pytest

torch = pytest.importorskip("torch")

from models.build import build_model, count_deploy_params, load_config  # noqa: E402
from models.token.detail_inject import RoutedDetailInject  # noqa: E402
from tests._variants import variant_cfg  # noqa: E402
from utils.export import onnx_export  # noqa: E402

VARIANTS = ["detail_inject", "detail_inject_global", "detail_inject_before_head",
            "detail_inject_writeback_p2", "detail_inject_no_token"]
BASE_OF = {"detail_inject_writeback_p2": "writeback_p2", "detail_inject_no_token": "no_global_token"}
TARGETS = [{"boxes": torch.tensor([[20.0, 20.0, 40.0, 44.0]]), "labels": torch.tensor([1])}]


def _model(name):
    torch.manual_seed(0)
    return build_model(variant_cfg(name))


@pytest.mark.parametrize("name", VARIANTS)
def test_identity_at_initialisation(name):
    """gamma starts at 0: switching the module on -- and moving P2's local path after
    the selector -- must not change a fresh model."""
    torch.manual_seed(0)
    base = build_model(variant_cfg(BASE_OF.get(name, "main"))).eval()
    inj = _model(name).eval()
    missing, unexpected = inj.load_state_dict(base.state_dict(), strict=False)
    assert not unexpected and all(k.startswith("detail_inject.") for k in missing)
    x = torch.randn(1, 3, 256, 256)
    with torch.no_grad():
        a, b = base(x), inj(x)
    for u, v in zip(a[0] + a[1], b[0] + b[1]):
        assert torch.equal(u, v)


def test_injection_reaches_p2_only():
    model = _model("detail_inject").eval()
    with torch.no_grad():
        model.detail_inject.gamma.fill_(1.0)
    ref = _model("detail_inject").eval()
    x = torch.randn(1, 3, 256, 256)
    with torch.no_grad():
        fa, fb = model.extract_feats(x), ref.extract_feats(x)
    assert not torch.equal(fa[0], fb[0])
    for u, v in zip(fa[1:], fb[1:]):
        assert torch.equal(u, v)


def test_cost():
    cfg = load_config("configs/models/model_main.yaml")
    dim, mid = cfg["model"]["neck"]["channels"], cfg["model"]["backbone"]["channels"][0] // 2
    main = count_deploy_params(build_model(variant_cfg("main")))
    inj = count_deploy_params(_model("detail_inject"))
    assert inj - main == 4 * mid * dim + dim + dim          # 1x1 conv + bias + gamma


@pytest.mark.parametrize("name", ["detail_inject", "detail_inject_global", "detail_inject_before_head"])
def test_gradients_reach_the_module(name):
    model = _model(name).train()
    with torch.no_grad():
        model.detail_inject.gamma.fill_(0.1)
    sum(model.loss(torch.randn(1, 3, 256, 256), TARGETS).values()).backward()
    for p in (model.detail_inject.gamma, model.detail_inject.proj.weight):
        assert p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0


def test_gamma_learns_from_zero():
    model = _model("detail_inject").train()
    sum(model.loss(torch.randn(1, 3, 256, 256), TARGETS).values()).backward()
    g = model.detail_inject.gamma.grad
    assert g is not None and g.abs().sum() > 0


def test_detached_mask_leaves_the_scorer_to_its_supervision():
    model = _model("detail_inject").train()
    with torch.no_grad():
        model.detail_inject.gamma.fill_(0.1)
    feats = torch.randn(1, 64, 64, 64)
    sel = model.selector({"P3": feats[:, :, ::2, ::2], "P4": feats[:, :, ::4, ::4],
                          "P5": feats[:, :, ::8, ::8]})
    out = model.detail_inject(torch.randn(1, 64, 64, 64), torch.randn(1, 16, 128, 128), sel["score_maps"])
    out.sum().backward()
    assert model.selector.selectors["P3"].scorer.conv[-1].weight.grad is None


def test_mask_is_local():
    """Stem detail enters where the score is high and nowhere else."""
    di = RoutedDetailInject(2, 4, mask="score", source="P3", dilate=1)
    with torch.no_grad():
        di.gamma.fill_(1.0)
    p2, s2 = torch.randn(1, 4, 8, 8), torch.randn(1, 2, 16, 16)
    score = torch.full((1, 1, 4, 4), -30.0)
    score[..., 0, 0] = 30.0                   # an object in the top-left P3 cell only
    diff = (di(p2, s2, {"P3": score}) - p2).abs().sum(1)[0]
    assert diff[:2, :2].sum() > 0
    assert diff[2:, :].max() < 1e-6 and diff[:, 2:].max() < 1e-6


def test_bad_settings_are_rejected():
    with pytest.raises(ValueError):
        RoutedDetailInject(2, 4, mask="edges")
    with pytest.raises(ValueError):
        build_model(variant_cfg("patchify_stem") | {"model": {**variant_cfg("patchify_stem")["model"],
                                                              "detail_inject": {"enabled": True}}})
    cfg = variant_cfg("detail_inject")
    cfg["model"]["detail_inject"]["inject_at"] = "nowhere"
    with pytest.raises(ValueError):
        build_model(cfg)
    cfg = variant_cfg("no_global_token")
    cfg["model"]["detail_inject"] = {"enabled": True, "mask": "score"}
    with pytest.raises(ValueError):
        build_model(cfg)


@pytest.mark.parametrize("name", ["detail_inject", "detail_inject_global"])
def test_export_is_static(tmp_path, name):
    onnx = pytest.importorskip("onnx")
    model = _model(name).eval()
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
    model = _model("detail_inject").train()
    with torch.no_grad():
        model.detail_inject.gamma.fill_(0.5)
    with torch.autocast("cpu", dtype=torch.float16):
        losses = model.loss(torch.randn(1, 3, 256, 256), TARGETS)
    assert all(torch.isfinite(v) for v in losses.values()), losses
