"""Baselines must run through the IDENTICAL head/loss/assigner path (P1-6).

If a baseline silently diverges from the shared protocol, the main table stops
being a controlled comparison.
"""
import pytest

torch = pytest.importorskip("torch")

from models.build import build_model, load_config  # noqa: E402

BASELINES = [
    "configs/baselines/csp_n.yaml",
    "configs/baselines/csp_t.yaml",
]


@pytest.mark.parametrize("path", BASELINES)
def test_baseline_forward_and_backward(path):
    cfg = load_config(path)
    cfg["model"]["num_classes"] = 10
    model = build_model(cfg).train()
    targets = [{"boxes": torch.tensor([[10.0, 10.0, 50.0, 50.0]]), "labels": torch.tensor([1])}]
    losses = model.loss(torch.randn(1, 3, 256, 256), targets)
    total = sum(losses.values())
    assert torch.isfinite(total)
    total.backward()


@pytest.mark.parametrize("path", BASELINES)
def test_baseline_shares_head_and_assigner(path):
    cfg = load_config(path)
    cfg["model"]["num_classes"] = 10
    model = build_model(cfg)
    main = build_model({**load_config("configs/models/model_main.yaml"),
                        "model": {**load_config("configs/models/model_main.yaml")["model"],
                                  "num_classes": 10}})
    assert type(model.head) is type(main.head)
    assert model.head.reg_max == main.head.reg_max
    assert type(model.assigner) is type(main.assigner)
    assert model.strides == main.strides


def test_csp_backbone_emits_expected_strides():
    cfg = load_config("configs/baselines/csp_n.yaml")
    cfg["model"]["num_classes"] = 10
    model = build_model(cfg).eval()
    with torch.no_grad():
        _, _, feats = model(torch.randn(1, 3, 256, 256))
    for f, s in zip(feats, model.strides):
        assert f.shape[-2:] == (256 // s, 256 // s)


def test_token_path_is_actually_disabled_in_baselines():
    cfg = load_config("configs/baselines/csp_n.yaml")
    cfg["model"]["num_classes"] = 10
    model = build_model(cfg)
    assert model.use_token is False
    assert model.selector is None and model.writeback is None and model.ema_router is None


@pytest.mark.parametrize("baseline,paired", [
    ("configs/baselines/csp_n.yaml", "configs/models/model_main.yaml"),
    ("configs/baselines/csp_t.yaml", "configs/models/model_edge_s.yaml"),
])
def test_baseline_is_parameter_matched(baseline, paired):
    """A baseline with half the parameters makes any win meaningless -- the first
    version of these configs was ~0.45x its paired model. Backbone params must
    stay within 10%; neck and head are shared, so totals then match as well."""
    def build(path):
        cfg = load_config(path)
        cfg["model"]["num_classes"] = 10
        return build_model(cfg)

    b, m = build(baseline), build(paired)
    pb = sum(p.numel() for p in b.backbone.parameters())
    pm = sum(p.numel() for p in m.backbone.parameters())
    assert abs(pb - pm) / pm < 0.10, f"backbone {pb/1e6:.2f}M vs paired {pm/1e6:.2f}M"
    assert b.neck.out_channels == m.neck.out_channels
