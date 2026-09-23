"""Forward/backward smoke tests for every structural switch."""
import pytest

torch = pytest.importorskip("torch")

from models.build import build_model, load_config  # noqa: E402

CONFIGS = [
    "configs/models/model_main.yaml",
    "configs/models/model_edge_s.yaml",
    "configs/ablation/no_global_token.yaml",
    "configs/ablation/no_local_cnn.yaml",
    "configs/ablation/no_p2.yaml",
    "configs/ablation/no_fpn.yaml",
    "configs/ablation/no_geometric_writeback.yaml",
    "configs/ablation/broadcast_writeback.yaml",
    "configs/ablation/token_budget_128.yaml",
    "configs/ablation/mixer_none.yaml",
    "configs/ablation/mixer_deep.yaml",
    "configs/ablation/global_topk_routing.yaml",
    "configs/ablation/token_src_p5.yaml",
    "configs/ablation/token_src_p4p5.yaml",
    "configs/ablation/writeback_p2.yaml",
    "configs/ablation/random_routing.yaml",
    "configs/ablation/no_ema_routing.yaml",
    "configs/ablation/ema_same_view.yaml",
]


@pytest.mark.parametrize("path", CONFIGS)
def test_forward_shapes(path):
    cfg = load_config(path)
    cfg["model"]["num_classes"] = 10
    model = build_model(cfg).eval()
    x = torch.randn(2, 3, 256, 256)
    with torch.no_grad():
        cls, reg, feats = model(x)
    assert len(cls) == len(model.levels) == len(feats)
    for c, r, f, s in zip(cls, reg, feats, model.strides):
        assert c.shape[0] == 2 and c.shape[1] == 10
        assert c.shape[-2:] == (256 // s, 256 // s)
        assert r.shape[1] == 4 * (model.head.reg_max + 1)


@pytest.mark.parametrize("path", CONFIGS)
def test_loss_backward(path):
    cfg = load_config(path)
    cfg["model"]["num_classes"] = 10
    model = build_model(cfg).train()
    x = torch.randn(2, 3, 256, 256)
    targets = [
        {"boxes": torch.tensor([[20.0, 20.0, 60.0, 60.0], [100.0, 90.0, 130.0, 140.0]]),
         "labels": torch.tensor([0, 3])},
        {"boxes": torch.zeros(0, 4), "labels": torch.zeros(0, dtype=torch.long)},   # empty image
    ]
    losses = model.loss(x, targets)
    total = sum(losses.values())
    assert torch.isfinite(total), losses
    total.backward()
    grads = [p.grad for p in model.parameters() if p.requires_grad and p.grad is not None]
    assert grads, "no gradients produced"
    assert all(torch.isfinite(g).all() for g in grads)


def test_predict_returns_boxes():
    cfg = load_config("configs/models/model_main.yaml")
    cfg["model"]["num_classes"] = 10
    model = build_model(cfg).eval()
    out = model.predict(torch.randn(2, 3, 256, 256), score_thr=0.0, max_det=10)
    assert len(out) == 2
    for o in out:
        assert o["boxes"].shape[-1] == 4
        assert o["boxes"].shape[0] == o["scores"].shape[0] == o["labels"].shape[0]


def test_token_geometry_is_normalised():
    cfg = load_config("configs/models/model_main.yaml")
    cfg["model"]["num_classes"] = 10
    model = build_model(cfg).eval()
    with torch.no_grad():
        model.extract_feats(torch.randn(1, 3, 256, 256))
    assert model._last_student_maps is not None


def test_ema_teacher_updates_but_takes_no_gradient():
    cfg = load_config("configs/models/model_main.yaml")
    cfg["model"]["num_classes"] = 10
    model = build_model(cfg).train()
    before = [p.clone() for p in model.ema_router.teacher.parameters()]
    for p in model.selector.parameters():
        p.data.add_(0.5)
    model.ema_step()
    after = list(model.ema_router.teacher.parameters())
    assert any(not torch.equal(a, b) for a, b in zip(before, after)), "teacher never moved"
    assert all(not p.requires_grad for p in model.ema_router.teacher.parameters())
