"""Token routing must actually learn, and the EMA view must actually differ.

Two defects these tests pin down:

1. The scorer received NO gradient from the detection loss. ``top-k`` supplies
   only indices, the selected tokens were plain gathers of the feature map, and
   the score values were never used downstream -- so the routing was never
   trained on the task. ``score_gate`` is the fix.
2. The EMA teacher saw the student's exact input, so the consistency loss only
   measured weight drift and decayed to ~5e-6 on its own. The photometric view
   is the fix.
"""
import pytest

torch = pytest.importorskip("torch")

from models.build import build_model, load_config  # noqa: E402
from models.token.ema_token_router import photometric_view, preserved_bn_stats  # noqa: E402


def _model(path, **token_overrides):
    cfg = load_config(path)
    cfg["model"]["num_classes"] = 10
    cfg["model"]["token"].update(token_overrides)
    return build_model(cfg).train()


def _targets():
    return [{"boxes": torch.tensor([[20.0, 20.0, 60.0, 60.0], [90.0, 80.0, 140.0, 150.0]]),
             "labels": torch.tensor([0, 3])},
            {"boxes": torch.tensor([[30.0, 40.0, 80.0, 90.0]]), "labels": torch.tensor([1])}]


def _scorer_grad(model) -> float:
    total = 0.0
    for p in model.selector.parameters():
        if p.grad is not None:
            total += float(p.grad.abs().sum())
    return total


def test_scorer_learns_from_the_detection_loss_alone():
    """The regression test for defect 1. EMA is OFF, so the ONLY route into the
    scorer is the detection loss through the score gate."""
    model = _model("configs/ablation/no_ema_routing.yaml")
    losses = model.loss(torch.randn(2, 3, 256, 256), _targets())
    assert "loss_token" not in losses
    sum(losses.values()).backward()
    assert _scorer_grad(model) > 0, "scorer got no gradient from the detection loss"


def test_random_routing_leaves_the_scorer_untrained():
    """Without the gate and without EMA nothing reaches the scorer -- which is
    exactly what makes random_routing.yaml a clean random baseline."""
    model = _model("configs/ablation/random_routing.yaml")
    sum(model.loss(torch.randn(2, 3, 256, 256), _targets()).values()).backward()
    assert _scorer_grad(model) == 0.0


def test_photometric_view_changes_appearance_not_geometry():
    torch.manual_seed(0)
    x = torch.rand(4, 3, 32, 32)
    y = photometric_view(x)
    assert y.shape == x.shape
    assert float(y.min()) >= 0.0 and float(y.max()) <= 1.0
    assert not torch.allclose(x, y), "the view must differ from the input"
    # geometry: a bright pixel stays where it was
    z = torch.zeros(1, 3, 16, 16)
    z[..., 5, 9] = 1.0
    zy = photometric_view(z, noise=0.0)
    assert zy[0, 0].flatten().argmax().item() == 5 * 16 + 9


def test_teacher_sees_a_different_view_than_the_student():
    model = _model("configs/models/model_main.yaml")
    x = torch.randn(2, 3, 256, 256)
    feats = model.neck(model.backbone(x))
    feats = model.local_path(feats)
    tok_in = {lv: f for lv, f in zip(model.levels, feats) if lv in model.token_levels}
    t_in = model._teacher_inputs(x, tok_in)
    assert any(not torch.allclose(t_in[lv], tok_in[lv]) for lv in tok_in)


def test_same_view_ablation_really_is_the_same_input():
    model = _model("configs/ablation/ema_same_view.yaml")
    tok_in = {"P3": torch.randn(1, 64, 8, 8)}
    assert model._teacher_inputs(torch.randn(1, 3, 64, 64), tok_in) is tok_in


def test_teacher_branch_leaves_batchnorm_running_stats_untouched():
    model = _model("configs/models/model_main.yaml")
    bns = [m for m in model.modules() if isinstance(m, torch.nn.modules.batchnorm._BatchNorm)]
    assert bns, "expected BatchNorm layers in neck / local path"
    before = [(m.running_mean.clone(), m.running_var.clone()) for m in bns]
    x = torch.randn(2, 3, 256, 256)
    feats = model.local_path(model.neck(model.backbone(x)))
    tok_in = {lv: f for lv, f in zip(model.levels, feats) if lv in model.token_levels}
    snap = [(m.running_mean.clone(), m.running_var.clone()) for m in bns]   # after student pass
    model._teacher_inputs(x, tok_in)
    for (rm, rv), m in zip(snap, bns):
        assert torch.equal(rm, m.running_mean) and torch.equal(rv, m.running_var), \
            "the perturbed teacher pass leaked into BatchNorm running stats"
    assert before  # student pass is allowed to update them; the teacher pass is not


def test_preserved_bn_stats_restores_on_exception():
    bn = torch.nn.BatchNorm2d(4).train()
    rm = bn.running_mean.clone()
    with pytest.raises(RuntimeError):
        with preserved_bn_stats(bn):
            bn(torch.randn(2, 4, 3, 3) * 10 + 5)
            raise RuntimeError("boom")
    assert torch.equal(bn.running_mean, rm)


def test_consistency_loss_is_live_under_the_photometric_view():
    model = _model("configs/models/model_main.yaml")
    losses = model.loss(torch.randn(2, 3, 256, 256), _targets())
    assert "loss_token" in losses
    assert torch.isfinite(losses["loss_token"]) and float(losses["loss_token"]) > 0
