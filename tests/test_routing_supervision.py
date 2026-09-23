"""The scorer must be told where objects are, and must not be decayed flat.

The defect these tests pin down: with only the score gate and the EMA
consistency KL, nothing opposes a flat score map -- the KL is zero for any two
identical maps -- and weight decay drives the scorer toward one. The first
200-epoch run ended with ``score_entropy`` 0.999997: top-k over a flat map is
random routing. ``routing_sup`` (GT-centre focal loss) and ``scorer_no_decay``
are the fix; ``configs/ablation/no_routing_supervision.yaml`` reproduces the
defective configuration exactly.
"""
import pytest

torch = pytest.importorskip("torch")

from engine.trainer import build_optimizer  # noqa: E402
from losses.token_routing import TokenRoutingLoss, center_heatmap  # noqa: E402
from models.build import build_model, load_config  # noqa: E402
from tests._variants import variant_cfg  # noqa: E402


def _targets():
    return [{"boxes": torch.tensor([[20.0, 20.0, 60.0, 60.0], [90.0, 80.0, 140.0, 150.0]]),
             "labels": torch.tensor([0, 3])},
            {"boxes": torch.tensor([[30.0, 40.0, 80.0, 90.0]]), "labels": torch.tensor([1])}]


def _scorer_grad(model) -> float:
    return sum(float(p.grad.abs().sum()) for p in model.selector.parameters() if p.grad is not None)


# ---------------------------------------------------------------- heatmap
def test_heatmap_peaks_at_the_centre_cell():
    heat = center_heatmap(torch.tensor([[16.0, 16.0, 48.0, 48.0]]), 8, 8, stride=8)
    assert heat[4, 4] == 1.0                          # centre (32, 32) lies in cell (4, 4)
    assert int(heat.eq(1.0).sum()) == 1
    assert float(heat.min()) >= 0.0 and float(heat.max()) <= 1.0
    assert heat[4, 5] > heat[4, 7] > 0                # decays away from the centre


def test_heatmap_of_an_empty_image_is_all_negative():
    heat = center_heatmap(torch.zeros(0, 4), 6, 5, stride=16)
    assert heat.shape == (6, 5) and float(heat.abs().sum()) == 0.0


def test_tiny_object_on_a_coarse_level_keeps_the_minimum_radius():
    """A 6-px object on P5 (stride 32) is a fraction of a cell; sigma_min keeps its
    heatmap from degenerating to a single spike with no neighbours."""
    heat = center_heatmap(torch.tensor([[100.0, 100.0, 106.0, 106.0]]), 10, 10, stride=32)
    assert heat[3, 3] == 1.0
    assert 0.0 < float(heat[3, 4]) < 1.0


def test_routing_loss_prefers_peaked_maps_over_flat_ones():
    """A flat map is exactly what the EMA KL cannot penalise; this loss must."""
    boxes = [torch.tensor([[16.0, 16.0, 48.0, 48.0]])]
    heat = center_heatmap(boxes[0], 8, 8, stride=8)
    loss = TokenRoutingLoss(loss_weight=1.0)
    flat = loss({"P3": torch.zeros(1, 1, 8, 8)}, boxes, {"P3": 8})
    peaked = loss({"P3": (heat * 12 - 6).view(1, 1, 8, 8)}, boxes, {"P3": 8})
    assert float(peaked) < float(flat)


# ------------------------------------------------------------------ model
def test_main_config_supervises_routing():
    model = build_model(variant_cfg("main")).train()
    losses = model.loss(torch.randn(2, 3, 256, 256), _targets())
    assert "loss_route" in losses and torch.isfinite(losses["loss_route"])
    assert float(losses["loss_route"]) > 0


def test_supervision_reaches_the_scorer_on_the_first_step():
    """The score gate's gradient is zero at initialisation (the write-back gate
    gamma starts at 0 -- see test_routing.py). The routing loss acts on the score
    map directly, so the scorer learns from step one. No gate, no EMA: this is the
    supervision's own path."""
    model = build_model(variant_cfg("routing_sup_only")).train()
    losses = model.loss(torch.randn(2, 3, 256, 256), _targets())
    losses["loss_route"].backward()
    assert _scorer_grad(model) > 0


def test_scorer_output_conv_is_exempt_from_weight_decay():
    model = build_model(variant_cfg("main"))
    names = model.no_weight_decay()
    assert names == {f"selector.selectors.{lv}.scorer.conv.3.weight" for lv in model.token_levels}
    opt = build_optimizer(model, {"lr": 1e-3, "weight_decay": 0.05})
    params = dict(model.named_parameters())
    decayed = {id(p) for g in opt.param_groups if g["weight_decay"] > 0 for p in g["params"]}
    assert all(id(params[n]) not in decayed for n in names)
    # the rest of the scorer is decayed as before
    assert id(params[f"selector.selectors.{model.token_levels[0]}.scorer.conv.0.weight"]) in decayed


# ---------------------------------------------------------------- ablation
def test_ablation_reproduces_the_unsupervised_run():
    """no_routing_supervision must be the first run's configuration exactly: no
    routing loss, and the optimizer decays the same parameters it did then."""
    cfg = load_config("configs/ablation/no_routing_supervision.yaml")
    cfg["model"]["num_classes"] = 10
    model = build_model(cfg).train()
    assert model.routing_loss is None and model.no_weight_decay() == set()
    assert "loss_route" not in model.loss(torch.randn(2, 3, 256, 256), _targets())
    opt = build_optimizer(model, {"lr": 1e-3, "weight_decay": 0.05})
    decayed = {id(p) for g in opt.param_groups if g["weight_decay"] > 0 for p in g["params"]}
    for sel in model.selector.selectors.values():
        assert id(sel.scorer.conv[-1].weight) in decayed
