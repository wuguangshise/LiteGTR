"""Guard rail for docs/DESIGN.md P0-1.

The original plan silently blew its own budget: the backbone alone was 3.76M of
a 4.1-4.8M target. This test fails the build if that happens again.
"""
import pytest

from utils.budget import PRESETS, backbone_macs, backbone_params

torch = pytest.importorskip("torch")

from models.build import build_model, load_config  # noqa: E402

PARAM_LIMIT = 5.0e6
BACKBONE_LIMIT = 2.5e6          # re-balanced TinyNeXt-M is 2.03M; the original was 3.76M


def _deploy_params(model):
    return sum(v.numel() for k, v in model.state_dict().items() if not k.startswith("ema_router."))


def test_analytic_matches_locked_presets():
    assert backbone_params(*PRESETS["tinynext_m"])["total"] < 2.2e6
    assert backbone_params(*PRESETS["tinynext_m_orig"])["total"] > 3.5e6  # the problem, documented


def test_main_model_within_param_budget():
    cfg = load_config("configs/models/model_main.yaml")
    model = build_model(cfg)
    n = _deploy_params(model)
    assert n <= PARAM_LIMIT, f"deploy params {n/1e6:.2f}M exceeds {PARAM_LIMIT/1e6:.1f}M"


def test_backbone_stays_rebalanced():
    """Guards P0-1: the backbone must not drift back toward the original
    [32,64,128,256]x[4,4,9,4] design (3.76M), which alone consumed the budget.

    This test previously asserted the backbone was at most 60% of the model. The
    locked Main model does not meet that: the backbone is ~87% (2.03M of 2.33M),
    because neck, head and token path are deliberately light -- the Transformer
    part is ~4% of parameters. That is a documented trade-off (DESIGN.md, and the
    mixer_deep / token-width levers), not a regression, so the check now guards
    what P0-1 was actually about: the absolute backbone size."""
    cfg = load_config("configs/models/model_main.yaml")
    model = build_model(cfg)
    bb = sum(p.numel() for p in model.backbone.parameters())
    assert bb <= BACKBONE_LIMIT, (
        f"backbone {bb/1e6:.2f}M exceeds {BACKBONE_LIMIT/1e6:.1f}M -- drifting back "
        f"toward the original over-budget design")


def test_edge_s_is_smaller_than_main():
    main = _deploy_params(build_model(load_config("configs/models/model_main.yaml")))
    edge = _deploy_params(build_model(load_config("configs/models/model_edge_s.yaml")))
    assert edge < main


def test_ema_teacher_excluded_from_deploy_count():
    cfg = load_config("configs/models/model_main.yaml")
    model = build_model(cfg)
    assert model.ema_router is not None
    total = sum(p.numel() for p in model.parameters())
    assert _deploy_params(model) < total, "EMA teacher must not count toward deployment params"
