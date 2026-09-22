"""Guard rail for docs/DESIGN.md P0-1.

The original plan silently blew its own budget: the backbone alone was 3.76M of
a 4.1-4.8M target. This test fails the build if that happens again.
"""
import pytest

from utils.budget import PRESETS, backbone_macs, backbone_params

torch = pytest.importorskip("torch")

from models.build import build_model, load_config  # noqa: E402

PARAM_LIMIT = 5.0e6
BACKBONE_SHARE_LIMIT = 0.60     # backbone must not eat the token path's budget


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


def test_backbone_does_not_dominate():
    cfg = load_config("configs/models/model_main.yaml")
    model = build_model(cfg)
    bb = sum(p.numel() for p in model.backbone.parameters())
    total = _deploy_params(model)
    assert bb / total <= BACKBONE_SHARE_LIMIT, (
        f"backbone is {100*bb/total:.0f}% of the model -- the token path is being starved")


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
