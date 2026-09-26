"""Box losses: NWD (losses/nwd.py), CIoU (losses/giou.py) and the default recipe."""
import pytest

torch = pytest.importorskip("torch")

from losses.giou import CIoULoss, bbox_ciou_aligned, bbox_iou_aligned  # noqa: E402
from losses.nwd import NWDLoss, nwd_aligned  # noqa: E402
from models.build import build_model, load_config  # noqa: E402


def test_nwd_is_size_invariant_to_offsets():
    small = torch.tensor([[0.0, 0.0, 6.0, 6.0]])
    large = torch.tensor([[0.0, 0.0, 60.0, 60.0]])
    shift = torch.tensor([[2.0, 0.0, 2.0, 0.0]])
    assert torch.allclose(nwd_aligned(small, small + shift), nwd_aligned(large, large + shift))
    assert torch.allclose(nwd_aligned(small, small), torch.ones(1), atol=1e-3)


def test_nwd_gives_gradient_without_overlap():
    pred = torch.tensor([[0.0, 0.0, 4.0, 4.0]], requires_grad=True)
    tgt = torch.tensor([[10.0, 10.0, 14.0, 14.0]])
    NWDLoss()(pred, tgt).backward()
    assert pred.grad.abs().sum() > 0


@pytest.mark.parametrize("path", ["configs/models/model_main.yaml", "configs/baselines/csp_n.yaml"])
def test_every_model_trains_with_giou(path):
    cfg = load_config(path)
    assert (cfg["loss"]["iou_type"], cfg["loss"]["iou_weight"], cfg["loss"]["nwd_weight"]) == ("giou", 2.0, 0.0)
    cfg["model"]["num_classes"] = 10
    model = build_model(cfg).train()
    assert model.nwd is None
    targets = [{"boxes": torch.tensor([[20.0, 20.0, 26.0, 26.0]]), "labels": torch.tensor([0])}]
    losses = model.loss(torch.randn(1, 3, 128, 128), targets)
    assert "loss_giou" in losses and "loss_ciou" not in losses and "loss_nwd" not in losses
    assert torch.isfinite(losses["loss_giou"])


def test_ciou_nwd_ablation_trains():
    cfg = load_config("configs/ablation/ciou_nwd.yaml")
    assert cfg["loss"]["nwd_weight"] > 0 and cfg["loss"]["iou_type"] == "ciou"
    cfg["model"]["num_classes"] = 10
    model = build_model(cfg).train()
    targets = [{"boxes": torch.tensor([[20.0, 20.0, 26.0, 26.0]]), "labels": torch.tensor([0])}]
    losses = model.loss(torch.randn(1, 3, 128, 128), targets)
    assert "loss_nwd" in losses and torch.isfinite(losses["loss_nwd"])
    assert "loss_ciou" in losses and "loss_giou" not in losses and torch.isfinite(losses["loss_ciou"])
    sum(losses.values()).backward()


def test_ciou_equals_iou_for_identical_and_concentric_same_shape():
    a = torch.tensor([[0.0, 0.0, 10.0, 10.0]])
    assert torch.allclose(bbox_ciou_aligned(a, a), torch.ones(1), atol=1e-5)
    inner = torch.tensor([[2.5, 2.5, 7.5, 7.5]])        # same centre and aspect ratio
    assert torch.allclose(bbox_ciou_aligned(inner, a), bbox_iou_aligned(inner, a), atol=1e-5)


def test_ciou_pulls_centres_when_one_box_contains_the_other():
    outer = torch.tensor([[0.0, 0.0, 20.0, 20.0]])
    pred = torch.tensor([[1.0, 1.0, 6.0, 6.0]], requires_grad=True)   # inside, off-centre
    CIoULoss()(pred, outer).backward()
    # moving the box toward the centre (+x, +y) must lower the loss
    assert (pred.grad[0, [0, 2]].sum() < 0) and (pred.grad[0, [1, 3]].sum() < 0)


def test_old_giou_configs_still_build():
    cfg = load_config("configs/models/model_main.yaml")
    cfg["model"]["num_classes"] = 10
    cfg["loss"].pop("iou_type"); cfg["loss"].pop("iou_weight"); cfg["loss"]["giou_weight"] = 2.0
    model = build_model(cfg)
    assert model.iou_type == "giou" and model.box_iou.loss_weight == 2.0
