"""NWD box loss (losses/nwd.py) and its place in the default recipe."""
import pytest

torch = pytest.importorskip("torch")

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
def test_every_model_trains_with_nwd(path):
    cfg = load_config(path)
    assert cfg["loss"]["nwd_weight"] > 0
    cfg["model"]["num_classes"] = 10
    model = build_model(cfg).train()
    targets = [{"boxes": torch.tensor([[20.0, 20.0, 26.0, 26.0]]), "labels": torch.tensor([0])}]
    losses = model.loss(torch.randn(1, 3, 128, 128), targets)
    assert "loss_nwd" in losses and torch.isfinite(losses["loss_nwd"])
    sum(losses.values()).backward()
