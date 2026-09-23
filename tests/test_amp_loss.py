"""AMP regression: the loss must not assume predictions and ground truth share a dtype.

Under autocast the head emits half while the DataLoader supplies float32 ground
truth. The assigner's IoU then promotes to float32, and writing that back into a
half target tensor raises

    RuntimeError: Index put requires the source and destination dtypes match

which is a first-batch crash, not something that shows up later. CPU autocast
with bfloat16 reproduces the same promotion path without needing a GPU.
"""
import pytest

torch = pytest.importorskip("torch")

from models.build import build_model, load_config  # noqa: E402

CONFIGS = ["configs/models/model_main.yaml", "configs/ablation/no_global_token.yaml"]


def _targets():
    return [
        {"boxes": torch.tensor([[20.0, 20.0, 60.0, 60.0], [90.0, 80.0, 140.0, 150.0]]),
         "labels": torch.tensor([0, 3])},
        {"boxes": torch.zeros(0, 4), "labels": torch.zeros(0, dtype=torch.long)},
    ]


@pytest.mark.parametrize("path", CONFIGS)
def test_loss_runs_under_autocast(path):
    cfg = load_config(path)
    cfg["model"]["num_classes"] = 10
    model = build_model(cfg).train()
    x = torch.randn(2, 3, 256, 256)
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        losses = model.loss(x, _targets())
    total = sum(losses.values())
    assert torch.isfinite(total), losses
    total.backward()


def test_loss_terms_are_fp32_under_autocast():
    """Every loss term must come back fp32 regardless of the autocast dtype."""
    cfg = load_config("configs/models/model_main.yaml")
    cfg["model"]["num_classes"] = 10
    model = build_model(cfg).train()
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        losses = model.loss(torch.randn(2, 3, 256, 256), _targets())
    for name, v in losses.items():
        assert v.dtype == torch.float32, f"{name} came back {v.dtype}"


def test_float32_ground_truth_with_half_predictions_does_not_raise():
    """The exact failing combination, stated directly."""
    cfg = load_config("configs/models/model_main.yaml")
    cfg["model"]["num_classes"] = 10
    model = build_model(cfg).train().half()
    targets = _targets()                      # stays float32, as the DataLoader gives it
    losses = model.loss(torch.randn(2, 3, 256, 256).half(), targets)
    assert torch.isfinite(sum(losses.values()))
