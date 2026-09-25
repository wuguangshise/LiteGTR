"""A step with a non-finite loss must leave no trace.

What happened without this guard: an fp16 overflow made the loss NaN (see
tests/test_fp16_attention.py). The GradScaler skipped the update, but the NaN
forward had already written BatchNorm running statistics -- copied into the
weight EMA, so validation read 0 from then on -- and each such step halved the
loss scale. Once it reached 0, the next finite step wrote NaN into every weight.
"""
import pytest

torch = pytest.importorskip("torch")

from engine.trainer import Trainer  # noqa: E402


class _Tiny(torch.nn.Module):
    """A model with the detector's training interface: loss() and ema_step()."""

    def __init__(self):
        super().__init__()
        self.lin = torch.nn.Linear(4, 4)
        self.bn = torch.nn.BatchNorm1d(4)

    def loss(self, images, targets):
        return {"loss_a": self.bn(self.lin(images)).pow(2).mean()}

    def ema_step(self):
        pass


def _cfg():
    return {"train": {"epochs": 2, "amp": False, "optimizer": "adamw", "lr": 1e-2,
                      "weight_decay": 0.0, "warmup_epochs": 0, "scheduler": "cosine",
                      "monitor": "mAP50_95", "grad_clip": 10.0,
                      "model_ema": {"enabled": True, "decay": 0.9, "tau": 1}}}


def _batch(bad=False):
    x = torch.randn(8, 4)
    if bad:
        x[0, 0] = float("inf")      # an overflowed activation: the BN forward turns it into NaN
    return x, [{}] * 8


def _trainer(tmp_path, loader):
    return Trainer(_Tiny(), loader, loader, _cfg(), torch.device("cpu"), tmp_path, ["a"])


def test_nonfinite_step_is_skipped_and_bn_statistics_restored(tmp_path):
    torch.manual_seed(0)
    loader = [_batch(), _batch(bad=True), _batch()]
    tr = _trainer(tmp_path, loader)
    out = tr.train_one_epoch(1)

    assert out["nonfinite_steps"] == 1
    assert all(torch.isfinite(p).all() for p in tr.model.parameters())
    assert torch.isfinite(tr.model.bn.running_mean).all()
    assert torch.isfinite(tr.model.bn.running_var).all()
    assert all(torch.isfinite(v).all() for v in tr.model_ema.state_dict().values()
               if v.is_floating_point())
    assert all(v == v for k, v in out.items() if k != "_data_wait"), out   # no NaN means
    assert tr.model_ema.updates == 2                                        # only the good steps


def test_bn_statistics_match_a_run_without_the_bad_batch(tmp_path):
    torch.manual_seed(0)
    good1, bad, good2 = _batch(), _batch(bad=True), _batch()
    torch.manual_seed(1)
    with_bad = _trainer(tmp_path / "a", [good1, bad, good2])
    torch.manual_seed(1)
    without = _trainer(tmp_path / "b", [good1, good2])
    with_bad.train_one_epoch(1)
    without.train_one_epoch(1)
    assert torch.equal(with_bad.model.bn.running_mean, without.model.bn.running_mean)
    assert torch.equal(with_bad.model.bn.running_var, without.model.bn.running_var)


def test_schedule_advances_on_a_skipped_step(tmp_path):
    loader = [_batch(), _batch(bad=True), _batch()]
    tr = _trainer(tmp_path, loader)
    tr.train_one_epoch(1)
    assert tr.scheduler.last_epoch == len(loader)
