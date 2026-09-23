"""Resuming must restore the run exactly where it stopped.

The earlier resume path used CheckpointManager.load(), which is meant for
evaluation and prefers the EMA weights. It loaded the EMA-smoothed weights into
the model being optimised, never restored the ModelEMA (left as a copy of the
randomly initialised model, so validation mAP collapsed after a resume), and
restarted `best` at -inf so the next epoch overwrote best.pt regardless.
"""
import pytest

torch = pytest.importorskip("torch")

from engine.trainer import Trainer  # noqa: E402


class _Tiny(torch.nn.Module):
    def __init__(self, width: int = 4):
        super().__init__()
        self.lin = torch.nn.Linear(4, width)
        self.bn = torch.nn.BatchNorm1d(width)

    def forward(self, x):
        return self.bn(self.lin(x))


def _cfg():
    return {"train": {"epochs": 10, "amp": False, "optimizer": "adamw", "lr": 1e-3,
                      "weight_decay": 0.05, "warmup_epochs": 1, "scheduler": "flat_cosine",
                      "flat_epochs": 5, "monitor": "mAP50_95",
                      "model_ema": {"enabled": True, "decay": 0.9999, "tau": 2000}}}


def _trainer(out, width=4):
    loader = [0] * 4   # __init__ only needs len() to size the schedule
    return Trainer(_Tiny(width), loader, loader, _cfg(), torch.device("cpu"), out, ["a"])


def _vals(sd):
    return list(sd.values())


@pytest.fixture
def saved(tmp_path):
    a = _trainer(tmp_path / "a")
    for _ in range(3):
        a.model_ema.update(a.model)
    with torch.no_grad():                     # move the raw weights AFTER the EMA update,
        for p in a.model.parameters():        # so raw and EMA weights are clearly different
            p.add_(1.0)
    a.ckpt.save(a.model, a.optimizer, a.scheduler, epoch=3, metrics={"mAP50_95": 0.2},
                cfg=_cfg(), model_ema=a.model_ema, scaler=a.scaler)
    return a, tmp_path / "a" / "weights" / "last.pt"


def test_resume_restores_raw_weights_into_the_model(saved, tmp_path):
    a, ck = saved
    b = _trainer(tmp_path / "b")
    assert b.resume(ck) == 4
    for x, y in zip(_vals(a.model.state_dict()), _vals(b.model.state_dict())):
        assert torch.equal(x, y)
    # and NOT the EMA weights -- the original bug
    assert any(not torch.equal(x, y)
               for x, y in zip(_vals(b.model.state_dict()), _vals(b.model_ema.state_dict())))


def test_resume_restores_the_model_ema_and_its_update_count(saved, tmp_path):
    a, ck = saved
    b = _trainer(tmp_path / "b")
    b.resume(ck)
    for x, y in zip(_vals(a.model_ema.state_dict()), _vals(b.model_ema.state_dict())):
        assert torch.equal(x, y)
    assert b.model_ema.updates == a.model_ema.updates == 3


def test_resume_restores_best_so_a_worse_epoch_cannot_overwrite_best_pt(saved, tmp_path):
    _, ck = saved
    b = _trainer(tmp_path / "b")
    b.resume(ck)
    assert b.ckpt.best == 0.2
    assert b.ckpt.save(b.model, b.optimizer, b.scheduler, epoch=4, metrics={"mAP50_95": 0.1},
                       cfg=_cfg(), model_ema=b.model_ema, scaler=b.scaler) is False
    assert b.ckpt.best == 0.2


def test_resume_refuses_a_checkpoint_from_a_different_model(saved, tmp_path):
    _, ck = saved
    other = _trainer(tmp_path / "c", width=8)
    with pytest.raises(RuntimeError):
        other.resume(ck)
