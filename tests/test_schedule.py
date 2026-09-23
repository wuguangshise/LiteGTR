"""LR schedule shape and the close-mosaic switch."""
import logging
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from torch.utils.data import DataLoader, Dataset  # noqa: E402

from engine.trainer import Trainer, build_scheduler  # noqa: E402

SPE = 10  # steps per epoch


def _lrs(cfg):
    opt = torch.optim.SGD([torch.nn.Parameter(torch.zeros(1))], lr=1.0)
    sch = build_scheduler(opt, cfg, SPE)
    out = []
    for _ in range(cfg["epochs"] * SPE):
        out.append(opt.param_groups[0]["lr"])
        opt.step()
        sch.step()
    return out


def test_flat_cosine_holds_the_peak_then_decays_to_the_floor():
    lrs = _lrs(dict(epochs=20, warmup_epochs=2, scheduler="flat_cosine", flat_epochs=10,
                    final_lr_ratio=0.01))
    assert lrs[0] < 0.2                                   # warming up
    assert all(abs(v - 1.0) < 1e-9 for v in lrs[2 * SPE:10 * SPE]), "flat phase must sit at peak"
    assert lrs[10 * SPE + 5] < 1.0                        # decay has started
    assert abs(lrs[-1] - 0.01) < 0.02                     # ends near the floor
    assert all(a >= b - 1e-12 for a, b in zip(lrs[10 * SPE:], lrs[10 * SPE + 1:])), "monotone decay"


def test_flat_cosine_keeps_more_lr_mid_run_than_cosine():
    base = dict(epochs=20, warmup_epochs=2, flat_epochs=10, final_lr_ratio=0.01)
    fc = _lrs({**base, "scheduler": "flat_cosine"})
    co = _lrs({**base, "scheduler": "cosine"})
    assert fc[10 * SPE] > co[10 * SPE]


def test_unknown_scheduler_is_rejected():
    with pytest.raises(ValueError):
        _lrs(dict(epochs=2, scheduler="step"))


class _Toy(Dataset):
    def __init__(self):
        self.mosaic_prob = 0.5

    def __len__(self):
        return 8

    def __getitem__(self, i):
        return torch.tensor(self.mosaic_prob)


def test_close_mosaic_rebuilds_the_loader_so_workers_see_the_change():
    """Mutating the dataset in the main process does not reach existing worker
    copies; the loader has to be recreated. Assert what the NEW loader yields."""
    ds = _Toy()
    loader = DataLoader(ds, batch_size=4, shuffle=True, num_workers=0, drop_last=True)
    ns = SimpleNamespace(train_loader=loader, no_aug_epochs=3, _mosaic_closed=False,
                         recorder=SimpleNamespace(logger=logging.getLogger("t")))
    Trainer._close_mosaic(ns, epoch=18)
    assert ns._mosaic_closed is True
    assert ns.train_loader is not loader
    assert ns.train_loader.batch_size == 4 and ns.train_loader.drop_last is True
    assert all(float(b.max()) == 0.0 for b in ns.train_loader)
