"""Training loop: AMP, cosine schedule with warmup, EMA routing step, checkpoints."""
from __future__ import annotations

import math
import time
from pathlib import Path

import torch
from tqdm import tqdm

from torch.utils.data import DataLoader, RandomSampler

from engine.checkpoint import CheckpointManager
from engine.ema import ModelEMA
from engine.evaluator import collect_token_stats, evaluate, postprocess_cfg, vis_cfg
from engine.recorder import Recorder


def _sig(v: float, digits: int = 5) -> float:
    """Round to significant figures, not decimal places.

    ``round(v, 5)`` keeps five DECIMAL places, so any loss below 5e-6 is written
    as 0.0 -- to the log and to results.csv, where the precision is then gone for
    good. Small terms such as the token-consistency loss are exactly the ones you
    need to see. Five significant figures leaves ordinary values unchanged
    (0.93329 stays 0.93329) while keeping 4.9e-06 as 4.9e-06.
    """
    return float(f"{v:.{digits}g}")


def build_optimizer(model, cfg: dict):
    t = cfg.get("optimizer", "adamw").lower()
    lr = cfg.get("lr", 1e-3)
    wd = cfg.get("weight_decay", 0.05)
    skip = model.no_weight_decay() if hasattr(model, "no_weight_decay") else set()
    decay, no_decay = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        exempt = p.ndim <= 1 or n.endswith(".gamma") or "level_embed" in n or n in skip
        (no_decay if exempt else decay).append(p)
    groups = [{"params": decay, "weight_decay": wd}, {"params": no_decay, "weight_decay": 0.0}]
    if t == "adamw":
        return torch.optim.AdamW(groups, lr=lr, betas=cfg.get("betas", (0.9, 0.999)))
    if t == "sgd":
        return torch.optim.SGD(groups, lr=lr, momentum=cfg.get("momentum", 0.937), nesterov=True)
    raise ValueError(f"unknown optimizer: {t}")


def build_scheduler(optimizer, cfg: dict, steps_per_epoch: int):
    """Linear warmup, then either plain cosine or flat-then-cosine.

    ``flat_cosine`` (DEIM, CVPR 2025 -- ``lrsheduler: flatcosine``) holds the peak
    learning rate for ``flat_epochs`` before the cosine decay begins. DEIM holds
    it for about half the schedule (``flat_epoch: 29`` of ~58). Training from
    scratch benefits most: the network spends longer at a learning rate high
    enough to move, instead of starting to decay while it is still far from a
    good basin.

    The decay floor is ``final_lr_ratio`` of the base rate. DEIM's 0.5 is tuned
    for a ~58-epoch fine-tune of a pretrained backbone; a 200-epoch from-scratch
    run keeps the conventional low floor.
    """
    epochs = cfg["epochs"]
    warmup = cfg.get("warmup_epochs", 3) * steps_per_epoch
    total = epochs * steps_per_epoch
    final = cfg.get("final_lr_ratio", 0.01)
    kind = cfg.get("scheduler", "cosine")
    if kind not in ("cosine", "flat_cosine"):
        raise ValueError(f"unknown scheduler: {kind!r} (expected 'cosine' or 'flat_cosine')")
    flat_end = max(cfg.get("flat_epochs", 0) * steps_per_epoch, warmup) if kind == "flat_cosine" else warmup

    def fn(step: int) -> float:
        if step < warmup:
            return (step + 1) / max(warmup, 1)
        if step < flat_end:
            return 1.0
        p = (step - flat_end) / max(total - flat_end, 1)
        return final + (1 - final) * 0.5 * (1 + math.cos(math.pi * min(p, 1.0)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, fn)


class Trainer:
    def __init__(self, model, train_loader, val_loader, cfg: dict, device, out_dir: str | Path,
                 classes: list[str]):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.cfg = cfg
        self.device = device
        self.classes = classes
        tc = cfg["train"]
        self.epochs = tc["epochs"]
        self.amp = tc.get("amp", True) and device.type == "cuda"
        self.clip = tc.get("grad_clip", 10.0)
        self.optimizer = build_optimizer(self.model, tc)
        self.scheduler = build_scheduler(self.optimizer, tc, max(len(train_loader), 1))
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.amp)
        # weight EMA (generic training trick) -- distinct from the token-routing
        # EMA teacher, which is part of the method itself
        ec = tc.get("model_ema", {})
        self.model_ema = (ModelEMA(self.model, decay=ec.get("decay", 0.9998),
                                   tau=ec.get("tau", 2000)) if ec.get("enabled", True) else None)
        self.recorder = Recorder(out_dir)
        self.ckpt = CheckpointManager(out_dir, monitor=tc.get("monitor", "mAP50_95"),
                                      save_period=tc.get("save_period", 0))
        self.start_epoch = 1
        # Turn mosaic off for the final epochs (YOLO close_mosaic=10; DEIM no_aug_epoch=8)
        # so the model finishes on the real image distribution. HSV and flip stay on.
        self.no_aug_epochs = tc.get("no_aug_epochs", 0)
        self._mosaic_closed = False
        # BatchNorm running statistics are written during the FORWARD pass, before a
        # non-finite loss can be seen; train_one_epoch restores them on a skipped step.
        self._bn_buffers = [b for m in self.model.modules()
                            if isinstance(m, torch.nn.modules.batchnorm._BatchNorm)
                            and m.track_running_stats
                            for b in (m.running_mean, m.running_var)]

    def train_one_epoch(self, epoch: int) -> dict:
        """One pass over the training set.

        Per-step statistics are accumulated as detached GPU tensors and read back
        only every ``log_every`` steps. ``float(tensor)`` forces the CPU to wait
        for the GPU, so reading every loss term every step serialises the two
        instead of letting the CPU queue the next step while the GPU runs this
        one -- a real cost for a small model with many small kernels. The maths
        is unchanged; only when the numbers are copied to the host differs.

        Also returns ``_data_wait``: seconds spent blocked on the DataLoader. If
        it is a large share of the epoch, the input pipeline is the bottleneck.

        A step whose loss is not finite is skipped whole -- no backward, no update,
        no EMA -- and the BatchNorm statistics its forward wrote are put back. Handing
        a NaN loss to the GradScaler instead is not harmless: the scaler skips the
        update but the NaN forward has already poisoned BatchNorm running statistics
        (which the weight EMA copies, so validation reads 0), and every such step
        halves the loss scale. Enough of them drive it to exactly 0, and then a finite
        step writes NaN into every weight: 0 gradients pass the scaler's inf check and
        are only then multiplied by 1/0. That is how one overflowing layer ended a
        200-epoch run in epoch 2. The check costs one host sync per step; skipped
        steps are counted in ``nonfinite_steps`` and left out of the epoch means.
        """
        self.model.train()
        agg: dict[str, torch.Tensor] = {}
        nb = 0
        skipped = 0
        log_every = self.cfg["train"].get("log_every", 20)
        data_wait = 0.0
        pbar = tqdm(self.train_loader, desc=f"epoch {epoch}/{self.epochs}", leave=False)
        it = iter(pbar)
        while True:
            t_fetch = time.perf_counter()
            try:
                images, targets = next(it)
            except StopIteration:
                break
            data_wait += time.perf_counter() - t_fetch

            images = images.to(self.device, non_blocking=True)
            bn_state = [b.clone() for b in self._bn_buffers]
            with torch.autocast(device_type=self.device.type, enabled=self.amp):
                losses = self.model.loss(images, targets)
                total = sum(losses.values())
            if not torch.isfinite(total):
                with torch.no_grad():
                    for b, saved in zip(self._bn_buffers, bn_state):
                        b.copy_(saved)
                skipped += 1
                self.scheduler.step()      # the schedule is defined in steps; keep it aligned
                continue
            self.optimizer.zero_grad(set_to_none=True)
            self.scaler.scale(total).backward()
            if self.clip:
                self.scaler.unscale_(self.optimizer)
                gn = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)
                agg["grad_norm"] = agg.get("grad_norm", 0.0) + gn.detach()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.scheduler.step()
            self.model.ema_step()          # token-routing teacher; no-op when disabled
            if self.model_ema is not None:
                self.model_ema.update(self.model)

            for k, v in losses.items():
                agg[k] = agg.get(k, 0.0) + v.detach()
            agg["loss"] = agg.get("loss", 0.0) + total.detach()
            nb += 1
            if nb % log_every == 0:        # the only per-step host sync
                pbar.set_postfix(loss=f"{float(total):.3f}",
                                 lr=f"{self.scheduler.get_last_lr()[0]:.2e}")
        out = {k: float(v) / max(nb, 1) for k, v in agg.items()}
        out["nonfinite_steps"] = skipped
        if skipped:
            self.recorder.logger.warning(
                f"epoch {epoch}: skipped {skipped} step(s) with a non-finite loss")
        out["_data_wait"] = data_wait
        return out

    def _close_mosaic(self, epoch: int) -> None:
        """Disable mosaic and rebuild the loader.

        Setting ``dataset.mosaic_prob`` alone is not enough: with persistent
        workers -- and with Windows' spawn start method in general -- each worker
        holds its own pickled copy of the dataset, so a change made in the main
        process never reaches them. The loader is recreated with identical
        settings so fresh workers pick up the change. Checked every epoch, so a
        run resumed past the switch point also comes back with mosaic off.
        """
        old = self.train_loader
        ds = old.dataset
        if getattr(ds, "mosaic_prob", 0.0) == 0.0:
            self._mosaic_closed = True
            return
        ds.mosaic_prob = 0.0
        self.train_loader = DataLoader(
            ds, batch_size=old.batch_size, shuffle=isinstance(old.sampler, RandomSampler),
            num_workers=old.num_workers, collate_fn=old.collate_fn, pin_memory=old.pin_memory,
            drop_last=old.drop_last, persistent_workers=old.num_workers > 0)
        del old
        self._mosaic_closed = True
        self.recorder.logger.info(f"epoch {epoch}: mosaic closed for the final "
                                  f"{self.no_aug_epochs} epochs")

    def fit(self) -> dict:
        self.recorder.logger.info(f"training for {self.epochs} epochs on {self.device}")
        for epoch in range(self.start_epoch, self.epochs + 1):
            t0 = time.time()
            if (self.no_aug_epochs and not self._mosaic_closed
                    and epoch > self.epochs - self.no_aug_epochs):
                self._close_mosaic(epoch)
            if self.device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(self.device)
            tr = self.train_one_epoch(epoch)
            data_wait = tr.pop("_data_wait", 0.0)
            t_train = time.time() - t0
            row = {"epoch": epoch, **{f"train/{k}": _sig(v) for k, v in tr.items()},
                   "lr": self.scheduler.get_last_lr()[0]}
            if self.device.type == "cuda":
                # A main run went from ~150 s to ~1850 s of pure train time per epoch
                # at epoch 155, with data wait unchanged, and was back to normal after
                # a resume. On Windows the driver silently moves CUDA allocations into
                # system RAM once the card is full ("sysmem fallback"), which is ~10x
                # slower and raises no error. Logging the peak here makes a creeping
                # footprint visible in results.csv; handing the allocator's cached,
                # fragmented blocks back once per epoch keeps it from creeping.
                gib = 1024 ** 3
                row["mem/peak_reserved_GiB"] = round(torch.cuda.max_memory_reserved(self.device) / gib, 2)
                row["mem/peak_allocated_GiB"] = round(torch.cuda.max_memory_allocated(self.device) / gib, 2)
                torch.cuda.empty_cache()

            t_val0 = time.time()
            t_tok = 0.0
            if self._should_validate(epoch):
                eval_model = self._eval_model()
                # run artefacts (confusion matrix, PR curves, val_predictions) are
                # written on the final epoch only -- they are slow and only the
                # last state is reported
                save_dir = self.recorder.dir if epoch == self.epochs else None
                overall, by_cond = evaluate(eval_model, self.val_loader, self.device,
                                            self.classes, amp=self.amp, save_dir=save_dir,
                                            vis=vis_cfg(self.cfg), **postprocess_cfg(self.cfg))
                row.update({f"val/{k}": _sig(v) for k, v in overall.items()})
                if by_cond:
                    self.recorder.log_conditions(epoch, by_cond)
                    self.recorder.logger.info(
                        "per-condition mAP50:95 -> " +
                        ", ".join(f"{c}={m['mAP50_95']:.4f}(n={m['num_images']})"
                                  for c, m in by_cond.items()))
                t_tok0 = time.time()
                ts = collect_token_stats(eval_model, self.val_loader, self.device)
                t_tok = time.time() - t_tok0
                if ts:
                    self.recorder.log_tokens({"epoch": epoch, **ts})
                self.ckpt.save(self.model, self.optimizer, self.scheduler,
                               epoch, overall, self.cfg, model_ema=self.model_ema,
                               scaler=self.scaler)
            else:
                self.ckpt.save(self.model, self.optimizer, self.scheduler, epoch, {}, self.cfg,
                               model_ema=self.model_ema, scaler=self.scaler)

            t_val = time.time() - t_val0
            row["time"] = round(time.time() - t0, 1)
            # where the epoch went: data wait is time blocked on the DataLoader
            row["time/data"] = round(data_wait, 1)
            row["time/train"] = round(t_train - data_wait, 1)
            row["time/val"] = round(t_val - t_tok, 1)
            row["time/token_stats"] = round(t_tok, 1)
            self.recorder.log_epoch(row)
            self.recorder.logger.info(" | ".join(f"{k}={v}" for k, v in row.items()))
            self.recorder.plot_curves()

        # Held by the checkpoint manager, not a local, so a resumed run that never
        # improves still reports the best from before the interruption.
        best = self.ckpt.best_metrics
        self.recorder.save_json("best_metrics.json", best)
        return best

    def resume(self, path: str | Path) -> int:
        """Restore a run from ``last.pt`` and return the epoch to continue from.

        Deliberately NOT ``CheckpointManager.load()``, which is for evaluation and
        prefers the EMA weights. Resuming that way had three consequences:

        * the EMA-smoothed weights were loaded into the model being optimised, so
          training continued from a different point than where it stopped;
        * the ModelEMA -- built in ``__init__`` as a copy of the *randomly
          initialised* model -- was never restored, and since validation uses it,
          mAP collapsed after a resume until ~10k steps of decay 0.9999 washed the
          random weights out;
        * ``best`` restarted at -inf, so the first epoch overwrote best.pt.

        Loading is strict: resuming with a different model config than the one
        that wrote the checkpoint must fail loudly rather than load partially.
        """
        ck = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(ck["model"], strict=True)
        epoch = int(ck.get("epoch", 0))
        if self.model_ema is not None:
            if ck.get("model_ema"):
                self.model_ema.load_state_dict(ck["model_ema"])
                # Older checkpoints did not record the update count. Resetting it to 0
                # would restart the decay ramp and let raw weights overwrite the EMA;
                # optimiser steps so far is exact, since the loader drops the last batch.
                self.model_ema.updates = int(ck.get("ema_updates",
                                                    epoch * len(self.train_loader)))
            else:   # checkpoint predates EMA state: start the EMA from the raw weights
                self.model_ema.load_state_dict(self.model.state_dict())
        if ck.get("optimizer"):
            self.optimizer.load_state_dict(ck["optimizer"])
        if ck.get("scheduler"):
            self.scheduler.load_state_dict(ck["scheduler"])
        if ck.get("scaler") and self.scaler.is_enabled():
            self.scaler.load_state_dict(ck["scaler"])
        if "best" in ck:
            self.ckpt.best = ck["best"]
            self.ckpt.best_metrics = ck.get("best_metrics") or {}
        else:
            self._best_from_log()            # older checkpoint: recover best from results.csv
        self.start_epoch = epoch + 1
        self.recorder.logger.info(
            f"resumed from {path}: continuing at epoch {self.start_epoch}, "
            f"best {self.ckpt.monitor}={self.ckpt.best}, ema_updates="
            f"{self.model_ema.updates if self.model_ema is not None else '-'}")
        return self.start_epoch

    def _best_from_log(self) -> None:
        """Recover best-so-far from results.csv for checkpoints that predate the
        ``best`` field, so the first resumed epoch cannot overwrite best.pt."""
        path = self.recorder.results_csv
        col = f"val/{self.ckpt.monitor}"
        if not path.exists():
            return
        import csv
        with open(path, "r", newline="", encoding="utf-8") as f:
            rows = [r for r in csv.DictReader(f) if r.get(col) not in (None, "")]
        if not rows:
            return
        top = max(rows, key=lambda r: float(r[col]))
        self.ckpt.best = float(top[col])
        self.ckpt.best_metrics = {k.split("/", 1)[1]: float(v) for k, v in top.items()
                                  if k.startswith("val/") and v not in (None, "")}
        self.ckpt.best_metrics["epoch"] = int(float(top["epoch"]))

    def _should_validate(self, epoch: int) -> bool:
        """Every ``val_interval`` epochs, and EVERY epoch in the final
        ``val_dense_last`` epochs.

        Validation never touches the weights, so this cannot change the trained
        model -- only how finely best.pt is chosen. With a flat-then-cosine
        schedule the best epoch lands in the decay tail (and the mosaic-off
        phase sits inside it), which is exactly where validation stays dense.
        """
        tc = self.cfg["train"]
        interval = max(int(tc.get("val_interval", 1)), 1)
        dense = int(tc.get("val_dense_last", 0))
        return epoch % interval == 0 or epoch > self.epochs - dense or epoch == self.epochs

    def _eval_model(self):
        """Evaluate the EMA weights when weight-EMA is on, else the raw model."""
        return self.model if self.model_ema is None else self.model_ema.ema
