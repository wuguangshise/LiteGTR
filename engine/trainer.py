"""Training loop: AMP, cosine schedule with warmup, EMA routing step, checkpoints."""
from __future__ import annotations

import math
import time
from pathlib import Path

import torch
from tqdm import tqdm

from engine.checkpoint import CheckpointManager
from engine.ema import ModelEMA
from engine.evaluator import collect_token_stats, evaluate
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
    decay, no_decay = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (no_decay if p.ndim <= 1 or n.endswith(".gamma") or "level_embed" in n else decay).append(p)
    groups = [{"params": decay, "weight_decay": wd}, {"params": no_decay, "weight_decay": 0.0}]
    if t == "adamw":
        return torch.optim.AdamW(groups, lr=lr, betas=cfg.get("betas", (0.9, 0.999)))
    if t == "sgd":
        return torch.optim.SGD(groups, lr=lr, momentum=cfg.get("momentum", 0.937), nesterov=True)
    raise ValueError(f"unknown optimizer: {t}")


def build_scheduler(optimizer, cfg: dict, steps_per_epoch: int):
    epochs = cfg["epochs"]
    warmup = cfg.get("warmup_epochs", 3) * steps_per_epoch
    total = epochs * steps_per_epoch
    final = cfg.get("final_lr_ratio", 0.01)

    def fn(step: int) -> float:
        if step < warmup:
            return (step + 1) / max(warmup, 1)
        p = (step - warmup) / max(total - warmup, 1)
        return final + (1 - final) * 0.5 * (1 + math.cos(math.pi * p))

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

    def train_one_epoch(self, epoch: int) -> dict:
        self.model.train()
        agg: dict[str, float] = {}
        nb = 0
        pbar = tqdm(self.train_loader, desc=f"epoch {epoch}/{self.epochs}", leave=False)
        for images, targets in pbar:
            images = images.to(self.device, non_blocking=True)
            with torch.autocast(device_type=self.device.type, enabled=self.amp):
                losses = self.model.loss(images, targets)
                total = sum(losses.values())
            self.optimizer.zero_grad(set_to_none=True)
            self.scaler.scale(total).backward()
            if self.clip:
                self.scaler.unscale_(self.optimizer)
                gn = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)
                agg["grad_norm"] = agg.get("grad_norm", 0.0) + float(gn)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.scheduler.step()
            self.model.ema_step()          # token-routing teacher; no-op when disabled
            if self.model_ema is not None:
                self.model_ema.update(self.model)

            for k, v in losses.items():
                agg[k] = agg.get(k, 0.0) + float(v)
            agg["loss"] = agg.get("loss", 0.0) + float(total)
            nb += 1
            pbar.set_postfix(loss=f"{float(total):.3f}", lr=f"{self.scheduler.get_last_lr()[0]:.2e}")
        return {k: v / max(nb, 1) for k, v in agg.items()}

    def fit(self) -> dict:
        self.recorder.logger.info(f"training for {self.epochs} epochs on {self.device}")
        best: dict = {}
        for epoch in range(self.start_epoch, self.epochs + 1):
            t0 = time.time()
            tr = self.train_one_epoch(epoch)
            row = {"epoch": epoch, **{f"train/{k}": _sig(v) for k, v in tr.items()},
                   "lr": self.scheduler.get_last_lr()[0]}

            if epoch % self.cfg["train"].get("val_interval", 1) == 0 or epoch == self.epochs:
                eval_model = self._eval_model()
                # run artefacts (confusion matrix, PR curves, val_predictions) are
                # written on the final epoch only -- they are slow and only the
                # last state is reported
                save_dir = self.recorder.dir if epoch == self.epochs else None
                overall, by_cond = evaluate(eval_model, self.val_loader, self.device,
                                            self.classes, amp=self.amp, save_dir=save_dir)
                row.update({f"val/{k}": _sig(v) for k, v in overall.items()})
                if by_cond:
                    self.recorder.log_conditions(epoch, by_cond)
                    self.recorder.logger.info(
                        "per-condition mAP50:95 -> " +
                        ", ".join(f"{c}={m['mAP50_95']:.4f}(n={m['num_images']})"
                                  for c, m in by_cond.items()))
                ts = collect_token_stats(eval_model, self.val_loader, self.device)
                if ts:
                    self.recorder.log_tokens({"epoch": epoch, **ts})
                improved = self.ckpt.save(self.model, self.optimizer, self.scheduler,
                                          epoch, overall, self.cfg, model_ema=self.model_ema)
                if improved:
                    best = dict(overall)
                    best["epoch"] = epoch
            else:
                self.ckpt.save(self.model, self.optimizer, self.scheduler, epoch, {}, self.cfg,
                               model_ema=self.model_ema)

            row["time"] = round(time.time() - t0, 1)
            self.recorder.log_epoch(row)
            self.recorder.logger.info(" | ".join(f"{k}={v}" for k, v in row.items()))
            self.recorder.plot_curves()

        self.recorder.save_json("best_metrics.json", best)
        return best

    def _eval_model(self):
        """Evaluate the EMA weights when weight-EMA is on, else the raw model."""
        return self.model if self.model_ema is None else self.model_ema.ema
