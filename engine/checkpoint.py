"""best.pt / last.pt / epoch_xx.pt, YOLO-style."""
from __future__ import annotations

from pathlib import Path

import torch


class CheckpointManager:
    def __init__(self, out_dir: str | Path, monitor: str = "mAP50_95", mode: str = "max",
                 save_period: int = 0):
        self.dir = Path(out_dir) / "weights"
        self.dir.mkdir(parents=True, exist_ok=True)
        self.monitor = monitor
        self.mode = mode
        self.save_period = save_period
        self.best = -float("inf") if mode == "max" else float("inf")
        self.best_metrics: dict = {}

    def _better(self, value: float) -> bool:
        return value > self.best if self.mode == "max" else value < self.best

    def save(self, model, optimizer, scheduler, epoch: int, metrics: dict, cfg: dict,
             model_ema=None, scaler=None) -> bool:
        # Decide improvement first so the payload records best-so-far INCLUDING this
        # epoch -- resume() restores it, otherwise the first epoch after a resume
        # would overwrite best.pt however bad it was.
        value = metrics.get(self.monitor, float("nan"))
        improved = value == value and self._better(value)   # NaN-safe
        if improved:
            self.best = value
            self.best_metrics = {**metrics, "epoch": epoch}
        payload = {
            "model": model.state_dict(),
            "model_ema": model_ema.state_dict() if model_ema is not None else None,
            "ema_updates": model_ema.updates if model_ema is not None else 0,
            "optimizer": optimizer.state_dict() if optimizer else None,
            "scheduler": scheduler.state_dict() if scheduler else None,
            "scaler": scaler.state_dict() if scaler is not None and scaler.is_enabled() else None,
            "epoch": epoch,
            "best": self.best,
            "best_metrics": self.best_metrics,
            "metrics": metrics,
            "config": cfg,
        }
        torch.save(payload, self.dir / "last.pt")
        if self.save_period and epoch % self.save_period == 0:
            torch.save(payload, self.dir / f"epoch_{epoch:03d}.pt")
        if improved:
            torch.save(payload, self.dir / "best.pt")
        return improved

    @staticmethod
    def load(path: str | Path, model, optimizer=None, scheduler=None, map_location="cpu",
             prefer_ema: bool = True) -> dict:
        """Load weights for EVALUATION: EMA weights when present, since they are what
        training evaluated. Do not use this to resume training -- see
        ``Trainer.resume()``, which restores raw weights and the EMA separately."""
        ck = torch.load(path, map_location=map_location, weights_only=False)
        sd = ck.get("model_ema") if (prefer_ema and ck.get("model_ema")) else ck["model"]
        model.load_state_dict(sd, strict=False)
        if optimizer and ck.get("optimizer"):
            optimizer.load_state_dict(ck["optimizer"])
        if scheduler and ck.get("scheduler"):
            scheduler.load_state_dict(ck["scheduler"])
        return ck
