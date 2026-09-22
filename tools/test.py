"""Test-split evaluation (same path as val.py, different default split)."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from datasets.base import collate_fn  # noqa: E402
from datasets.builder import build_dataset  # noqa: E402
from engine.checkpoint import CheckpointManager  # noqa: E402
from engine.evaluator import evaluate  # noqa: E402
from models.build import build_model, load_config  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", nargs="+", required=True)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--batch", type=int, default=8)
    a = ap.parse_args()

    cfg = load_config(*a.config)
    ds = build_dataset(cfg, a.split, train=False)
    cfg["model"]["num_classes"] = len(ds.classes)
    loader = DataLoader(ds, batch_size=a.batch, shuffle=False, num_workers=cfg.get("loader", {}).get("num_workers", 4),
                        collate_fn=collate_fn)
    device = torch.device(a.device)
    model = build_model(cfg)
    CheckpointManager.load(a.weights, model, map_location=device)
    model.to(device)

    overall, by_cond = evaluate(model, loader, device, ds.classes, desc=a.split)
    print("\n== overall ==")
    for k, v in overall.items():
        print(f"  {k:12s} {v:.4f}" if isinstance(v, float) else f"  {k:12s} {v}")
    if by_cond:
        print("\n== per condition ==")
        for c, m in by_cond.items():
            print(f"  {c:6s} n={m['num_images']:5d} mAP50:95={m['mAP50_95']:.4f} "
                  f"mAP50={m['mAP50']:.4f} AP_s={m['AP_small']:.4f}")


if __name__ == "__main__":
    main()
