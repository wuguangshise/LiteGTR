"""Evaluate a checkpoint; prints overall and per-condition metrics.

With ``--save-dir`` it also regenerates ``confusion_matrix.png``, ``pr_curve.png``
and ``val_predictions/``. Those need an inference pass -- unlike the training
curves, they cannot be rebuilt from ``results.csv`` -- so this is the way to
recover them if matplotlib was missing during training, or to produce them for a
checkpoint that is not the final epoch.
"""
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
from engine.evaluator import evaluate, postprocess_cfg  # noqa: E402
from models.build import build_model, load_config  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", nargs="+", required=True)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--split", default="val")
    ap.add_argument("--data-root", default=None, help="override data.root from the YAML")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--save-dir", default=None,
                    help="regenerate confusion_matrix.png / pr_curve.png / val_predictions/ here")
    ap.add_argument("--num-vis", type=int, default=16, help="how many qualitative images to dump")
    a = ap.parse_args()

    cfg = load_config(*a.config)
    if a.data_root:
        cfg["data"]["root"] = a.data_root
    ds = build_dataset(cfg, a.split, train=False)
    cfg["model"]["num_classes"] = len(ds.classes)
    loader = DataLoader(ds, batch_size=a.batch, shuffle=False, num_workers=cfg.get("loader", {}).get("num_workers", 4),
                        collate_fn=collate_fn)
    device = torch.device(a.device)
    model = build_model(cfg)
    CheckpointManager.load(a.weights, model, map_location=device)
    model.to(device)

    overall, by_cond = evaluate(model, loader, device, ds.classes, desc=a.split,
                                save_dir=a.save_dir, num_vis=a.num_vis, **postprocess_cfg(cfg))
    print("\n== overall ==")
    for k, v in overall.items():
        print(f"  {k:12s} {v:.4f}" if isinstance(v, float) else f"  {k:12s} {v}")
    if by_cond:
        print("\n== per condition ==")
        for c, m in by_cond.items():
            print(f"  {c:6s} n={m['num_images']:5d} mAP50:95={m['mAP50_95']:.4f} "
                  f"mAP50={m['mAP50']:.4f} AP_s={m['AP_small']:.4f}")
    if a.save_dir:
        print(f"\nplots + qualitative predictions written to {a.save_dir}")


if __name__ == "__main__":
    main()
