"""Write VisDrone test-dev submission files (original-image coordinates).

Internal evaluation keeps predictions and GT both in letterboxed space, which is
self-consistent. A submission is not: it must be in ORIGINAL image pixels, so
this maps back through the letterbox transform (``utils/boxes.unletterbox``).

VisDrone submission line: ``x,y,w,h,score,category,-1,-1`` with category in the
official 1..10 numbering (our contiguous 0..9 + 1).

    python tools/submit_visdrone.py --config configs/datasets/visdrone_rgb.yaml \
        configs/models/model_main.yaml --weights runs/train/<name>/weights/best.pt \
        --split test --out runs/submit/test-dev
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from datasets.base import collate_fn  # noqa: E402
from datasets.builder import build_dataset  # noqa: E402
from engine.checkpoint import CheckpointManager  # noqa: E402
from models.build import build_model, load_config  # noqa: E402
from utils.boxes import unletterbox  # noqa: E402


@torch.no_grad()
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", nargs="+", required=True)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--out", default="runs/submit")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--score-thr", type=float, default=0.01)
    ap.add_argument("--max-det", type=int, default=500)
    a = ap.parse_args()

    cfg = load_config(*a.config)
    if cfg["data"]["name"].lower() != "visdrone":
        raise SystemExit("this submission format is VisDrone-specific")
    ds = build_dataset(cfg, a.split, train=False)
    cfg["model"]["num_classes"] = len(ds.classes)
    loader = DataLoader(ds, batch_size=a.batch, shuffle=False, collate_fn=collate_fn,
                        num_workers=cfg.get("loader", {}).get("num_workers", 4))

    device = torch.device(a.device)
    model = build_model(cfg)
    CheckpointManager.load(a.weights, model, map_location=device)
    model.to(device).eval()

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    size = cfg["data"].get("img_size", 640)
    n = 0
    for images, targets in tqdm(loader, desc="submit"):
        preds = model.predict(images.to(device), score_thr=a.score_thr, max_det=a.max_det)
        for t, p in zip(targets, preds):
            meta = t["meta"]
            ori_hw = tuple(meta["ori_shape"])
            boxes = unletterbox(p["boxes"].float().cpu().numpy(), ori_hw, size)
            scores = p["scores"].float().cpu().numpy()
            labels = p["labels"].cpu().numpy()
            lines = []
            for b, s, l in zip(boxes, scores, labels):
                w, h = b[2] - b[0], b[3] - b[1]
                if w <= 1 or h <= 1:
                    continue
                lines.append(f"{b[0]:.2f},{b[1]:.2f},{w:.2f},{h:.2f},{s:.4f},{int(l) + 1},-1,-1")
            (out / f"{Path(meta['file_name']).stem}.txt").write_text("\n".join(lines),
                                                                    encoding="utf-8")
            n += 1

    print(f"wrote {n} result files to {out}")
    print("Zip the DIRECTORY CONTENTS (not the folder) for the VisDrone submission portal.")


if __name__ == "__main__":
    main()
