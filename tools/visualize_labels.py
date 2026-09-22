"""Draw ground-truth boxes onto sampled images -- the P0-5 safety net.

A coordinate shift that was never applied (DroneVehicle's 100-px border) does
not raise an exception. It just trains a quietly wrong model. Look at the images.

    python tools/visualize_labels.py --config configs/datasets/dronevehicle_rgb.yaml \
        --split val --num 20 --out runs/label_check
"""
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from datasets.dronevehicle import build_dataset  # noqa: E402
from models.build import load_config  # noqa: E402

COLORS = [(66, 135, 245), (66, 245, 135), (245, 135, 66), (245, 66, 135),
          (135, 66, 245), (245, 200, 66), (66, 245, 245), (200, 66, 245),
          (120, 200, 120), (200, 120, 120)]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", nargs="+", required=True)
    ap.add_argument("--split", default="val", choices=["train", "val", "test"])
    ap.add_argument("--num", type=int, default=20)
    ap.add_argument("--out", default="runs/label_check")
    ap.add_argument("--raw", action="store_true", help="draw on the raw image, before augmentation")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    cfg = load_config(*a.config)
    ds = build_dataset(cfg, a.split, train=False)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    random.seed(a.seed)
    idxs = random.sample(range(len(ds)), min(a.num, len(ds)))

    empty = 0
    for i in idxs:
        if a.raw:
            img, boxes, labels, meta = ds.load_raw(i)
            img = img.copy()
        else:
            tensor, target = ds[i]
            img = (tensor.numpy().transpose(1, 2, 0)[:, :, ::-1] * 255).astype(np.uint8).copy()
            boxes = target["boxes"].numpy()
            labels = target["labels"].numpy()
            meta = target["meta"]
        if len(boxes) == 0:
            empty += 1
        for b, l in zip(boxes, labels):
            c = COLORS[int(l) % len(COLORS)]
            p1, p2 = (int(b[0]), int(b[1])), (int(b[2]), int(b[3]))
            cv2.rectangle(img, p1, p2, c, 1)
            cv2.putText(img, ds.classes[int(l)], (p1[0], max(p1[1] - 2, 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, c, 1, cv2.LINE_AA)
        tag = meta.get("condition", "")
        cv2.putText(img, f"{meta.get('file_name','')} n={len(boxes)} {tag}", (5, 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1, cv2.LINE_AA)
        cv2.imwrite(str(out / f"{i:06d}_{meta.get('file_name', 'img')}"), img)

    print(f"wrote {len(idxs)} images to {out}")
    if empty:
        print(f"WARNING: {empty}/{len(idxs)} sampled images had ZERO boxes -- "
              f"check the label directory and the class-name mapping.")
    print("Open them. Boxes must sit ON the objects, not offset by ~100 px.")


if __name__ == "__main__":
    main()
