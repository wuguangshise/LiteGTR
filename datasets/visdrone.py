"""VisDrone2019-DET.

Evaluation protocol (LOCKED -- docs/DESIGN.md P1-9)
---------------------------------------------------
* class 0 ``ignored regions`` and class 11 ``others`` are NOT objects.
  ``ignore_mode="mask"`` (default) paints ignored regions with the letterbox
  pad value so no feature ever fires there -- this matches the official
  toolkit's intent more closely than silently dropping the annotation.
* the remaining 10 categories map to contiguous ids 0..9.
* evaluation images are NOT painted by default (``data.eval_ignore_mode: drop``):
  a COCO-json evaluation, as used by RemDet and the mmdet VisDrone results, sees
  the untouched image and has no ignore regions.
* metrics are computed with **COCO API** (``datasets/metrics.py``) on ``val``, in
  original-image pixels, up to 1000 detections per image (mmdet's CocoMetric).
  The official MATLAB toolkit yields slightly different numbers; whichever you
  pick must be stated in the paper. Do not mix the two across tables.
* ``test-dev`` requires online submission and is not used for ablations.

Annotation line: ``x,y,w,h,score,category,truncation,occlusion``
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from datasets.base import DetectionDataset, imread

CLASSES = ["pedestrian", "people", "bicycle", "car", "van",
           "truck", "tricycle", "awning-tricycle", "bus", "motor"]
IGNORED_ID, OTHERS_ID = 0, 11


class VisDroneDataset(DetectionDataset):
    classes = CLASSES

    def __init__(self, root: str, split: str = "train", img_size: int = 640,
                 train: bool = True, mosaic_prob: float = 0.5,
                 ignore_mode: str = "mask", pad_value: int = 114):
        super().__init__(img_size, train, mosaic_prob)
        assert ignore_mode in ("mask", "drop")
        self.ignore_mode = ignore_mode
        self.pad_value = pad_value
        self.root = Path(root)
        img_dir = self.root / split / "images"
        ann_dir = self.root / split / "annotations"
        if not img_dir.is_dir():                      # tolerate VisDrone2019-DET-<split> layout
            cand = list(self.root.glob(f"*{split}*/images"))
            if cand:
                img_dir = cand[0]
                ann_dir = cand[0].parent / "annotations"
        self.img_dir, self.ann_dir = img_dir, ann_dir
        self.images = sorted(p for p in img_dir.iterdir() if p.suffix.lower() in {".jpg", ".png"})
        if not self.images:
            raise FileNotFoundError(f"no VisDrone images under {img_dir}")

    def __len__(self) -> int:
        return len(self.images)

    def load_raw(self, index: int):
        p = self.images[index]
        img = imread(p)
        boxes, labels, ignores = [], [], []
        ann = self.ann_dir / f"{p.stem}.txt"
        if ann.exists():
            for line in ann.read_text(encoding="utf-8").splitlines():
                parts = [v for v in line.replace(" ", "").split(",") if v != ""]
                if len(parts) < 6:
                    continue
                x, y, w, h = (float(v) for v in parts[:4])
                cat = int(parts[5])
                if w <= 0 or h <= 0:
                    continue
                if cat == IGNORED_ID:
                    ignores.append([x, y, x + w, y + h])
                elif cat != OTHERS_ID and 1 <= cat <= 10:
                    boxes.append([x, y, x + w, y + h])
                    labels.append(cat - 1)
        if ignores and self.ignore_mode == "mask":
            for x1, y1, x2, y2 in np.asarray(ignores, dtype=int):
                img[max(y1, 0):max(y2, 0), max(x1, 0):max(x2, 0)] = self.pad_value
        boxes = np.asarray(boxes, dtype=np.float32).reshape(-1, 4)
        labels = np.asarray(labels, dtype=np.int64).reshape(-1)
        return img, boxes, labels, {"image_id": index, "file_name": p.name,
                                    "ori_shape": img.shape[:2], "condition": "day"}
