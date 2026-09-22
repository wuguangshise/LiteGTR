"""Metric DEFINITIONS only.

The evaluation *loop* lives in ``engine/evaluator.py`` -- this split resolves the
duplicate-``evaluator.py`` ambiguity flagged in docs/DESIGN.md P2-11.
"""
from __future__ import annotations

import contextlib
import io

import numpy as np


class COCOMeanAP:
    """COCO-style AP via pycocotools, fed from in-memory predictions.

    Also exposes AP-small / medium / large, which are the headline numbers for
    UAV work, and supports evaluating a SUBSET of image ids so the evaluator can
    report per-condition metrics (day / night / dark) without re-running
    inference.
    """

    def __init__(self, classes: list[str]):
        self.classes = classes
        self.reset()

    def reset(self) -> None:
        self._gt: list[dict] = []
        self._dt: list[dict] = []
        self._images: dict[int, dict] = {}
        self._ann_id = 1

    def add(self, image_id: int, height: int, width: int, condition: str,
            gt_boxes: np.ndarray, gt_labels: np.ndarray,
            dt_boxes: np.ndarray, dt_scores: np.ndarray, dt_labels: np.ndarray) -> None:
        self._images[image_id] = {"id": image_id, "height": int(height), "width": int(width),
                                  "condition": condition}
        for b, l in zip(np.asarray(gt_boxes).reshape(-1, 4), np.asarray(gt_labels).reshape(-1)):
            w, h = float(b[2] - b[0]), float(b[3] - b[1])
            self._gt.append({"id": self._ann_id, "image_id": image_id, "category_id": int(l) + 1,
                             "bbox": [float(b[0]), float(b[1]), w, h], "area": w * h, "iscrowd": 0})
            self._ann_id += 1
        for b, s, l in zip(np.asarray(dt_boxes).reshape(-1, 4), np.asarray(dt_scores).reshape(-1),
                           np.asarray(dt_labels).reshape(-1)):
            self._dt.append({"image_id": image_id, "category_id": int(l) + 1,
                             "bbox": [float(b[0]), float(b[1]), float(b[2] - b[0]), float(b[3] - b[1])],
                             "score": float(s)})

    def evaluate(self, conditions: list[str] | None = None) -> dict:
        from pycocotools.coco import COCO
        from pycocotools.cocoeval import COCOeval

        img_ids = [i for i, m in self._images.items()
                   if conditions is None or m["condition"] in conditions]
        if not img_ids or not self._gt:
            return {k: float("nan") for k in
                    ("mAP50_95", "mAP50", "mAP75", "AP_small", "AP_medium", "AP_large", "num_images")}

        gt = {
            "images": [{"id": i, "height": self._images[i]["height"], "width": self._images[i]["width"]}
                       for i in img_ids],
            "annotations": [a for a in self._gt if a["image_id"] in set(img_ids)],
            "categories": [{"id": i + 1, "name": c} for i, c in enumerate(self.classes)],
        }
        with contextlib.redirect_stdout(io.StringIO()):
            coco_gt = COCO()
            coco_gt.dataset = gt
            coco_gt.createIndex()
            dt = [d for d in self._dt if d["image_id"] in set(img_ids)]
            if not dt:
                return {k: 0.0 for k in
                        ("mAP50_95", "mAP50", "mAP75", "AP_small", "AP_medium", "AP_large")} | \
                       {"num_images": len(img_ids)}
            coco_dt = coco_gt.loadRes(dt)
            e = COCOeval(coco_gt, coco_dt, "bbox")
            e.params.imgIds = img_ids
            e.evaluate(); e.accumulate(); e.summarize()
        s = e.stats
        return {"mAP50_95": float(s[0]), "mAP50": float(s[1]), "mAP75": float(s[2]),
                "AP_small": float(s[3]), "AP_medium": float(s[4]), "AP_large": float(s[5]),
                "num_images": len(img_ids)}

    def available_conditions(self) -> list[str]:
        return sorted({m["condition"] for m in self._images.values()})
