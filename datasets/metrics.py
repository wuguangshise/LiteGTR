"""Metric DEFINITIONS only.

The evaluation *loop* lives in ``engine/evaluator.py`` -- this split resolves the
duplicate-``evaluator.py`` ambiguity flagged in docs/DESIGN.md P2-11.
"""
from __future__ import annotations

import contextlib
import io

import numpy as np


KEYS = ("mAP50_95", "mAP50", "mAP75", "AP_small", "AP_medium", "AP_large", "AP_vt", "AP_t")

# Up to 1000 detections per image count, as in mmdet/mmyolo's CocoMetric
# (proposal_nums=(100, 300, 1000); COCO's AP uses the last) -- the protocol of
# RemDet and the other VisDrone papers we compare with. pycocotools' default of
# 100 silently drops correct detections on the many VisDrone images with more
# than 100 objects.
MAX_DETS = [100, 300, 1000]
# COCO buckets (small < 32^2 <= medium < 96^2 <= large) plus AI-TOD's
# very tiny (2-8 px) and tiny (8-16 px), all in ORIGINAL-image pixels.
AREA_RNG = [[0, 1e10], [0, 32 ** 2], [32 ** 2, 96 ** 2], [96 ** 2, 1e10],
            [2 ** 2, 8 ** 2], [8 ** 2, 16 ** 2]]
AREA_LBL = ["all", "small", "medium", "large", "verytiny", "tiny"]


class COCOMeanAP:
    """COCO-style AP via pycocotools, fed from in-memory predictions.

    Boxes must be in ORIGINAL-image pixels (engine/evaluator.py maps them back
    from the letterbox): COCO's size buckets are defined there, and that is where
    every paper we compare with measures them. Reports AP-small / medium / large
    plus AI-TOD's AP_vt / AP_t, and supports evaluating a SUBSET of image ids so
    the evaluator can report per-condition metrics (day / night / dark) without
    re-running inference.
    """

    def __init__(self, classes: list[str]):
        self.classes = classes
        self.reset()

    def reset(self) -> None:
        self._gt: list[dict] = []
        self._dt: list[dict] = []
        self._images: dict[int, dict] = {}
        self._ann_id = 1
        self._last_eval = None      # COCOeval object from the most recent evaluate()

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
            return {k: float("nan") for k in (*KEYS, "num_images")}

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
                return {k: 0.0 for k in KEYS} | {"num_images": len(img_ids)}
            coco_dt = coco_gt.loadRes(dt)
            e = COCOeval(coco_gt, coco_dt, "bbox")
            e.params.imgIds = img_ids
            e.params.maxDets = list(MAX_DETS)
            e.params.areaRng = [list(r) for r in AREA_RNG]
            e.params.areaRngLbl = list(AREA_LBL)
            e.evaluate(); e.accumulate(); e.summarize()
        self._last_eval = e
        s = e.stats
        return {"mAP50_95": float(s[0]), "mAP50": float(s[1]), "mAP75": float(s[2]),
                "AP_small": float(s[3]), "AP_medium": float(s[4]), "AP_large": float(s[5]),
                "AP_vt": self._ap(e, "verytiny"), "AP_t": self._ap(e, "tiny"),
                "num_images": len(img_ids)}

    @staticmethod
    def _ap(e, area: str) -> float:
        """AP@[.5:.95] for one extra area range at the largest maxDets -- what
        ``summarize()`` reports for small/medium/large, for the AI-TOD ranges."""
        a = e.params.areaRngLbl.index(area)
        p = e.eval["precision"][:, :, :, a, -1]
        p = p[p > -1]
        return float(p.mean()) if p.size else -1.0

    def available_conditions(self) -> list[str]:
        return sorted({m["condition"] for m in self._images.values()})

    def pr_curves(self, iou_index: int = 0, area_index: int = 0) -> dict:
        """Per-class (recall, precision) arrays from the last ``evaluate()`` call.

        ``COCOeval.eval['precision']`` has shape [T, R, K, A, M]: IoU threshold,
        recall threshold, category, area range, maxDets. Defaults select
        IoU=0.50 and area='all' at the largest maxDets setting.
        """
        e = self._last_eval
        if e is None or not getattr(e, "eval", None):
            return {}
        prec = e.eval["precision"]
        rec_thrs = e.params.recThrs
        out: dict[str, tuple] = {}
        for k, name in enumerate(self.classes):
            if k >= prec.shape[2]:
                break
            p = prec[iou_index, :, k, area_index, -1]
            valid = p > -1
            if valid.any():
                out[name] = (np.asarray(rec_thrs)[valid], np.asarray(p)[valid])
        return out
