"""Box format conversion and letterbox coordinate restoration.

Predictions live in letterboxed ``img_size`` space. Internal evaluation keeps
both GT and predictions in that space, so it is self-consistent -- but anything
leaving the framework (VisDrone test-dev submission, visual comparison against
the original image) must be mapped back to original-image pixels.
"""
from __future__ import annotations

import numpy as np


def letterbox_params(ori_hw: tuple[int, int], size: int) -> tuple[float, float, float]:
    """Return ``(ratio, pad_x, pad_y)`` matching ``datasets.transforms.LetterBox``."""
    h, w = ori_hw
    r = min(size / h, size / w)
    nh, nw = int(round(h * r)), int(round(w * r))
    return r, (size - nw) / 2.0, (size - nh) / 2.0


def unletterbox(boxes: np.ndarray, ori_hw: tuple[int, int], size: int) -> np.ndarray:
    """Map xyxy boxes from letterboxed space back to the original image."""
    boxes = np.asarray(boxes, dtype=np.float32).reshape(-1, 4).copy()
    if not len(boxes):
        return boxes
    r, px, py = letterbox_params(ori_hw, size)
    boxes[:, [0, 2]] = (boxes[:, [0, 2]] - px) / r
    boxes[:, [1, 3]] = (boxes[:, [1, 3]] - py) / r
    h, w = ori_hw
    boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, w)
    boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, h)
    return boxes


def xyxy_to_xywh(boxes: np.ndarray) -> np.ndarray:
    b = np.asarray(boxes, dtype=np.float32).reshape(-1, 4).copy()
    b[:, 2] -= b[:, 0]
    b[:, 3] -= b[:, 1]
    return b


def box_areas(boxes: np.ndarray) -> np.ndarray:
    b = np.asarray(boxes, dtype=np.float32).reshape(-1, 4)
    return np.maximum(b[:, 2] - b[:, 0], 0) * np.maximum(b[:, 3] - b[:, 1], 0)


def iou_matrix(a: np.ndarray, b: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    """Pairwise IoU between (N,4) and (M,4) xyxy arrays -> (N, M). NumPy only."""
    a = np.asarray(a, dtype=np.float32).reshape(-1, 4)
    b = np.asarray(b, dtype=np.float32).reshape(-1, 4)
    if not len(a) or not len(b):
        return np.zeros((len(a), len(b)), dtype=np.float32)
    lt = np.maximum(a[:, None, :2], b[None, :, :2])
    rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
    wh = np.clip(rb - lt, 0, None)
    inter = wh[..., 0] * wh[..., 1]
    return inter / (box_areas(a)[:, None] + box_areas(b)[None, :] - inter + eps)


COCO_AREA_RANGES = {"small": (0, 32 ** 2), "medium": (32 ** 2, 96 ** 2), "large": (96 ** 2, 1e10)}


def size_bucket(area: float) -> str:
    for name, (lo, hi) in COCO_AREA_RANGES.items():
        if lo <= area < hi:
            return name
    return "large"
