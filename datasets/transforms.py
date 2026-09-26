"""Augmentation pipeline. Mosaic is on by default -- it is one of the largest
single wins for small-object AP on UAV imagery."""
from __future__ import annotations

import random

import cv2
import numpy as np


class LetterBox:
    """Resize with preserved aspect ratio and pad to a square canvas."""

    def __init__(self, size: int = 640, pad_value: int = 114):
        self.size = size
        self.pad = pad_value

    def __call__(self, img: np.ndarray, boxes: np.ndarray):
        h, w = img.shape[:2]
        r = min(self.size / h, self.size / w)
        nh, nw = int(round(h * r)), int(round(w * r))
        if (nh, nw) != (h, w):
            img = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
        top = (self.size - nh) // 2
        left = (self.size - nw) // 2
        canvas = np.full((self.size, self.size, 3), self.pad, dtype=img.dtype)
        canvas[top:top + nh, left:left + nw] = img
        if len(boxes):
            boxes = boxes.copy()
            boxes[:, [0, 2]] = boxes[:, [0, 2]] * r + left
            boxes[:, [1, 3]] = boxes[:, [1, 3]] * r + top
        return canvas, boxes


class RandomHSV:
    def __init__(self, h: float = 0.015, s: float = 0.7, v: float = 0.4):
        self.gains = (h, s, v)

    def __call__(self, img: np.ndarray, boxes: np.ndarray):
        r = np.random.uniform(-1, 1, 3) * self.gains + 1
        hue, sat, val = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2HSV))
        dtype = img.dtype
        x = np.arange(0, 256, dtype=np.int16)
        lut_h = ((x * r[0]) % 180).astype(dtype)
        lut_s = np.clip(x * r[1], 0, 255).astype(dtype)
        lut_v = np.clip(x * r[2], 0, 255).astype(dtype)
        merged = cv2.merge((cv2.LUT(hue, lut_h), cv2.LUT(sat, lut_s), cv2.LUT(val, lut_v)))
        return cv2.cvtColor(merged, cv2.COLOR_HSV2BGR), boxes


class RandomHFlip:
    def __init__(self, p: float = 0.5):
        self.p = p

    def __call__(self, img: np.ndarray, boxes: np.ndarray):
        if random.random() < self.p:
            img = img[:, ::-1]
            if len(boxes):
                w = img.shape[1]
                boxes = boxes.copy()
                boxes[:, [0, 2]] = w - boxes[:, [2, 0]]
        return np.ascontiguousarray(img), boxes


def mosaic4(samples: list[tuple[np.ndarray, np.ndarray, np.ndarray]], size: int = 640,
            pad_value: int = 114):
    """Classic 4-image mosaic. ``samples`` is [(img, boxes_xyxy, labels), ...] of length 4."""
    s = size
    canvas = np.full((s * 2, s * 2, 3), pad_value, dtype=np.uint8)
    cx = int(random.uniform(s * 0.5, s * 1.5))
    cy = int(random.uniform(s * 0.5, s * 1.5))
    out_boxes, out_labels = [], []
    for i, (img, boxes, labels) in enumerate(samples):
        h, w = img.shape[:2]
        r = min(s / h, s / w)
        img = cv2.resize(img, (int(w * r), int(h * r)))
        h, w = img.shape[:2]
        if i == 0:
            x1a, y1a, x2a, y2a = max(cx - w, 0), max(cy - h, 0), cx, cy
            x1b, y1b = w - (x2a - x1a), h - (y2a - y1a)
        elif i == 1:
            x1a, y1a, x2a, y2a = cx, max(cy - h, 0), min(cx + w, s * 2), cy
            x1b, y1b = 0, h - (y2a - y1a)
        elif i == 2:
            x1a, y1a, x2a, y2a = max(cx - w, 0), cy, cx, min(s * 2, cy + h)
            x1b, y1b = w - (x2a - x1a), 0
        else:
            x1a, y1a, x2a, y2a = cx, cy, min(cx + w, s * 2), min(s * 2, cy + h)
            x1b, y1b = 0, 0
        canvas[y1a:y2a, x1a:x2a] = img[y1b:y1b + (y2a - y1a), x1b:x1b + (x2a - x1a)]
        if len(boxes):
            b = boxes.copy() * r
            b[:, [0, 2]] += x1a - x1b
            b[:, [1, 3]] += y1a - y1b
            out_boxes.append(b)
            out_labels.append(labels)

    boxes = np.concatenate(out_boxes) if out_boxes else np.zeros((0, 4), np.float32)
    labels = np.concatenate(out_labels) if out_labels else np.zeros((0,), np.int64)
    if len(boxes):
        np.clip(boxes, 0, s * 2, out=boxes)
        keep = (boxes[:, 2] - boxes[:, 0] > 2) & (boxes[:, 3] - boxes[:, 1] > 2)
        boxes, labels = boxes[keep], labels[keep]
    # centre-crop the 2s canvas back to s
    ox, oy = s // 2, s // 2
    canvas = canvas[oy:oy + s, ox:ox + s]
    if len(boxes):
        boxes[:, [0, 2]] -= ox
        boxes[:, [1, 3]] -= oy
        np.clip(boxes, 0, s, out=boxes)
        keep = (boxes[:, 2] - boxes[:, 0] > 2) & (boxes[:, 3] - boxes[:, 1] > 2)
        boxes, labels = boxes[keep], labels[keep]
    return canvas, boxes, labels


def random_scale(img: np.ndarray, boxes: np.ndarray, ratio: float, pad_value: int = 114,
                 min_side: float = 2.0, min_visible: float = 0.25):
    """Scale jitter at a fixed canvas size: zoom by ``s ~ U(1 - ratio, 1 + ratio)``.

    Zooming in crops a random window of the enlarged image; zooming out pastes the
    shrunken image at a random position on a ``pad_value`` canvas -- the output keeps
    the input's shape either way. Boxes follow; a box is kept if both sides are at
    least ``min_side`` px after clipping and at least ``min_visible`` of it is still
    inside the canvas. Returns ``(img, boxes, keep)`` with ``keep`` indexing the input
    boxes, so labels can be filtered alongside. YOLO's ``scale=0.5`` is ``ratio=0.5``.
    """
    s = random.uniform(1.0 - ratio, 1.0 + ratio)
    h, w = img.shape[:2]
    nh, nw = max(int(round(h * s)), 1), max(int(round(w * s)), 1)
    scaled = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    if s >= 1.0:
        ox, oy = random.randint(0, nw - w), random.randint(0, nh - h)
        out = np.ascontiguousarray(scaled[oy:oy + h, ox:ox + w])
        dx, dy = -ox, -oy
    else:
        ox, oy = random.randint(0, w - nw), random.randint(0, h - nh)
        out = np.full_like(img, pad_value)
        out[oy:oy + nh, ox:ox + nw] = scaled
        dx, dy = ox, oy
    boxes = np.asarray(boxes, dtype=np.float32).reshape(-1, 4)
    if not len(boxes):
        return out, boxes, np.zeros(0, dtype=np.int64)
    b = boxes * s
    b[:, [0, 2]] += dx
    b[:, [1, 3]] += dy
    area = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    c = b.copy()
    c[:, [0, 2]] = c[:, [0, 2]].clip(0, w)
    c[:, [1, 3]] = c[:, [1, 3]].clip(0, h)
    cw, ch = c[:, 2] - c[:, 0], c[:, 3] - c[:, 1]
    keep = np.nonzero((cw >= min_side) & (ch >= min_side)
                      & (cw * ch >= min_visible * np.maximum(area, 1e-6)))[0]
    return out, c[keep], keep


class Compose:
    def __init__(self, transforms: list):
        self.transforms = transforms

    def __call__(self, img, boxes):
        for t in self.transforms:
            img, boxes = t(img, boxes)
        return img, boxes


def build_train_transforms(size: int = 640, hsv: bool = True, flip: bool = True) -> Compose:
    ts = []
    if hsv:
        ts.append(RandomHSV())
    if flip:
        ts.append(RandomHFlip())
    ts.append(LetterBox(size))
    return Compose(ts)


def build_val_transforms(size: int = 640) -> Compose:
    return Compose([LetterBox(size)])
