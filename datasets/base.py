"""Shared detection-dataset machinery (mosaic, collate, tensor conversion)."""
from __future__ import annotations

import random
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from datasets.transforms import (LetterBox, build_train_transforms, build_val_transforms,
                                 mosaic4)


class DetectionDataset(Dataset):
    """Subclasses implement ``load_raw(index) -> (img, boxes_xyxy, labels, meta)``."""

    classes: list[str] = []

    def __init__(self, img_size: int = 640, train: bool = True, mosaic_prob: float = 0.5):
        self.img_size = img_size
        self.train = train
        self.mosaic_prob = mosaic_prob if train else 0.0
        self.tf = build_train_transforms(img_size) if train else build_val_transforms(img_size)
        # mosaic already emits an img_size canvas, so the letterbox step is skipped
        # for that branch -- everything else in the chain still applies.
        self.post_mosaic = [t for t in self.tf.transforms if not isinstance(t, LetterBox)]

    # --- to be provided by subclasses -------------------------------------
    def load_raw(self, index: int):
        raise NotImplementedError

    def __len__(self) -> int:
        raise NotImplementedError

    # ----------------------------------------------------------------------
    def __getitem__(self, index: int):
        if self.mosaic_prob > 0 and random.random() < self.mosaic_prob:
            idxs = [index] + [random.randrange(len(self)) for _ in range(3)]
            samples = []
            for i in idxs:
                img, boxes, labels, m = self.load_raw(i)
                samples.append((img, boxes, labels))
                if i == index:
                    meta = m
            img, boxes, labels = mosaic4(samples, self.img_size)
            for t in self.post_mosaic:
                img, boxes = t(img, boxes)
        else:
            img, boxes, labels, meta = self.load_raw(index)
            n_before = len(boxes)
            img, boxes = self.tf(img, boxes)
            if n_before and len(boxes):
                keep = (boxes[:, 2] - boxes[:, 0] > 1) & (boxes[:, 3] - boxes[:, 1] > 1)
                boxes, labels = boxes[keep], labels[keep]

        img = np.ascontiguousarray(img[:, :, ::-1].transpose(2, 0, 1))  # BGR->RGB, HWC->CHW
        tensor = torch.from_numpy(img).float().div_(255.0)
        target = {
            "boxes": torch.from_numpy(np.asarray(boxes, dtype=np.float32)).reshape(-1, 4),
            "labels": torch.from_numpy(np.asarray(labels, dtype=np.int64)).reshape(-1),
            "meta": meta or {},
        }
        return tensor, target


def collate_fn(batch):
    imgs = torch.stack([b[0] for b in batch])
    targets = [b[1] for b in batch]
    return imgs, targets


def imread(path: str | Path) -> np.ndarray:
    img = cv2.imread(str(path))
    if img is None:
        raise FileNotFoundError(f"failed to read image: {path}")
    return img
