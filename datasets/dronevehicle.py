"""DroneVehicle (RGB), HBB protocol, with day/night condition tagging.

Before first use run, in order:
    1. ``datasets.prepare.crop_dronevehicle_border``  (P0-5 (1): 100-px white margin)
    2. ``datasets.prepare.obb_to_hbb``                (P0-5 (2): OBB -> circumscribed HBB)
    3. ``tools/visualize_labels.py``                  (confirm alignment by eye)

Condition tagging (P1-10)
-------------------------
The paper's second dataset is positioned as a **cross-illumination robustness**
check, not a leaderboard comparison -- HBB numbers are not comparable to
published OBB results anyway.  Each image is tagged ``day`` / ``night`` / ``dark``
so the evaluator can report per-condition mAP.

Tags come from ``conditions.txt`` (``<stem> <condition>`` per line) when present.
Otherwise they are derived from mean luminance with the thresholds below, which
is reproducible and good enough for grouping; the thresholds are configurable
and the chosen source is printed at construction so it lands in the log.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from datasets.base import DetectionDataset, imread

CLASSES = ["car", "truck", "bus", "van", "freight_car"]
DEFAULT_THRESHOLDS = (60.0, 110.0)   # mean V: < 60 -> dark, < 110 -> night, else day


class DroneVehicleDataset(DetectionDataset):
    classes = CLASSES

    def __init__(self, root: str, split: str = "train", img_size: int = 640,
                 train: bool = True, mosaic_prob: float = 0.5, modality: str = "rgb",
                 label_dirname: str = "hbb_labels", thresholds: tuple[float, float] = DEFAULT_THRESHOLDS,
                 conditions_file: str | None = None):
        super().__init__(img_size, train, mosaic_prob)
        assert modality in ("rgb", "ir"), "modality must be 'rgb' or 'ir'"
        self.root = Path(root)
        self.modality = modality
        self.thresholds = thresholds
        base = self.root / split
        self.img_dir = base / modality
        self.lbl_dir = base / label_dirname
        if not self.img_dir.is_dir():
            raise FileNotFoundError(f"missing {self.img_dir} -- did you run the prepare scripts?")
        self.images = sorted(p for p in self.img_dir.iterdir() if p.suffix.lower() in {".jpg", ".png"})

        cf = Path(conditions_file) if conditions_file else base / "conditions.txt"
        self.conditions: dict[str, str] = {}
        if cf.exists():
            for line in cf.read_text(encoding="utf-8").splitlines():
                parts = line.split()
                if len(parts) >= 2:
                    self.conditions[parts[0]] = parts[1]
            print(f"[DroneVehicle] conditions from file: {cf} ({len(self.conditions)} entries)")
        else:
            print(f"[DroneVehicle] {cf} not found -- tagging day/night/dark by luminance "
                  f"(thresholds={thresholds})")

    def __len__(self) -> int:
        return len(self.images)

    def _condition(self, stem: str, img: np.ndarray) -> str:
        if stem in self.conditions:
            return self.conditions[stem]
        v = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)[..., 2].mean()
        dark_t, night_t = self.thresholds
        return "dark" if v < dark_t else ("night" if v < night_t else "day")

    def load_raw(self, index: int):
        p = self.images[index]
        img = imread(p)
        if self.modality == "ir" and img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        boxes, labels = [], []
        lbl = self.lbl_dir / f"{p.stem}.txt"
        if lbl.exists():
            for line in lbl.read_text(encoding="utf-8").splitlines():
                parts = line.split()
                if len(parts) != 5:
                    continue
                c, x1, y1, x2, y2 = int(parts[0]), *[float(v) for v in parts[1:]]
                if x2 > x1 and y2 > y1:
                    boxes.append([x1, y1, x2, y2])
                    labels.append(c)
        boxes = np.asarray(boxes, dtype=np.float32).reshape(-1, 4)
        labels = np.asarray(labels, dtype=np.int64).reshape(-1)
        return img, boxes, labels, {"image_id": index, "file_name": p.name,
                                    "ori_shape": img.shape[:2],
                                    "condition": self._condition(p.stem, img)}


def build_dataset(cfg: dict, split: str, train: bool):
    """Factory used by tools/train.py -- keeps dataset choice in YAML."""
    d = cfg["data"]
    name = d["name"].lower()
    common = dict(root=d["root"], split=d[f"{split}_split"], img_size=d.get("img_size", 640),
                  train=train, mosaic_prob=d.get("mosaic_prob", 0.5) if train else 0.0)
    if name == "visdrone":
        from datasets.visdrone import VisDroneDataset
        return VisDroneDataset(ignore_mode=d.get("ignore_mode", "mask"), **common)
    if name == "dronevehicle":
        return DroneVehicleDataset(modality=d.get("modality", "rgb"),
                                   label_dirname=d.get("label_dirname", "hbb_labels"), **common)
    raise ValueError(f"unknown dataset: {d['name']}")
