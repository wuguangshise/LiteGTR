"""Dataset factory -- dataset choice lives in YAML, never in code.

Previously this sat inside ``datasets/dronevehicle.py``, which meant building a
VisDrone dataset imported the DroneVehicle module. Separated so each dataset
module only knows about itself.
"""
from __future__ import annotations

from datasets.base import DetectionDataset


def build_dataset(cfg: dict, split: str, train: bool) -> DetectionDataset:
    d = cfg["data"]
    name = d["name"].lower()
    common = dict(
        root=d["root"],
        split=d[f"{split}_split"],
        img_size=d.get("img_size", 640),
        train=train,
        mosaic_prob=d.get("mosaic_prob", 0.5) if train else 0.0,
    )
    if name == "visdrone":
        from datasets.visdrone import VisDroneDataset

        # Training may paint ignored regions out; evaluation uses the untouched image
        # by default, as a COCO-json evaluation (RemDet, mmdet) does.
        mode = d.get("ignore_mode", "mask") if train else d.get("eval_ignore_mode", "drop")
        return VisDroneDataset(ignore_mode=mode, **common)
    if name == "dronevehicle":
        from datasets.dronevehicle import DroneVehicleDataset

        return DroneVehicleDataset(
            modality=d.get("modality", "rgb"),
            ann_dirname=d.get("ann_dirname"),
            conditions_file=d.get("conditions_file"),
            rebuild_cache=d.get("rebuild_cache", False),
            border=d.get("border", -1),
            **common,
        )
    raise ValueError(f"unknown dataset: {d['name']!r} (expected 'visdrone' or 'dronevehicle')")
