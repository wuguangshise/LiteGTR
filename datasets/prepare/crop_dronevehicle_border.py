"""DEPRECATED -- the dataset now crops the border losslessly at load time.

Do not use this for training data. It decodes and RE-ENCODES every image, and on
a JPEG source that round-trip adds compression artefacts at exactly the scale a
12-px vehicle occupies. ``datasets/dronevehicle.py`` does the same crop as a
numpy slice, which is free and lossless, so there is nothing to gain here.

Kept only for the case where you need physically cropped copies for an external
tool. Training should point at the ORIGINAL download.

Strip DroneVehicle's 100-px white border and shift annotations accordingly.

WHY THIS EXISTS (docs/DESIGN.md P0-5, the single easiest way to silently ruin
this dataset): DroneVehicle ships 840x712 images with a ~100 px white margin on
each side; the true content is 640x512.  The margin was added so that rotated
boxes near the edge stay drawable, and **the annotation coordinates are relative
to the padded image**.  Training on the raw images wastes resolution budget,
makes P2 burn MACs on blank pixels, and puts your numbers out of step with every
published result.

Cropping without shifting the labels does not raise an error -- it just trains a
slightly wrong model.  Always follow this with ``tools/visualize_labels.py``.

Usage
-----
    python -m datasets.prepare.crop_dronevehicle_border \
        --src /data/DroneVehicle/train/rgb --dst /data/DroneVehicle_c/train/rgb \
        --ann-src /data/DroneVehicle/train/rgb_xml --ann-dst /data/DroneVehicle_c/train/rgb_xml
"""
from __future__ import annotations

import argparse
import xml.etree.ElementTree as ET
from pathlib import Path

import cv2
import numpy as np

BORDER = 100  # px, per side


def detect_border(img: np.ndarray, thresh: int = 240) -> int:
    """Return the measured white margin, so an already-cropped copy is a no-op."""
    gray = img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape[:2]
    top = 0
    while top < h // 3 and gray[top].mean() > thresh:
        top += 1
    return top


def shift_xml(xml_path: Path, out_path: Path, dx: int, dy: int, w: int, h: int) -> None:
    tree = ET.parse(xml_path)
    root = tree.getroot()
    size = root.find("size")
    if size is not None:
        size.find("width").text = str(w)
        size.find("height").text = str(h)
    for tag_x, tag_y in (("xmin", "ymin"), ("xmax", "ymax"), ("cx", "cy")):
        for node in root.iter():
            if node.find(tag_x) is not None and node.find(tag_y) is not None:
                node.find(tag_x).text = str(float(node.find(tag_x).text) - dx)
                node.find(tag_y).text = str(float(node.find(tag_y).text) - dy)
    for poly in root.iter("polygon"):     # x1,y1..x4,y4 form used by DroneVehicle
        for i in range(1, 5):
            nx, ny = poly.find(f"x{i}"), poly.find(f"y{i}")
            if nx is not None and ny is not None:
                nx.text = str(float(nx.text) - dx)
                ny.text = str(float(ny.text) - dy)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tree.write(out_path, encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", required=True)
    ap.add_argument("--ann-src", required=True)
    ap.add_argument("--ann-dst", required=True)
    ap.add_argument("--border", type=int, default=-1, help="-1 = auto-detect per image")
    a = ap.parse_args()

    src, dst = Path(a.src), Path(a.dst)
    ann_src, ann_dst = Path(a.ann_src), Path(a.ann_dst)
    dst.mkdir(parents=True, exist_ok=True)

    files = sorted([p for p in src.iterdir() if p.suffix.lower() in {".jpg", ".png", ".jpeg"}])
    if not files:
        raise SystemExit(f"no images under {src}")

    n_skipped = 0
    for p in files:
        img = cv2.imread(str(p))
        b = detect_border(img) if a.border < 0 else a.border
        if b < 10:
            n_skipped += 1
            b = 0
        h, w = img.shape[:2]
        crop = img[b:h - b, b:w - b] if b else img
        cv2.imwrite(str(dst / p.name), crop)
        xml = ann_src / (p.stem + ".xml")
        if xml.exists():
            shift_xml(xml, ann_dst / xml.name, b, b, crop.shape[1], crop.shape[0])

    print(f"[crop] {len(files)} images -> {dst}")
    if n_skipped:
        print(f"[crop] {n_skipped} images had no detectable border (already cropped?) -- copied as-is")
    print("[crop] NEXT: run tools/visualize_labels.py and eyeball ~20 samples before training.")


if __name__ == "__main__":
    main()
