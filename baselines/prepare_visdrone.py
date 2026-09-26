"""
把 LiteGTR 训练用的原始 VisDrone2019-DET 数据，转成各对比方法官方仓库读取的格式。

    同一批图片、同一套类别：VisDrone 类别 1-10 -> 0-9；类别 0（ignored regions）和
    11（others）的标注丢弃，和 LiteGTR 评估时的处理一致。图片不复制、不改动。

输出：
    * Ultralytics（YOLOv8n / YOLO11n）：<split>/labels/<图片名>.txt，归一化 cx cy w h。
      Ultralytics 按 images/ -> labels/ 找标签，所以写在每个 split 的 images 旁边
      （和 Ultralytics 官方 VisDrone.yaml 的转换方式相同）。
    * mmyolo（RemDet）和 DEIM：COCO json，category_id 为 0-9（DEIM 直接把
      category_id 当类别编号；mmdet 按类别名对应，两者都能用）。

只依赖标准库和 Pillow（读图片尺寸，只读文件头）。
"""
from __future__ import annotations

import json
from pathlib import Path

CLASSES = ["pedestrian", "people", "bicycle", "car", "van", "truck", "tricycle",
           "awning-tricycle", "bus", "motor"]
IMG_EXT = {".jpg", ".jpeg", ".png"}


def image_size(path: Path) -> tuple[int, int]:
    from PIL import Image

    with Image.open(path) as im:
        return im.size                       # (w, h)


def read_split(split_dir: Path) -> list[dict]:
    """[{file, width, height, boxes: [(x, y, w, h, cls0), ...]}] for one VisDrone split."""
    img_dir, ann_dir = split_dir / "images", split_dir / "annotations"
    if not img_dir.is_dir():
        raise FileNotFoundError(f"没有找到 {img_dir}")
    out = []
    for p in sorted(q for q in img_dir.iterdir() if q.suffix.lower() in IMG_EXT):
        w, h = image_size(p)
        boxes = []
        ann = ann_dir / f"{p.stem}.txt"
        if ann.exists():
            for line in ann.read_text(encoding="utf-8").splitlines():
                parts = [v for v in line.replace(" ", "").split(",") if v != ""]
                if len(parts) < 6:
                    continue
                x, y, bw, bh = (float(v) for v in parts[:4])
                cat = int(parts[5])
                if bw <= 0 or bh <= 0 or not 1 <= cat <= 10:
                    continue
                boxes.append((x, y, bw, bh, cat - 1))
        out.append({"file": p.name, "width": w, "height": h, "boxes": boxes})
    return out


def write_yolo_labels(split_dir: Path, records: list[dict]) -> Path:
    lab_dir = split_dir / "labels"
    lab_dir.mkdir(exist_ok=True)
    for r in records:
        w, h = r["width"], r["height"]
        lines = []
        for x, y, bw, bh, c in r["boxes"]:
            x1, y1 = max(x, 0.0), max(y, 0.0)                   # clip to the image
            x2, y2 = min(x + bw, w), min(y + bh, h)
            if x2 <= x1 or y2 <= y1:
                continue
            lines.append(f"{c} {(x1 + x2) / 2 / w:.6f} {(y1 + y2) / 2 / h:.6f} "
                         f"{(x2 - x1) / w:.6f} {(y2 - y1) / h:.6f}")
        (lab_dir / (Path(r["file"]).stem + ".txt")).write_text("\n".join(lines), encoding="utf-8")
    return lab_dir


def write_coco(records: list[dict], out_json: Path) -> Path:
    images, anns = [], []
    # image and annotation ids start at 1: pycocotools' COCOeval stores the matched GT id
    # per detection and treats 0 as "unmatched", so a GT with id 0 would score as a miss
    for i, r in enumerate(records, start=1):
        images.append({"id": i, "file_name": r["file"], "width": r["width"], "height": r["height"]})
        for x, y, bw, bh, c in r["boxes"]:
            anns.append({"id": len(anns) + 1, "image_id": i, "category_id": c,
                         "bbox": [x, y, bw, bh], "area": bw * bh, "iscrowd": 0})
    coco = {"images": images, "annotations": anns,
            "categories": [{"id": i, "name": n} for i, n in enumerate(CLASSES)]}
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(coco), encoding="utf-8")
    return out_json


def expected_paths(data_root: str | Path, train_split: str, val_split: str,
                   out_dir: str | Path) -> dict:
    """Where prepare() puts everything -- without writing anything."""
    data_root, out_dir = Path(data_root), Path(out_dir)
    paths = {"ultralytics_yaml": out_dir / "visdrone_ultralytics.yaml"}
    for key, split in (("train", train_split), ("val", val_split)):
        paths[f"{key}_images"] = data_root / split / "images"
        paths[f"{key}_labels"] = data_root / split / "labels"
        paths[f"{key}_coco"] = out_dir / f"visdrone_{key}_coco.json"
    return paths


def prepare(data_root: str | Path, train_split: str, val_split: str, out_dir: str | Path) -> dict:
    """Convert both splits once (skipped when the outputs exist); returns expected_paths()."""
    data_root = Path(data_root)
    paths = expected_paths(data_root, train_split, val_split, out_dir)
    for key, split in (("train", train_split), ("val", val_split)):
        if not paths[f"{key}_coco"].exists() or not paths[f"{key}_labels"].is_dir():
            recs = read_split(data_root / split)
            write_yolo_labels(data_root / split, recs)
            write_coco(recs, paths[f"{key}_coco"])
            print(f"  {split}: {len(recs)} 张图片，{sum(len(r['boxes']) for r in recs)} 个目标")
    yaml = paths["ultralytics_yaml"]
    names = "\n".join(f"  {i}: {n}" for i, n in enumerate(CLASSES))
    yaml.write_text(f"path: {data_root.as_posix()}\ntrain: {train_split}/images\n"
                    f"val: {val_split}/images\nnames:\n{names}\n", encoding="utf-8")
    return paths
