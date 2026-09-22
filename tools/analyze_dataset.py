"""Dataset statistics -- the empirical basis for two design arguments.

1. **Token budget (P0-3).** The concern is that ~56 tokens cannot represent a
   VisDrone scene holding ~53 objects on average and 300+ in dense images. This
   prints the real objects-per-image distribution for YOUR copy of the data, so
   the budget sweep starts from measurement rather than from a quoted average.
2. **Class imbalance (P0-5).** DroneVehicle is dominated by ``car``; the printed
   distribution is what makes 3-seed reporting non-negotiable.

Also reports the COCO small/medium/large split, which decides whether P2 is
earning its MAC cost at all.

    python tools/analyze_dataset.py --config configs/datasets/visdrone_rgb.yaml --split train
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from datasets.builder import build_dataset  # noqa: E402
from models.build import load_config  # noqa: E402
from utils.boxes import box_areas, size_bucket  # noqa: E402


def pct(x: np.ndarray, q: float) -> float:
    return float(np.percentile(x, q)) if len(x) else float("nan")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", nargs="+", required=True)
    ap.add_argument("--split", default="train")
    ap.add_argument("--limit", type=int, default=0, help="0 = whole split")
    ap.add_argument("--out", default=None, help="optional path for a text report")
    a = ap.parse_args()

    cfg = load_config(*a.config)
    ds = build_dataset(cfg, a.split, train=False)
    n_imgs = len(ds) if a.limit <= 0 else min(a.limit, len(ds))

    per_img: list[int] = []
    cls_counter: Counter = Counter()
    cond_counter: Counter = Counter()
    bucket_counter: Counter = Counter()
    all_w: list[float] = []
    all_h: list[float] = []

    for i in range(n_imgs):
        _, boxes, labels, meta = ds.load_raw(i)
        per_img.append(len(boxes))
        cond_counter[meta.get("condition", "day")] += 1
        for l in labels:
            cls_counter[ds.classes[int(l)]] += 1
        if len(boxes):
            b = np.asarray(boxes, dtype=np.float32).reshape(-1, 4)
            all_w.extend((b[:, 2] - b[:, 0]).tolist())
            all_h.extend((b[:, 3] - b[:, 1]).tolist())
            for ar in box_areas(b):
                bucket_counter[size_bucket(float(ar))] += 1

    arr = np.asarray(per_img, dtype=np.float32)
    w = np.asarray(all_w, dtype=np.float32)
    h = np.asarray(all_h, dtype=np.float32)
    total = int(arr.sum())

    L = []
    L.append(f"dataset      : {cfg['data']['name']}  split={a.split}")
    L.append(f"images       : {n_imgs}")
    L.append(f"objects      : {total}")
    L.append("")
    L.append("objects per image")
    L.append(f"  mean {arr.mean():8.1f}   median {np.median(arr):8.1f}   max {arr.max():8.0f}")
    L.append(f"  p75  {pct(arr,75):8.1f}   p90    {pct(arr,90):8.1f}   p99 {pct(arr,99):8.1f}")
    L.append(f"  images with 0 objects: {int((arr == 0).sum())}")
    L.append("")
    L.append("object size (original pixels)")
    L.append(f"  width  median {np.median(w):6.1f}  p10 {pct(w,10):6.1f}  p90 {pct(w,90):6.1f}")
    L.append(f"  height median {np.median(h):6.1f}  p10 {pct(h,10):6.1f}  p90 {pct(h,90):6.1f}")
    L.append("")
    L.append("COCO size buckets")
    for k in ("small", "medium", "large"):
        v = bucket_counter.get(k, 0)
        L.append(f"  {k:7s} {v:9d}  {100.0 * v / max(total, 1):5.1f}%")
    L.append("")
    L.append("class distribution")
    for name, c in cls_counter.most_common():
        L.append(f"  {name:18s} {c:9d}  {100.0 * c / max(total, 1):5.1f}%")
    if len(cond_counter) > 1:
        L.append("")
        L.append("illumination conditions")
        for name, c in cond_counter.most_common():
            L.append(f"  {name:8s} {c:7d} images  {100.0 * c / max(n_imgs, 1):5.1f}%")

    # --- the token-budget read-out -------------------------------------------
    tok = cfg.get("model", {}).get("token", {})
    if tok.get("budget"):
        budget = sum(tok["budget"].values())
        L.append("")
        L.append("token budget check (P0-3)")
        L.append(f"  configured budget      : {budget}")
        L.append(f"  mean objects / image   : {arr.mean():.1f}")
        L.append(f"  p90 objects / image    : {pct(arr, 90):.1f}")
        ratio = budget / max(arr.mean(), 1e-6)
        L.append(f"  tokens per mean object : {ratio:.2f}")
        if ratio < 1.0:
            L.append("  -> FEWER tokens than objects on a typical image. If the "
                     "no_global_token ablation")
            L.append("     barely moves, this is why. Sweep the budget before the "
                     "main experiments.")

    text = "\n".join(L)
    print(text)
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(text, encoding="utf-8")
        print(f"\nwritten {a.out}")


if __name__ == "__main__":
    main()
