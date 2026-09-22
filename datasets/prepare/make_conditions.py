"""Generate ``conditions.txt`` (``<stem> day|night|dark``) for DroneVehicle.

``DroneVehicleDataset`` falls back to on-the-fly luminance tagging, but that
recomputes per epoch and cannot be hand-corrected. Writing the file once makes
the day/night/dark split explicit, reproducible and editable -- which matters
because the cross-illumination result (docs/DESIGN.md P1-10) is a headline
table, not a diagnostic.

    python -m datasets.prepare.make_conditions --img-dir PREP/val/rgb --out PREP/val/conditions.txt
"""
from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import cv2


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--img-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--dark-below", type=float, default=60.0)
    ap.add_argument("--night-below", type=float, default=110.0)
    a = ap.parse_args()

    img_dir = Path(a.img_dir)
    files = sorted(p for p in img_dir.iterdir() if p.suffix.lower() in {".jpg", ".png", ".jpeg"})
    if not files:
        raise SystemExit(f"no images under {img_dir}")

    counts: Counter = Counter()
    lines = []
    for p in files:
        img = cv2.imread(str(p))
        if img is None:
            continue
        v = float(cv2.cvtColor(img, cv2.COLOR_BGR2HSV)[..., 2].mean())
        cond = "dark" if v < a.dark_below else ("night" if v < a.night_below else "day")
        counts[cond] += 1
        lines.append(f"{p.stem} {cond} {v:.1f}")

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"[conditions] wrote {out} ({len(lines)} images)")
    for k, v in counts.most_common():
        print(f"  {k:6s} {v:6d}  {100.0 * v / max(len(lines), 1):5.1f}%")
    print("Thresholds are a starting point -- inspect a few images per bucket and "
          "edit the file if a bucket looks wrong.")


if __name__ == "__main__":
    main()
