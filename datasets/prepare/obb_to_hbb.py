"""Convert DroneVehicle oriented boxes to axis-aligned boxes.

Protocol note for the paper (docs/DESIGN.md P0-5 / P1-10): published
DroneVehicle numbers are OBB mAP and are NOT comparable to what this produces.
The circumscribed HBB of a tilted vehicle is strictly larger than the vehicle,
so IoU is looser.  This is fine -- but it must be stated, and every baseline has
to be retrained under the same HBB protocol (which the unified-protocol decision
in P1-6 already requires anyway).

Emits one ``<stem>.txt`` per image: ``cls x1 y1 x2 y2`` in pixels.
"""
from __future__ import annotations

import argparse
import xml.etree.ElementTree as ET
from pathlib import Path

CLASSES = ["car", "truck", "bus", "van", "freight_car"]
# the released annotations contain several spellings of the same class
ALIASES = {
    "feright_car": "freight_car", "feright car": "freight_car", "freight car": "freight_car",
    "truvk": "truck", "truck ": "truck", "*": None, "": None,
}


def norm_class(name: str) -> str | None:
    n = name.strip().lower().replace("-", "_")
    n = ALIASES.get(n, n)
    return n if n in CLASSES else None


def obj_to_hbb(obj: ET.Element) -> tuple[float, float, float, float] | None:
    poly = obj.find("polygon")
    if poly is not None:
        xs = [float(poly.find(f"x{i}").text) for i in range(1, 5)]
        ys = [float(poly.find(f"y{i}").text) for i in range(1, 5)]
        return min(xs), min(ys), max(xs), max(ys)
    rb = obj.find("robndbox")
    if rb is not None:
        import math
        cx, cy = float(rb.find("cx").text), float(rb.find("cy").text)
        ww, hh = float(rb.find("w").text), float(rb.find("h").text)
        ang = float(rb.find("angle").text)
        c, s = abs(math.cos(ang)), abs(math.sin(ang))
        bw, bh = ww * c + hh * s, ww * s + hh * c
        return cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2
    bb = obj.find("bndbox")
    if bb is not None:
        return (float(bb.find("xmin").text), float(bb.find("ymin").text),
                float(bb.find("xmax").text), float(bb.find("ymax").text))
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ann-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    a = ap.parse_args()
    ann, out = Path(a.ann_dir), Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    counts = {c: 0 for c in CLASSES}
    dropped = 0
    for xml in sorted(ann.glob("*.xml")):
        lines = []
        for obj in ET.parse(xml).getroot().iter("object"):
            name_node = obj.find("name")
            if name_node is None:
                continue
            cname = norm_class(name_node.text or "")
            box = obj_to_hbb(obj)
            if cname is None or box is None or box[2] <= box[0] or box[3] <= box[1]:
                dropped += 1
                continue
            counts[cname] += 1
            lines.append(f"{CLASSES.index(cname)} {box[0]:.2f} {box[1]:.2f} {box[2]:.2f} {box[3]:.2f}")
        (out / f"{xml.stem}.txt").write_text("\n".join(lines), encoding="utf-8")

    total = sum(counts.values())
    print(f"[obb2hbb] {total} boxes written to {out} ({dropped} dropped)")
    print("[obb2hbb] class distribution (note the imbalance -> 3 seeds are mandatory, P2-14):")
    for c, n in sorted(counts.items(), key=lambda kv: -kv[1]):
        print(f"           {c:14s} {n:8d}  {100.0 * n / max(total, 1):5.1f}%")


if __name__ == "__main__":
    main()
