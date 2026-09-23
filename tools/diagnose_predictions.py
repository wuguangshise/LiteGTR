"""Why are there so many boxes? Measure it on the real validation set.

Three questions, one pass over the data, no retraining:

1. **Assignment coverage** (no weights needed). How many GT boxes contain no
   candidate point and therefore can never become positives -- the defect that
   ``assigner.tiny_fallback`` fixes. Reported by object size at the network input.
2. **What each drawn box is.** Every prediction at or above ``--conf`` (the
   threshold ``val_predictions/`` draws with) is classified, greedily by score:
   correct, duplicate of an already-found object of the same class, right place
   but wrong class (VisDrone's pedestrian / people), poorly localised, background.
3. **Post-processing sweep.** The same raw predictions under several NMS variants
   (IoU threshold, class-agnostic, containment suppression): mAP, AP_S and boxes
   per image. Post-processing never needs retraining -- pick the variant here.

    python tools/diagnose_predictions.py --config configs/datasets/visdrone_rgb.yaml \\
        configs/models/model_main.yaml --weights runs/train/main/weights/best.pt \\
        --data-root D:/dataset/VisDrone/LiteGTR --out runs/diagnose

    # coverage only, before any model is trained:
    python tools/diagnose_predictions.py --config configs/datasets/visdrone_rgb.yaml \\
        configs/models/model_main.yaml --data-root D:/dataset/VisDrone/LiteGTR
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from datasets.base import collate_fn  # noqa: E402
from datasets.metrics import COCOMeanAP  # noqa: E402
from models.build import build_model, load_config  # noqa: E402
from models.detector import postprocess  # noqa: E402
from models.head.gfl_head import GFLHead  # noqa: E402

# (name, nms_iou, agnostic, containment)
VARIANTS = [
    ("class-wise NMS 0.6 (current)", 0.6, False, None),
    ("class-wise NMS 0.5", 0.5, False, None),
    ("class-wise NMS 0.7", 0.7, False, None),
    ("class-agnostic NMS 0.6", 0.6, True, None),
    ("class-wise 0.6 + containment 0.8", 0.6, False, 0.8),
    ("class-agnostic 0.6 + containment 0.8", 0.6, True, 0.8),
]
SIZE_BINS = [(0, 4), (4, 8), (8, 16), (16, 32), (32, 1e9)]


def _bin_name(lo, hi):
    return f"{lo:g}-{hi:g}px" if hi < 1e9 else f">={lo:g}px"


# ------------------------------------------------------------------ coverage
def coverage(gt_boxes: torch.Tensor, points: torch.Tensor, strides: torch.Tensor,
             reg_max: int) -> torch.Tensor:
    """True for each GT that has at least one candidate point (inside, in range)."""
    if gt_boxes.numel() == 0:
        return torch.zeros(0, dtype=torch.bool)
    lt = points[:, None, :] - gt_boxes[None, :, :2]
    rb = gt_boxes[None, :, 2:] - points[:, None, :]
    ltrb = torch.cat([lt, rb], -1)
    inside = (ltrb.min(-1).values > 0) & ((ltrb / strides[:, None, None]).max(-1).values <= reg_max)
    return inside.any(0)


# ------------------------------------------------------------------ box types
def box_types(dt_b, dt_s, dt_l, gt_b, gt_l, conf: float, iou_thr: float = 0.5) -> dict:
    """Classify predictions >= conf, highest score first."""
    out = {"correct": 0, "duplicate": 0, "wrong_class": 0, "localisation": 0, "background": 0}
    order = np.argsort(-dt_s)
    order = order[dt_s[order] >= conf]
    if len(order) == 0:
        return out
    if len(gt_b) == 0:
        out["background"] += len(order)
        return out
    gt_b_t = torch.as_tensor(gt_b, dtype=torch.float32)
    ious = _iou(torch.as_tensor(dt_b[order], dtype=torch.float32), gt_b_t).numpy()  # (D, G)
    matched = np.zeros(len(gt_b), bool)
    for d, j in enumerate(order):
        same = gt_l == dt_l[j]
        iou_same = np.where(same, ious[d], 0.0)
        g = int(iou_same.argmax())
        if iou_same[g] >= iou_thr and not matched[g]:
            matched[g] = True
            out["correct"] += 1
        elif iou_same[g] >= iou_thr:
            out["duplicate"] += 1
        elif np.where(~same, ious[d], 0.0).max(initial=0.0) >= iou_thr:
            out["wrong_class"] += 1
        elif ious[d].max(initial=0.0) >= 0.1:
            out["localisation"] += 1
        else:
            out["background"] += 1
    return out


def _iou(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    from losses.giou import bbox_iou
    return bbox_iou(a, b)


# ------------------------------------------------------------------ main
@torch.no_grad()
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", nargs="+", required=True, help="dataset config + model config")
    ap.add_argument("--weights", default=None, help="checkpoint; omit to run the coverage check only")
    ap.add_argument("--data-root", default=None)
    ap.add_argument("--split", default="val")
    ap.add_argument("--conf", type=float, default=0.25, help="threshold val_predictions/ draws with")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--raw", action="store_true", help="raw weights instead of EMA")
    ap.add_argument("--out", default="runs/diagnose")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()

    from datasets.builder import build_dataset

    cfg = load_config(*a.config)
    if a.data_root:
        cfg["data"]["root"] = a.data_root
    ds = build_dataset(cfg, a.split, train=False)
    cfg["model"]["num_classes"] = len(ds.classes)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device(a.device)

    model = build_model(cfg)
    if a.weights:
        ck = torch.load(a.weights, map_location="cpu", weights_only=False)
        sd = ck["model"] if (a.raw or not ck.get("model_ema")) else ck["model_ema"]
        model.load_state_dict(sd, strict=False)
    model.to(device).eval()
    size = cfg["data"].get("img_size", 640)
    feats = [torch.zeros(1, 1, size // s, size // s) for s in model.strides]
    points, strides = GFLHead.make_points(feats, model.strides, "cpu", torch.float32)
    reg_max = model.head.reg_max

    loader = DataLoader(ds, batch_size=a.batch, shuffle=False, collate_fn=collate_fn, num_workers=0)
    cov_n = np.zeros(len(SIZE_BINS)); cov_miss = np.zeros(len(SIZE_BINS))
    cache = []                                  # per image: candidate scores/boxes + GT
    for images, targets in loader:
        if a.weights:
            cls_scores, bbox_preds, fts = model(images.to(device))
            cls, _, boxes, _, _ = model.head.decode(cls_scores, bbox_preds, fts)
            probs = cls.sigmoid()
        for i, t in enumerate(targets):
            gt_b, gt_l = t["boxes"], t["labels"]
            has = coverage(gt_b, points, strides, reg_max)
            side = ((gt_b[:, 2] - gt_b[:, 0]) * (gt_b[:, 3] - gt_b[:, 1])).clamp_min(0).sqrt()
            for k, (lo, hi) in enumerate(SIZE_BINS):
                m = (side >= lo) & (side < hi)
                cov_n[k] += int(m.sum()); cov_miss[k] += int((m & ~has).sum())
            if a.weights:
                s = probs[i]
                keep = s.max(-1).values > 0.02
                s, bx = s[keep], boxes[i][keep]
                if len(s) > 3000:
                    top = s.max(-1).values.topk(3000).indices
                    s, bx = s[top], bx[top]
                cache.append((s.float().cpu(), bx.float().cpu(), gt_b.numpy(), gt_l.numpy()))

    lines = [f"split={a.split}  images={len(ds)}  input={size}px", "",
             "[1] GT boxes with NO candidate point (never a positive without assigner.tiny_fallback)"]
    for k, (lo, hi) in enumerate(SIZE_BINS):
        n = int(cov_n[k])
        lines.append(f"    side {_bin_name(lo, hi):>9s}: {n:7d} GT ({100 * n / max(cov_n.sum(), 1):5.1f}%)"
                     f"   uncovered {100 * cov_miss[k] / max(n, 1):5.1f}%")
    lines.append(f"    total uncovered: {100 * cov_miss.sum() / max(cov_n.sum(), 1):.1f}% of all GT")

    rows = []
    if a.weights:
        h = w = size
        lines += ["", f"[2]+[3] post-processing variants (boxes drawn at score >= {a.conf})",
                  f"    {'variant':38s} {'mAP50:95':>8s} {'mAP50':>6s} {'AP_S':>6s} {'box/img':>7s} "
                  f"{'GT/img':>6s} {'correct':>7s} {'dup':>6s} {'wrongcls':>8s} {'loc':>6s} {'bg':>6s}"]
        n_img = len(cache)
        n_gt = sum(len(c[2]) for c in cache)
        for name, iou, agn, cont in VARIANTS:
            metric = COCOMeanAP(ds.classes)
            types = {"correct": 0, "duplicate": 0, "wrong_class": 0, "localisation": 0, "background": 0}
            drawn = 0
            for img_id, (s, bx, gt_b, gt_l) in enumerate(cache):
                p = postprocess(s, bx, (h, w), 0.02, iou, 500, 3000, agn, cont)
                db, dsc, dl = p["boxes"].numpy(), p["scores"].numpy(), p["labels"].numpy()
                metric.add(img_id, h, w, "day", gt_b, gt_l, db, dsc, dl)
                drawn += int((dsc >= a.conf).sum())
                for k2, v in box_types(db, dsc, dl, gt_b, gt_l, a.conf).items():
                    types[k2] += v
            m = metric.evaluate()
            tot = max(sum(types.values()), 1)
            row = {"variant": name, "nms_iou": iou, "agnostic": agn, "containment": cont,
                   "mAP50_95": m["mAP50_95"], "mAP50": m["mAP50"], "AP_small": m["AP_small"],
                   "boxes_per_img": drawn / n_img, "gt_per_img": n_gt / n_img,
                   **{f"frac_{k2}": v / tot for k2, v in types.items()}}
            rows.append(row)
            lines.append(f"    {name:38s} {m['mAP50_95']:8.4f} {m['mAP50']:6.4f} {m['AP_small']:6.4f} "
                         f"{drawn / n_img:7.1f} {n_gt / n_img:6.1f} "
                         + " ".join(f"{100 * types[k2] / tot:6.1f}%" for k2 in
                                    ("correct", "duplicate", "wrong_class", "localisation", "background")))
        best = max(rows, key=lambda r: r["mAP50_95"])
        lines += ["", f"    best mAP50:95: {best['variant']}",
                  "    dup = same object, same class, found again; wrongcls = right place, other class"]
        with open(out / "postprocess_sweep.csv", "w", newline="", encoding="utf-8") as f:
            wr = csv.DictWriter(f, fieldnames=list(rows[0]))
            wr.writeheader()
            wr.writerows(rows)

    text = "\n".join(lines)
    print(text)
    (out / "diagnosis.txt").write_text(text + "\n", encoding="utf-8")
    print(f"\nwritten to {out}/")


if __name__ == "__main__":
    main()
