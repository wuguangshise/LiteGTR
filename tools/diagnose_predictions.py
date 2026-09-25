"""Why are there so many boxes? Measure it on the real validation set.

Three questions, one pass over the data, no retraining:

1. **Assignment coverage** (no weights needed). How many GT boxes contain no
   candidate point and therefore can never become positives, under the plain
   inside-the-box rule and under STAL (``assigner.stal_size``). By object size.
2. **What each drawn box is.** Every prediction at or above ``--conf`` (the
   threshold ``val_predictions/`` draws with) is classified, greedily by score:
   correct, duplicate of an already-found object of the same class, right place
   but wrong class (VisDrone's pedestrian / people), poorly localised, background.
3. **Post-processing sweep.** The same raw predictions under several variants
   (score threshold, single/multi-label, NMS IoU, class-agnostic, containment):
   mAP, AP_S and boxes per image. Never needs retraining -- pick the variant here.

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
from engine.evaluator import to_original  # noqa: E402
from models.build import build_model, load_config  # noqa: E402
from models.detector import postprocess  # noqa: E402
from models.head.gfl_head import GFLHead  # noqa: E402

# (name, score_thr, nms_iou, multi_label, agnostic, containment)
VARIANTS = [
    ("old: 1-label, s>0.02, NMS 0.6", 0.02, 0.6, False, False, None),
    ("RemDet/Ultralytics: multi, s>0.001, NMS 0.7", 0.001, 0.7, True, False, None),
    ("multi, s>0.001, NMS 0.6", 0.001, 0.6, True, False, None),
    ("multi, s>0.001, NMS 0.7 + contain 0.8", 0.001, 0.7, True, False, 0.8),
    ("1-label, s>0.001, NMS 0.7", 0.001, 0.7, False, False, None),
    ("1-label agnostic NMS 0.6 + contain 0.8", 0.02, 0.6, False, True, 0.8),
]
SIZE_BINS = [(0, 4), (4, 8), (8, 16), (16, 32), (32, 1e9)]


def _bin_name(lo, hi):
    return f"{lo:g}-{hi:g}px" if hi < 1e9 else f">={lo:g}px"


# ------------------------------------------------------------------ coverage
def coverage(gt_boxes: torch.Tensor, points: torch.Tensor, strides: torch.Tensor,
             reg_max: int, stal_size: float = 0.0) -> torch.Tensor:
    """True for each GT with at least one candidate point. Mirrors the candidate rule in
    assigners/task_aligned_assigner.py (inside the box -- widened to ``stal_size`` if
    STAL is on -- and within the DFL regression range)."""
    if gt_boxes.numel() == 0:
        return torch.zeros(0, dtype=torch.bool)
    ltrb = torch.cat([points[:, None, :] - gt_boxes[None, :, :2],
                      gt_boxes[None, :, 2:] - points[:, None, :]], -1)
    if stal_size > 0:
        ctr = (gt_boxes[:, :2] + gt_boxes[:, 2:]) * 0.5
        half = (gt_boxes[:, 2:] - gt_boxes[:, :2]).clamp_min(stal_size) * 0.5
        sel = torch.cat([points[:, None, :] - (ctr - half)[None], (ctr + half)[None] - points[:, None, :]], -1)
    else:
        sel = ltrb
    ok = (sel.min(-1).values > 0) & ((ltrb / strides[:, None, None]).max(-1).values <= reg_max)
    return ok.any(0)


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
    stal = float(cfg.get("assigner", {}).get("stal_size", 0) or 8)
    max_det = int((cfg.get("test") or {}).get("max_det", 1000))
    cov_n = np.zeros(len(SIZE_BINS)); miss_plain = np.zeros(len(SIZE_BINS)); miss_stal = np.zeros(len(SIZE_BINS))
    cache = []                                  # per image: candidate scores/boxes + GT
    for images, targets in loader:
        if a.weights:
            cls_scores, bbox_preds, fts = model(images.to(device))
            cls, _, boxes, _, _ = model.head.decode(cls_scores, bbox_preds, fts)
            probs = cls.sigmoid()
        for i, t in enumerate(targets):
            gt_b, gt_l = t["boxes"], t["labels"]
            plain = coverage(gt_b, points, strides, reg_max)
            withs = coverage(gt_b, points, strides, reg_max, stal)
            side = ((gt_b[:, 2] - gt_b[:, 0]) * (gt_b[:, 3] - gt_b[:, 1])).clamp_min(0).sqrt()
            for k, (lo, hi) in enumerate(SIZE_BINS):
                m = (side >= lo) & (side < hi)
                cov_n[k] += int(m.sum())
                miss_plain[k] += int((m & ~plain).sum()); miss_stal[k] += int((m & ~withs).sum())
            if a.weights:
                s = probs[i]
                smax = s.max(-1).values
                keep = smax > 0.001
                s, bx, smax = s[keep], boxes[i][keep], smax[keep]
                if len(s) > 3000:                   # top locations; multi-label draws from these
                    top = smax.topk(3000).indices
                    s, bx = s[top], bx[top]
                cache.append((s.float().cpu(), bx.float().cpu(), gt_b.numpy(), gt_l.numpy(),
                              t.get("meta", {})))

    tot = max(cov_n.sum(), 1)
    lines = [f"split={a.split}  images={len(ds)}  input={size}px", "",
             f"[1] GT boxes with NO candidate point (never a positive)   plain rule | STAL {stal:g}px"]
    for k, (lo, hi) in enumerate(SIZE_BINS):
        n = int(cov_n[k])
        lines.append(f"    side {_bin_name(lo, hi):>9s}: {n:7d} GT ({100 * n / tot:5.1f}%)   "
                     f"uncovered {100 * miss_plain[k] / max(n, 1):5.1f}% | {100 * miss_stal[k] / max(n, 1):5.1f}%")
    lines.append(f"    total uncovered: {100 * miss_plain.sum() / tot:.1f}% | {100 * miss_stal.sum() / tot:.1f}% of all GT")

    rows = []
    if a.weights:
        h = w = size
        lines += ["", f"[2]+[3] post-processing variants (box types counted at score >= {a.conf})",
                  f"    {'variant':44s} {'mAP50:95':>8s} {'mAP50':>6s} {'AP_S':>6s} {'box/img':>7s} "
                  f"{'GT/img':>6s} {'correct':>7s} {'dup':>6s} {'wrongcls':>8s} {'loc':>6s} {'bg':>6s}"]
        n_img = len(cache)
        n_gt = sum(len(c[2]) for c in cache)      # GT at input size, as drawn
        for name, thr, iou, multi, agn, cont in VARIANTS:
            metric = COCOMeanAP(ds.classes)
            types = {"correct": 0, "duplicate": 0, "wrong_class": 0, "localisation": 0, "background": 0}
            drawn = 0
            for img_id, (s, bx, gt_b, gt_l, meta) in enumerate(cache):
                p = postprocess(s, bx, (h, w), thr, iou, max_det, 30000, agn, cont, multi)
                db, dsc, dl = p["boxes"].numpy(), p["scores"].numpy(), p["labels"].numpy()
                # mAP in original pixels (as engine/evaluator.py); box types at input size
                mh, mw, m_gb, m_gl, m_db = to_original(meta, (h, w), gt_b, gt_l, db)
                metric.add(img_id, mh, mw, "day", m_gb, m_gl, m_db, dsc, dl)
                drawn += int((dsc >= a.conf).sum())
                for k2, v in box_types(db, dsc, dl, gt_b, gt_l, a.conf).items():
                    types[k2] += v
            m = metric.evaluate()
            tt = max(sum(types.values()), 1)
            row = {"variant": name, "score_thr": thr, "nms_iou": iou, "multi_label": multi,
                   "agnostic": agn, "containment": cont,
                   "mAP50_95": m["mAP50_95"], "mAP50": m["mAP50"], "AP_small": m["AP_small"],
                   "AP_vt": m["AP_vt"], "AP_t": m["AP_t"],
                   "boxes_per_img": drawn / n_img, "gt_per_img": n_gt / n_img,
                   **{f"frac_{k2}": v / tt for k2, v in types.items()}}
            rows.append(row)
            lines.append(f"    {name:44s} {m['mAP50_95']:8.4f} {m['mAP50']:6.4f} {m['AP_small']:6.4f} "
                         f"{drawn / n_img:7.1f} {n_gt / n_img:6.1f} "
                         + " ".join(f"{100 * types[k2] / tt:6.1f}%" for k2 in
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
