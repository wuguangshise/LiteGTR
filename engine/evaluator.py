"""Evaluation LOOP (metric definitions live in datasets/metrics.py, see P2-11).

Reports overall metrics plus a per-condition breakdown (day / night / dark),
which is the primary DroneVehicle result in this paper (docs/DESIGN.md P1-10).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from datasets.metrics import COCOMeanAP
from utils.plots import ConfusionMatrix, draw_predictions, plot_pr_curves


@torch.no_grad()
def postprocess_cfg(cfg: dict) -> dict:
    """``test:`` block of a config -> keyword arguments for ``LiteGTR.predict``.

    One place, so training-time validation, tools/val.py, tools/test.py and the
    submission script always post-process identically.
    """
    t = cfg.get("test", {}) or {}
    return {"score_thr": t.get("score_thr", 0.02), "nms_iou": t.get("nms_iou", 0.6),
            "max_det": t.get("max_det", 500), "agnostic": t.get("agnostic", False),
            "containment": t.get("containment"), "multi_label": t.get("multi_label", False),
            "pre_nms": t.get("pre_nms", 30000 if t.get("multi_label") else 3000)}


def vis_cfg(cfg: dict) -> dict:
    """``vis:`` block of a config -> keyword arguments for :func:`display_filter`.

    Drawing only. Absent from a config means the same defaults, so checkpoints and
    configs from before the block existed also draw one box per object.
    """
    v = cfg.get("vis", {}) or {}
    return {"score_thr": v.get("score_thr", 0.3), "nms_iou": v.get("nms_iou", 0.6),
            "agnostic": v.get("agnostic", True), "containment": v.get("containment", 0.8),
            "label": v.get("label", "class")}


def display_filter(boxes: np.ndarray, scores: np.ndarray, labels: np.ndarray,
                   score_thr: float = 0.3, nms_iou: float = 0.6, agnostic: bool = True,
                   containment: float | None = 0.8, **_) -> tuple[np.ndarray, ...]:
    """Reduce evaluation detections to what a figure should show: a score threshold,
    then NMS (class-agnostic by default: one box per person, not a pedestrian box
    AND a people box) and containment. Metrics never see the result."""
    from torchvision.ops import batched_nms

    from models.detector import _not_contained

    keep = scores >= score_thr
    boxes, scores, labels = boxes[keep], scores[keep], labels[keep]
    if len(scores) < 2:
        return boxes, scores, labels
    b = torch.as_tensor(boxes, dtype=torch.float32)
    s = torch.as_tensor(scores, dtype=torch.float32)
    g = torch.zeros(len(s), dtype=torch.long) if agnostic else torch.as_tensor(labels).long()
    k = batched_nms(b, s, g, nms_iou)                     # sorted by score, descending
    if containment is not None and k.numel() > 1:
        k = k[_not_contained(b[k], g[k], containment)]
    k = k.numpy()
    return boxes[k], scores[k], labels[k]


def evaluate(model, loader, device, classes: list[str], score_thr: float = 0.02,
             nms_iou: float = 0.6, max_det: int = 500, amp: bool = False,
             agnostic: bool = False, containment: float | None = None,
             multi_label: bool = False, pre_nms: int = 3000,
             per_condition: bool = True, desc: str = "val",
             save_dir: str | Path | None = None, num_vis: int = 16,
             vis: dict | None = None) -> tuple[dict, dict]:
    """Returns ``(overall_metrics, per_condition_metrics)``.

    When ``save_dir`` is given, also writes ``confusion_matrix.png``,
    ``pr_curve.png`` and ``val_predictions/`` -- the run artefacts listed in
    docs/DESIGN.md. ``vis`` (see :func:`vis_cfg`) filters what is drawn only.
    """
    vis = vis if vis is not None else vis_cfg({})
    model.eval()
    metric = COCOMeanAP(classes)
    cm = ConfusionMatrix(len(classes)) if save_dir else None
    vis_dir = None
    if save_dir:
        vis_dir = Path(save_dir) / "val_predictions"
        vis_dir.mkdir(parents=True, exist_ok=True)
    img_id = 0
    for images, targets in tqdm(loader, desc=desc, leave=False):
        images = images.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, enabled=amp and device.type == "cuda"):
            preds = model.predict(images, score_thr=score_thr, nms_iou=nms_iou, max_det=max_det,
                                  agnostic=agnostic, containment=containment,
                                  multi_label=multi_label, pre_nms=pre_nms)
        h, w = images.shape[-2:]
        for bi, (t, p) in enumerate(zip(targets, preds)):
            gt_b = t["boxes"].cpu().numpy()
            gt_l = t["labels"].cpu().numpy()
            dt_b = p["boxes"].float().cpu().numpy()
            dt_s = p["scores"].float().cpu().numpy()
            dt_l = p["labels"].cpu().numpy()
            metric.add(image_id=img_id, height=h, width=w,
                       condition=t.get("meta", {}).get("condition", "day"),
                       gt_boxes=gt_b, gt_labels=gt_l,
                       dt_boxes=dt_b, dt_scores=dt_s, dt_labels=dt_l)
            if cm is not None:
                cm.update(dt_b, dt_s, dt_l, gt_b, gt_l)
            if vis_dir is not None and img_id < num_vis:
                img = (images[bi].float().cpu().numpy().transpose(1, 2, 0)[:, :, ::-1] * 255)
                name = t.get("meta", {}).get("file_name", f"{img_id:06d}.jpg")
                try:
                    import cv2
                    vb, vs, vl = display_filter(dt_b, dt_s, dt_l, **vis)
                    cv2.imwrite(str(vis_dir / f"{img_id:04d}_{name}"),
                                draw_predictions(img.astype(np.uint8), vb, vs, vl, gt_b, classes,
                                                 conf=0.0, label=vis["label"]))
                except Exception:
                    pass
            img_id += 1

    overall = metric.evaluate()
    if save_dir:
        cm.plot(Path(save_dir) / "confusion_matrix.png", classes)
        curves = metric.pr_curves()
        if curves:
            plot_pr_curves(curves, Path(save_dir) / "pr_curve.png")
    by_cond: dict[str, dict] = {}
    if per_condition:
        conds = metric.available_conditions()
        if len(conds) > 1:
            for c in conds:
                by_cond[c] = metric.evaluate(conditions=[c])
    return overall, by_cond


@torch.no_grad()
def collect_token_stats(model, loader, device, max_batches: int = 5) -> dict:
    """Routing diagnostics logged to ``token_stats.csv``.

    * ``score_entropy``  -- normalised entropy of each level's spatial score map.
      Near 1.0 means the scorer is not discriminating at all (every position
      equally important, so the token budget is being spent at random); near 0
      means it has collapsed onto a handful of positions.
    * ``keep_ratio``     -- selected tokens / candidate positions, i.e. how
      aggressive the fixed budget actually is at this resolution.
    * ``ema_agreement``  -- correlation between student and EMA-teacher score
      maps. This is the quantity ``no_ema_routing`` is supposed to destabilise.
    """
    if not getattr(model, "use_token", False):
        return {}
    model.eval()
    ent: list[float] = []
    agree: list[float] = []
    keep: list[float] = []
    for bi, (images, _) in enumerate(loader):
        if bi >= max_batches:
            break
        images = images.to(device, non_blocking=True)

        # one pass only: cache the token-level inputs the selector actually saw
        feats = model.neck(model.backbone(images))
        if model.local_path is not None:
            feats = model.local_path(feats)
        fmap = dict(zip(model.levels, feats))
        tok_in = {lv: fmap[lv] for lv in model.token_levels}

        student = model.selector(tok_in)["score_maps"]
        teacher = model.ema_router.teacher_score_maps(tok_in) if model.ema_router else {}

        for lv, s in student.items():
            b = s.shape[0]
            p = torch.softmax(s.reshape(b, -1), dim=-1)
            e = -(p * p.clamp_min(1e-12).log()).sum(-1) / float(np.log(p.shape[-1]))
            ent.append(float(e.mean()))
            n_sel = model.selector.selectors[lv].num_tokens
            keep.append(n_sel / float(s.shape[-1] * s.shape[-2]))
            if lv in teacher:
                pair = torch.stack([s.reshape(-1).float(), teacher[lv].reshape(-1).float()])
                c = torch.corrcoef(pair)[0, 1]
                if torch.isfinite(c):
                    agree.append(float(c))

    out = {
        "score_entropy": float(np.mean(ent)) if ent else float("nan"),
        "keep_ratio": float(np.mean(keep)) if keep else float("nan"),
    }
    if agree:
        out["ema_agreement"] = float(np.mean(agree))
    return out
