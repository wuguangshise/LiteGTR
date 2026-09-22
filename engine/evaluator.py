"""Evaluation LOOP (metric definitions live in datasets/metrics.py, see P2-11).

Reports overall metrics plus a per-condition breakdown (day / night / dark),
which is the primary DroneVehicle result in this paper (docs/DESIGN.md P1-10).
"""
from __future__ import annotations

import numpy as np
import torch
from tqdm import tqdm

from datasets.metrics import COCOMeanAP


@torch.no_grad()
def evaluate(model, loader, device, classes: list[str], score_thr: float = 0.02,
             nms_iou: float = 0.6, max_det: int = 500, amp: bool = False,
             per_condition: bool = True, desc: str = "val") -> tuple[dict, dict]:
    model.eval()
    metric = COCOMeanAP(classes)
    img_id = 0
    for images, targets in tqdm(loader, desc=desc, leave=False):
        images = images.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, enabled=amp and device.type == "cuda"):
            preds = model.predict(images, score_thr=score_thr, nms_iou=nms_iou, max_det=max_det)
        h, w = images.shape[-2:]
        for t, p in zip(targets, preds):
            metric.add(
                image_id=img_id, height=h, width=w,
                condition=t.get("meta", {}).get("condition", "day"),
                gt_boxes=t["boxes"].cpu().numpy(), gt_labels=t["labels"].cpu().numpy(),
                dt_boxes=p["boxes"].float().cpu().numpy(),
                dt_scores=p["scores"].float().cpu().numpy(),
                dt_labels=p["labels"].cpu().numpy(),
            )
            img_id += 1

    overall = metric.evaluate()
    by_cond: dict[str, dict] = {}
    if per_condition:
        conds = metric.available_conditions()
        if len(conds) > 1:
            for c in conds:
                by_cond[c] = metric.evaluate(conditions=[c])
    return overall, by_cond


@torch.no_grad()
def collect_token_stats(model, loader, device, max_batches: int = 20) -> dict:
    """Routing diagnostics: how concentrated the score maps are and how closely
    the student tracks the EMA teacher.  Logged to ``token_stats.csv``."""
    if not getattr(model, "use_token", False):
        return {}
    model.eval()
    ent, agree, n = [], [], 0
    for images, _ in loader:
        if n >= max_batches:
            break
        images = images.to(device, non_blocking=True)
        model.extract_feats(images)
        s_maps = model._last_student_maps or {}
        for lv, s in s_maps.items():
            p = torch.softmax(s.reshape(s.shape[0], -1), dim=-1)
            e = -(p * p.clamp_min(1e-12).log()).sum(-1) / np.log(p.shape[-1])
            ent.append(float(e.mean()))
        if model.ema_router is not None:
            t_maps = model.ema_router.teacher_score_maps(
                {lv: f for lv, f in zip(model.levels, model.extract_feats(images))
                 if lv in model.token_levels})
            for lv in s_maps:
                if lv in t_maps:
                    a = torch.corrcoef(torch.stack([
                        s_maps[lv].reshape(-1), t_maps[lv].reshape(-1)]))[0, 1]
                    if a == a:
                        agree.append(float(a))
        n += 1
    out = {"score_entropy": float(np.mean(ent)) if ent else float("nan")}
    if agree:
        out["ema_agreement"] = float(np.mean(agree))
    return out
