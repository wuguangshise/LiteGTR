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
def evaluate(model, loader, device, classes: list[str], score_thr: float = 0.02,
             nms_iou: float = 0.6, max_det: int = 500, amp: bool = False,
             per_condition: bool = True, desc: str = "val",
             save_dir: str | Path | None = None, num_vis: int = 16) -> tuple[dict, dict]:
    """Returns ``(overall_metrics, per_condition_metrics)``.

    When ``save_dir`` is given, also writes ``confusion_matrix.png``,
    ``pr_curve.png`` and ``val_predictions/`` -- the run artefacts listed in
    docs/DESIGN.md.
    """
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
            preds = model.predict(images, score_thr=score_thr, nms_iou=nms_iou, max_det=max_det)
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
                    cv2.imwrite(str(vis_dir / f"{img_id:04d}_{name}"),
                                draw_predictions(img.astype(np.uint8), dt_b, dt_s, dt_l,
                                                 gt_b, classes))
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
