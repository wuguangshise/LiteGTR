"""LiteGTR -- Lightweight Global-Token Refinement detector.

    RGB -> TinyNeXt -> C2..C5 -> projection(+FPN) -> P2..P5
                                       |
                        +--------------+--------------+
                        |                             |
                  Local CNN path            fixed-budget token selection
                        |                    (P3/P4/P5, local routing)
                        |                             |
                        |                        Token Mixer
                        |                             |
                        +---- geometry-aware writeback (residual) ----+
                                       |
                                   GFL head
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import batched_nms

from assigners.task_aligned_assigner import TaskAlignedAssigner
from losses.dfl import DistributionFocalLoss
from losses.giou import GIoULoss, bbox_iou_aligned
from losses.qfl import QualityFocalLoss
from losses.token_consistency import TokenConsistencyLoss
from models.backbone.builder import build_backbone
from models.head.gfl_head import GFLHead
from models.neck.pyramid_projection import LocalCNNPath, PyramidProjection
from models.token.ema_token_router import EMATokenRouter, bn_batch_stats_only, photometric_view
from models.token.geometric_writeback import MultiLevelWriteback
from models.token.token_mixer import TokenMixer
from models.token.token_selector import MultiLevelTokenSelector

LEVELS = ["P2", "P3", "P4", "P5"]


class LiteGTR(nn.Module):
    def __init__(self, cfg: dict):
        super().__init__()
        self.cfg = cfg
        mc = cfg["model"]
        self.num_classes = mc["num_classes"]
        self.use_p2 = mc.get("use_p2", True)
        self.levels = LEVELS if self.use_p2 else LEVELS[1:]
        self.strides = tuple(4 * (2 ** i) for i in range(4)) if self.use_p2 else (8, 16, 32)

        self.backbone = build_backbone(mc["backbone"], self.use_p2)

        dim = mc["neck"]["channels"]
        self.neck = PyramidProjection(self.backbone.out_channels, dim, use_fpn=mc["neck"].get("use_fpn", True))
        self.local_path = (LocalCNNPath(dim, len(self.levels), strides=self.strides)
                           if mc.get("use_local_cnn", True) else None)

        tk = mc["token"]
        self.use_token = tk.get("enabled", True)
        self.token_levels = [lv for lv in tk["levels"] if lv in self.levels]
        if self.use_token:
            self.selector = MultiLevelTokenSelector(dim, tk["budget"], tk["grids"], self.token_levels,
                                                    score_gate=tk.get("score_gate", True))
            self.mixer = TokenMixer(dim, num_heads=tk.get("num_heads", 1),
                                    mlp_ratio=tk.get("mlp_ratio", 2.0),
                                    num_layers=tk.get("mixer_layers", 1),
                                    num_levels=len(self.token_levels))
            wb_levels = tk.get("writeback_levels", self.token_levels)
            self.writeback = MultiLevelWriteback(dim, [lv for lv in wb_levels if lv in self.levels],
                                                 num_heads=tk.get("num_heads", 1),
                                                 mode=tk.get("writeback_mode", "geometric"))
            ema = tk.get("ema", {})
            self.ema_router = (EMATokenRouter(self.selector, ema["momentum"], ema.get("warmup_iters", 1000))
                               if ema.get("enabled", False) else None)
            # "photometric": teacher sees an illumination-perturbed view (the working design)
            # "same":        teacher sees the student's input -- inert, kept only as an ablation
            self.ema_view = ema.get("view", "photometric")
            assert self.ema_view in ("photometric", "same"), self.ema_view
            pv = ema.get("photometric", {})
            self.ema_view_cfg = dict(brightness=pv.get("brightness", 0.4),
                                     contrast=pv.get("contrast", 0.4),
                                     gamma=tuple(pv.get("gamma", (0.7, 1.5))),
                                     noise=pv.get("noise", 0.03))
        else:
            self.selector = self.mixer = self.writeback = self.ema_router = None

        hd = mc["head"]
        self.head = GFLHead(self.num_classes, dim, strides=self.strides,
                            stacked_convs=hd.get("stacked_convs", 2),
                            p2_stacked_convs=hd.get("p2_stacked_convs", 1),
                            reg_max=hd.get("reg_max", 16))

        lc = cfg.get("loss", {})
        self.qfl = QualityFocalLoss(loss_weight=lc.get("qfl_weight", 1.0))
        self.giou = GIoULoss(loss_weight=lc.get("giou_weight", 2.0))
        self.dfl = DistributionFocalLoss(loss_weight=lc.get("dfl_weight", 0.25))
        self.token_consistency = TokenConsistencyLoss(
            temperature=lc.get("token_temperature", 1.0),
            hard_ratio=lc.get("token_hard_ratio", 1.0),
            loss_weight=lc.get("token_weight", 0.5),
        )
        self.assigner = TaskAlignedAssigner(**cfg.get("assigner", {}))
        self._last_student_maps: dict | None = None
        self._last_teacher_maps: dict | None = None

    # ------------------------------------------------------------------ core
    def extract_feats(self, images: torch.Tensor) -> list[torch.Tensor]:
        feats = self.neck(self.backbone(images))
        if self.local_path is not None:
            feats = self.local_path(feats)
        fmap = dict(zip(self.levels, feats))

        self._last_student_maps = self._last_teacher_maps = None
        if self.use_token:
            tok_in = {lv: fmap[lv] for lv in self.token_levels}
            sel = self.selector(tok_in)
            self._last_student_maps = sel["score_maps"]
            if self.ema_router is not None and self.training:
                self._last_teacher_maps = self.ema_router.teacher_score_maps(
                    self._teacher_inputs(images, tok_in))
            mixed = self.mixer(sel["tokens"], sel["coords"], sel["level_ids"])
            fmap = self.writeback(fmap, mixed, sel["coords"])
        return [fmap[lv] for lv in self.levels]

    @torch.no_grad()
    def _teacher_inputs(self, images: torch.Tensor,
                        tok_in: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Token-level features for the EMA teacher.

        For the photometric view this re-runs backbone, neck and local path on a
        perturbed copy of the batch -- one extra forward, no backward. BatchNorm
        normalises with batch statistics but never writes its running buffers,
        so the perturbed batch cannot leak into validation statistics and no
        tensor saved by the student's forward is modified before backward().
        """
        if self.ema_view == "same":
            return tok_in
        xp = photometric_view(images, **self.ema_view_cfg)
        with bn_batch_stats_only(self.backbone, self.neck, self.local_path):
            feats = self.neck(self.backbone(xp))
            if self.local_path is not None:
                feats = self.local_path(feats)
        fmap = dict(zip(self.levels, feats))
        return {lv: fmap[lv] for lv in self.token_levels}

    def forward(self, images: torch.Tensor):
        feats = self.extract_feats(images)
        cls_scores, bbox_preds = self.head(feats)
        return cls_scores, bbox_preds, feats

    # ------------------------------------------------------------------ loss
    def loss(self, images: torch.Tensor, targets: list[dict]) -> dict:
        cls_scores, bbox_preds, feats = self(images)
        cls, reg, boxes, points, strides = self.head.decode(cls_scores, bbox_preds, feats)

        # AMP: compute the loss in fp32. Under autocast the head emits half, but
        # ground truth arrives from the DataLoader as float32, so the assigner's
        # IoU silently promotes and writing its result back into a half target
        # tensor raises. Casting here fixes that at the root, and QFL/DFL/GIoU are
        # numerically better behaved in fp32 anyway -- the standard AMP arrangement.
        cls, reg, boxes = cls.float(), reg.float(), boxes.float()
        points, strides = points.float(), strides.float()

        b = cls.shape[0]
        device = cls.device

        cls_targets = torch.zeros_like(cls)
        pos_boxes, pos_tgt_boxes, pos_reg, pos_points, pos_strides, pos_w = [], [], [], [], [], []
        num_pos = 0
        for i in range(b):
            gt_boxes = targets[i]["boxes"].to(device)
            gt_labels = targets[i]["labels"].to(device)
            res = self.assigner(cls[i].detach().sigmoid(), boxes[i].detach(), points,
                                gt_boxes, gt_labels,
                                point_strides=strides, reg_max=self.head.reg_max)
            fg = res["fg_mask"]
            if fg.any():
                idx = fg.nonzero(as_tuple=True)[0]
                lab = res["assigned_labels"][idx]
                cls_targets[i, idx, lab] = res["assigned_ious"][idx]
                g = res["assigned_gt"][idx]
                pos_boxes.append(boxes[i][idx])
                pos_tgt_boxes.append(gt_boxes[g])
                pos_reg.append(reg[i][idx])
                pos_points.append(points[idx])
                pos_strides.append(strides[idx])
                pos_w.append(res["assigned_ious"][idx])
                num_pos += int(fg.sum())

        avg = max(float(cls_targets.sum()), 1.0)
        losses = {"loss_qfl": self.qfl(cls.reshape(-1, self.num_classes),
                                       cls_targets.reshape(-1, self.num_classes), avg_factor=avg)}

        if num_pos > 0:
            pb = torch.cat(pos_boxes)
            tb = torch.cat(pos_tgt_boxes)
            pr = torch.cat(pos_reg)
            pp = torch.cat(pos_points)
            ps = torch.cat(pos_strides)
            w = torch.cat(pos_w)
            wsum = max(float(w.sum()), 1.0)
            losses["loss_giou"] = self.giou(pb, tb, weight=w, avg_factor=wsum)

            reg_max = self.head.reg_max
            tgt_dist = torch.stack([
                pp[:, 0] - tb[:, 0], pp[:, 1] - tb[:, 1],
                tb[:, 2] - pp[:, 0], tb[:, 3] - pp[:, 1],
            ], -1) / ps[:, None]
            tgt_dist = tgt_dist.clamp(0, reg_max - 0.01)
            losses["loss_dfl"] = self.dfl(
                pr.reshape(-1, reg_max + 1), tgt_dist.reshape(-1),
                weight=w[:, None].expand(-1, 4).reshape(-1), avg_factor=wsum * 4)
        else:
            losses["loss_giou"] = boxes.sum() * 0.0
            losses["loss_dfl"] = reg.sum() * 0.0

        if self._last_teacher_maps is not None:
            losses["loss_token"] = self.token_consistency(self._last_student_maps, self._last_teacher_maps)
        return losses

    # --------------------------------------------------------------- predict
    @torch.no_grad()
    def predict(self, images: torch.Tensor, score_thr: float = 0.02, nms_iou: float = 0.6,
                max_det: int = 500, pre_nms: int = 3000) -> list[dict]:
        cls_scores, bbox_preds, feats = self(images)
        cls, _, boxes, _, _ = self.head.decode(cls_scores, bbox_preds, feats)
        scores = cls.sigmoid()
        h, w = images.shape[-2:]
        results = []
        for i in range(images.shape[0]):
            s, bx = scores[i], boxes[i]
            s_max, labels = s.max(-1)
            keep = s_max > score_thr
            s_max, labels, bx = s_max[keep], labels[keep], bx[keep]
            if s_max.numel() > pre_nms:
                topv, topi = s_max.topk(pre_nms)
                s_max, labels, bx = topv, labels[topi], bx[topi]
            bx[:, 0::2] = bx[:, 0::2].clamp(0, w)
            bx[:, 1::2] = bx[:, 1::2].clamp(0, h)
            k = batched_nms(bx, s_max, labels, nms_iou)[:max_det]
            results.append({"boxes": bx[k], "scores": s_max[k], "labels": labels[k]})
        return results

    # ----------------------------------------------------------------- utils
    def ema_step(self) -> None:
        if self.ema_router is not None:
            self.ema_router.update(self.selector)

    def deploy_state_dict(self) -> dict:
        """State dict with the EMA teacher stripped -- what ONNX export uses."""
        return {k: v for k, v in self.state_dict().items() if not k.startswith("ema_router.")}
