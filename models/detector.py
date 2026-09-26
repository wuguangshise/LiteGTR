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
from losses.giou import CIoULoss, GIoULoss, bbox_iou_aligned
from losses.nwd import NWDLoss
from losses.qfl import QualityFocalLoss
from losses.token_consistency import TokenConsistencyLoss
from losses.token_routing import TokenRoutingLoss
from models.backbone.builder import build_backbone
from models.head.gfl_head import GFLHead
from models.neck.pyramid_projection import LocalCNNPath, PyramidProjection
from models.token.detail_enhance import RoutedDetailEnhance
from models.token.detail_inject import INJECT_AT, RoutedDetailInject
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
        p2_fusion = mc["neck"].get("p2_fusion", "add")
        if p2_fusion != "add" and not self.use_p2:
            raise ValueError("neck.p2_fusion changes the P3 -> P2 merge; it needs use_p2")
        self.neck = PyramidProjection(self.backbone.out_channels, dim, use_fpn=mc["neck"].get("use_fpn", True),
                                      p2_fusion=p2_fusion)
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
            # GT-centre supervision of the score maps -- see losses/token_routing.py.
            # Absent from a config means off, which reproduces runs made before it existed.
            rs = tk.get("routing_sup", {})
            self.routing_loss = (TokenRoutingLoss(loss_weight=rs.get("weight", 0.5),
                                                  sigma_ratio=rs.get("sigma_ratio", 1 / 6),
                                                  sigma_min=rs.get("sigma_min", 0.5))
                                 if rs.get("enabled", False) else None)
            self.scorer_no_decay = tk.get("scorer_no_decay", False)
        else:
            self.selector = self.mixer = self.writeback = self.ema_router = None
            self.routing_loss = None
            self.scorer_no_decay = False

        # Routed detail enhancement (models/token/detail_enhance.py): sharpen high-frequency
        # detail where the routing score map says objects are. Absent = off.
        de = mc.get("detail_enhance", {}) or {}
        self.detail_enhance = None
        if de.get("enabled", False):
            mask = de.get("mask", "score")
            if mask != "global" and not self.use_token:
                raise ValueError("detail_enhance.mask 'score'/'token' needs the token path; use 'global'")
            source = de.get("source", "P3")
            if mask == "score" and source not in self.token_levels:
                raise ValueError(f"detail_enhance.source {source!r} is not a token level {self.token_levels}")
            self.detail_enhance = RoutedDetailEnhance(
                dim, [lv for lv in de.get("levels", ["P2"]) if lv in self.levels],
                mask=mask, source=source, dilate=de.get("dilate", 3))

        # Routed detail injection (models/token/detail_inject.py): stride-2 stem detail
        # let back into P2 where the routing score map says objects are. Absent = off.
        di = mc.get("detail_inject", {}) or {}
        self.detail_inject = None
        self.inject_at = di.get("inject_at", "before_local")
        if di.get("enabled", False):
            mask = di.get("mask", "score")
            if not self.use_p2:
                raise ValueError("detail_inject writes into P2; it needs use_p2")
            if mask != "global" and not self.use_token:
                raise ValueError("detail_inject.mask 'score' needs the token path; use 'global'")
            source = di.get("source", "P3")
            if mask == "score" and source not in self.token_levels:
                raise ValueError(f"detail_inject.source {source!r} is not a token level {self.token_levels}")
            if self.inject_at not in INJECT_AT:
                raise ValueError(f"unknown detail_inject.inject_at {self.inject_at!r} (expected one of {INJECT_AT})")
            src = getattr(self.backbone, "stem_mid_channels", None)
            if src is None:
                raise ValueError("detail_inject needs the TinyNeXt conv stem (backbone.stem: conv)")
            self.detail_inject = RoutedDetailInject(src, dim, mask=mask, source=source,
                                                    dilate=di.get("dilate", 3),
                                                    detach_mask=di.get("detach_mask", True))

        hd = mc["head"]
        self.head = GFLHead(self.num_classes, dim, strides=self.strides,
                            stacked_convs=hd.get("stacked_convs", 2),
                            p2_stacked_convs=hd.get("p2_stacked_convs", 1),
                            reg_max=hd.get("reg_max", 16))

        lc = cfg.get("loss", {})
        self.qfl = QualityFocalLoss(loss_weight=lc.get("qfl_weight", 1.0))
        # IoU-type box term: "ciou" (default recipe) or "giou" (the original GFL choice).
        # Old configs that only set giou_weight keep training with GIoU.
        self.iou_type = lc.get("iou_type", "giou")
        iou_cls = {"ciou": CIoULoss, "giou": GIoULoss}[self.iou_type]
        self.box_iou = iou_cls(loss_weight=lc.get("iou_weight", lc.get("giou_weight", 2.0)))
        self.dfl = DistributionFocalLoss(loss_weight=lc.get("dfl_weight", 0.25))
        # NWD box term next to the IoU term; off in the default recipe (configs/ablation/ciou_nwd.yaml turns it on)
        nwd_w = lc.get("nwd_weight", 0.0)
        self.nwd = NWDLoss(constant=lc.get("nwd_constant", 12.8), loss_weight=nwd_w) if nwd_w > 0 else None
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
        stem_s2 = None
        if self.detail_inject is not None:
            c, stem_s2 = self.backbone(images, return_stem=True)
            feats = self.neck(c)
        else:
            feats = self.neck(self.backbone(images))
        fmap = dict(zip(self.levels, feats))
        # Routed detail injection before P2's local path: the routing map it needs comes
        # from the selector, so P2's local path waits for it. The local path is per level,
        # so running P2's later changes nothing else.
        late = "P2" if self.detail_inject is not None and self.inject_at == "before_local" else None
        if self.local_path is not None:
            for i, lv in enumerate(self.levels):
                if lv != late:
                    fmap[lv] = self.local_path.paths[i](fmap[lv])

        self._last_student_maps = self._last_teacher_maps = None
        score_maps = coords = None
        if self.use_token:
            tok_in = {lv: fmap[lv] for lv in self.token_levels}
            sel = self.selector(tok_in)
            self._last_student_maps = sel["score_maps"]
            if self.ema_router is not None and self.training:
                self._last_teacher_maps = self.ema_router.teacher_score_maps(
                    self._teacher_inputs(images, tok_in))
            mixed = self.mixer(sel["tokens"], sel["coords"], sel["level_ids"])
            score_maps, coords = sel["score_maps"], sel["coords"]
        if late is not None:
            fmap["P2"] = self.detail_inject(fmap["P2"], stem_s2, score_maps)
            if self.local_path is not None:
                fmap["P2"] = self.local_path.paths[self.levels.index("P2")](fmap["P2"])
        if self.use_token:
            fmap = self.writeback(fmap, mixed, coords)
        if self.detail_inject is not None and late is None:
            fmap["P2"] = self.detail_inject(fmap["P2"], stem_s2, score_maps)
        if self.detail_enhance is not None:
            fmap = self.detail_enhance(fmap, score_maps, coords)
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
        gts: list[torch.Tensor] = []
        pos_boxes, pos_tgt_boxes, pos_reg, pos_points, pos_strides, pos_w = [], [], [], [], [], []
        num_pos = 0
        for i in range(b):
            # targets are pinned by the DataLoader, so this copy can overlap compute
            gt_boxes = targets[i]["boxes"].to(device, non_blocking=True)
            gt_labels = targets[i]["labels"].to(device, non_blocking=True)
            gts.append(gt_boxes)
            res = self.assigner(cls[i].detach().sigmoid(), boxes[i].detach(), points,
                                gt_boxes, gt_labels,
                                point_strides=strides, reg_max=self.head.reg_max)
            # nonzero() has a data-dependent size, so it is the one host sync per image
            # that cannot be avoided. Branching on its numel() -- already on the host --
            # replaces the extra syncs that fg.any() and int(fg.sum()) used to add.
            idx = res["fg_mask"].nonzero(as_tuple=True)[0]
            if idx.numel():
                lab = res["assigned_labels"][idx]
                cls_targets[i, idx, lab] = res["assigned_ious"][idx]
                g = res["assigned_gt"][idx]
                pos_boxes.append(boxes[i][idx])
                pos_tgt_boxes.append(gt_boxes[g])
                pos_reg.append(reg[i][idx])
                pos_points.append(points[idx])
                pos_strides.append(strides[idx])
                pos_w.append(res["assigned_ious"][idx])
                num_pos += idx.numel()

        avg = cls_targets.sum().clamp_min(1.0)          # stays on device: no host sync
        losses = {"loss_qfl": self.qfl(cls.reshape(-1, self.num_classes),
                                       cls_targets.reshape(-1, self.num_classes), avg_factor=avg)}

        if num_pos > 0:
            pb = torch.cat(pos_boxes)
            tb = torch.cat(pos_tgt_boxes)
            pr = torch.cat(pos_reg)
            pp = torch.cat(pos_points)
            ps = torch.cat(pos_strides)
            w = torch.cat(pos_w)
            wsum = w.sum().clamp_min(1.0)
            losses[f"loss_{self.iou_type}"] = self.box_iou(pb, tb, weight=w, avg_factor=wsum)
            if self.nwd is not None:
                losses["loss_nwd"] = self.nwd(pb, tb, weight=w, avg_factor=wsum)

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
            losses[f"loss_{self.iou_type}"] = boxes.sum() * 0.0
            if self.nwd is not None:
                losses["loss_nwd"] = boxes.sum() * 0.0
            losses["loss_dfl"] = reg.sum() * 0.0

        if self._last_teacher_maps is not None:
            losses["loss_token"] = self.token_consistency(self._last_student_maps, self._last_teacher_maps)
        if self.routing_loss is not None and self._last_student_maps is not None:
            stride_of = dict(zip(self.levels, self.strides))
            losses["loss_route"] = self.routing_loss(self._last_student_maps, gts, stride_of)
        return losses

    # --------------------------------------------------------------- predict
    @torch.no_grad()
    def predict(self, images: torch.Tensor, score_thr: float = 0.02, nms_iou: float = 0.6,
                max_det: int = 500, pre_nms: int = 3000, agnostic: bool = False,
                containment: float | None = None, multi_label: bool = False) -> list[dict]:
        """Decoded, NMS-filtered detections per image.

        ``agnostic`` runs one NMS over all classes: on VisDrone a person is often
        predicted as both ``pedestrian`` and ``people``, and class-wise NMS keeps both.
        ``containment`` additionally drops a box when a higher-scoring kept box of the
        same class (any class if ``agnostic``) covers at least that fraction of its
        area: nested boxes on one tall object have IoU = small/large area, which falls
        below ``nms_iou`` and survives plain NMS. ``multi_label`` lets one location
        emit every class above ``score_thr`` instead of only its best one (mmyolo /
        Ultralytics evaluation). The evaluation protocol lives in the ``test:`` config
        block; tools/diagnose_predictions.py measures the alternatives.
        """
        cls_scores, bbox_preds, feats = self(images)
        cls, _, boxes, _, _ = self.head.decode(cls_scores, bbox_preds, feats)
        return [postprocess(s, bx, images.shape[-2:], score_thr, nms_iou, max_det, pre_nms,
                            agnostic, containment, multi_label)
                for s, bx in zip(cls.sigmoid(), boxes)]

    # ----------------------------------------------------------------- utils
    def ema_step(self) -> None:
        if self.ema_router is not None:
            self.ema_router.update(self.selector)

    def no_weight_decay(self) -> set[str]:
        """Parameter names the optimizer must not decay (``build_optimizer`` reads this).

        With ``scorer_no_decay`` the scorers' final 1x1 conv is exempt: its output
        IS the routing score, and decaying it drags every score toward the same
        value -- the flat-map collapse ``losses/token_routing.py`` describes.
        """
        if not self.scorer_no_decay or self.selector is None:
            return set()
        last = {id(sel.scorer.conv[-1].weight) for sel in self.selector.selectors.values()}
        return {n for n, p in self.named_parameters() if id(p) in last}

    def deploy_state_dict(self) -> dict:
        """State dict with the EMA teacher stripped -- what ONNX export uses."""
        return {k: v for k, v in self.state_dict().items() if not k.startswith("ema_router.")}


def postprocess(scores: torch.Tensor, boxes: torch.Tensor, hw: tuple[int, int],
                score_thr: float = 0.02, nms_iou: float = 0.6, max_det: int = 500,
                pre_nms: int = 3000, agnostic: bool = False,
                containment: float | None = None, multi_label: bool = False) -> dict:
    """One image: ``scores`` (L, C) probabilities, ``boxes`` (L, 4) xyxy -> detections."""
    h, w = hw
    if multi_label:
        loc, labels = (scores > score_thr).nonzero(as_tuple=True)
        s_max, bx = scores[loc, labels], boxes[loc].clone()
    else:
        s_max, labels = scores.max(-1)
        keep = s_max > score_thr
        s_max, labels, bx = s_max[keep], labels[keep], boxes[keep].clone()
    if s_max.numel() > pre_nms:
        topv, topi = s_max.topk(pre_nms)
        s_max, labels, bx = topv, labels[topi], bx[topi]
    bx[:, 0::2] = bx[:, 0::2].clamp(0, w)
    bx[:, 1::2] = bx[:, 1::2].clamp(0, h)
    groups = torch.zeros_like(labels) if agnostic else labels
    k = batched_nms(bx, s_max, groups, nms_iou)          # sorted by score, descending
    if containment is not None and k.numel() > 1:
        # A box can only be removed by a higher-scoring one, so the first ``max_det``
        # survivors depend only on a score-sorted prefix. Grow that prefix instead of
        # building the N x N matrix over every NMS survivor: under multi-label at
        # score 0.001 that is thousands of boxes per image -- minutes and GBs on CPU.
        m = min(k.numel(), 2 * max_det)
        while True:
            keep = _not_contained(bx[k[:m]], groups[k[:m]], containment)
            if int(keep.sum()) >= max_det or m == k.numel():
                break
            m = min(k.numel(), 2 * m)
        k = k[:m][keep]
    k = k[:max_det]
    return {"boxes": bx[k], "scores": s_max[k], "labels": labels[k]}


def _not_contained(bx: torch.Tensor, groups: torch.Tensor, thr: float) -> torch.Tensor:
    """Keep-mask over score-sorted boxes: drop j if an earlier box of its group covers
    >= ``thr`` of j's area. Earlier boxes that were themselves dropped still count --
    after NMS such chains are rare, and it keeps this a single vectorised pass."""
    area = ((bx[:, 2] - bx[:, 0]) * (bx[:, 3] - bx[:, 1])).clamp_min(1e-6)
    lt = torch.max(bx[:, None, :2], bx[None, :, :2])
    rb = torch.min(bx[:, None, 2:], bx[None, :, 2:])
    inter = (rb - lt).clamp_min(0).prod(-1)                  # (N, N)
    covered = inter / area[None, :]                          # [i, j]: share of j inside i
    earlier = torch.ones_like(covered, dtype=torch.bool).triu(1)
    same = groups[:, None] == groups[None, :]
    return ~((covered >= thr) & earlier & same).any(0)
