"""RFLA: Gaussian Receptive Field based Label Assignment (Xu et al., ECCV 2022).

Ported from the official implementation (github.com/Chasel-Tsui/mmdet-rfla:
``HieAssigner`` + ``RFLA_FCOSHead.get_targets`` + ``BboxDistanceMetric(mode='kl')``).

Why it suits tiny objects
-------------------------
IoU- and "point inside the box" rules favour large objects: a 4-px car may
contain no feature point at all, so it gets no positives. RFLA instead models
every feature point by its (effective) receptive field -- a Gaussian centred on
the point with a per-level size -- and every GT box by a Gaussian with
``sigma = w/2, h/2``. The Receptive Field Distance is

    RFD(p, g) = 1 / (1 + KL(N_p || N_g))

which is defined for every point-GT pair, inside the box or not, and prefers the
level whose receptive field matches the object's size.

Hierarchical Label Assignment (HLA)
-----------------------------------
1. every GT takes its ``topk[0]`` highest-RFD points as positives;
2. receptive fields are shrunk by ``ratio`` and every GT takes ``topk[1]`` more
   among the points still negative -- the second stage mainly helps objects that
   lost the first round to a neighbour.

Adapting it to this GFL detector (the paper uses FCOS / RetinaNet / Faster R-CNN)
-------------------------------------------------------------------------------
* Soft classification targets follow GFL: a positive's QFL target is the IoU of
  its predicted box with its GT, and its box-loss weight is its detached maximum
  class score -- GFL's own recipe for a static assigner.
* A point is never assigned a GT it cannot regress: DFL expresses at most
  ``reg_max`` stride units, so pairs outside that range are excluded.
* A point claimed by several GTs keeps the one with the highest RFD (the
  reference keeps whichever GT index came last).
* The reference leaves non-top-k points with RFD >= 0.8 unlabelled (ignored);
  they are negatives here. That needs a receptive field almost identical to the
  GT and centred on it, i.e. it is rare.

Receptive-field sizes
---------------------
RFLA derives them from the backbone's theoretical receptive field (TRF) times a
fraction (1/2 for a P2 head). TinyNeXt's stacked 7x7 depthwise blocks give a
TRF of roughly 52 / 248 / 1024 / 1424 px at P2-P5; halved, every object under
~75 px would go to P2 and the token-refined levels P3-P5 would get almost no
positives. The effective receptive field of deep depthwise stacks grows far more
slowly than the TRF, so the default ``rf_sizes`` are the values RFLA itself uses
for its P2 head (ResNet-50-FPN TRF x 1/2 = 17.5 / 45.5 / 133.5 / 213.5 px),
which keep the level split RFLA was validated with. Override per stride in the
config; tools/diagnose_predictions.py reports the resulting split.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from losses.giou import bbox_iou

# RFLA's receptive field per stride for its P2 configuration
# (configs/rfla/aitodv2_fcos_r50_p2_rfla_kld_1x.py: gen_trf() * fraction 1/2).
DEFAULT_RF_SIZES = {4: 17.5, 8: 45.5, 16: 133.5, 32: 213.5}


def receptive_field_distance(points: torch.Tensor, rf: torch.Tensor, gt_boxes: torch.Tensor,
                             eps: float = 1e-6) -> torch.Tensor:
    """RFD = 1 / (1 + KL(N_rf || N_gt)) for every point-GT pair -> (L, G).

    ``rf`` (L,) is each point's receptive-field side in pixels; both Gaussians
    use sigma = side / 2 per axis, as in the reference ``mode='kl'``.
    """
    gw = (gt_boxes[:, 2] - gt_boxes[:, 0]).clamp_min(eps)[None]         # (1, G)
    gh = (gt_boxes[:, 3] - gt_boxes[:, 1]).clamp_min(eps)[None]
    gc = (gt_boxes[:, :2] + gt_boxes[:, 2:]) * 0.5                       # (G, 2)
    d = points[:, None, :] - gc[None]                                    # (L, G, 2)
    r = rf.clamp_min(eps)[:, None]                                       # (L, 1)
    kl = 0.5 * (r ** 2 / gw ** 2 + r ** 2 / gh ** 2
                + 4 * d[..., 0] ** 2 / gw ** 2 + 4 * d[..., 1] ** 2 / gh ** 2
                + torch.log(gw ** 2 / r ** 2) + torch.log(gh ** 2 / r ** 2) - 2)
    return 1.0 / (1.0 + kl)


class RFLAAssigner(nn.Module):
    def __init__(self, topk: tuple[int, int] = (3, 1), ratio: float = 0.9,
                 rf_sizes: dict | None = None, enforce_reg_range: bool = True, **_):
        super().__init__()
        self.topk = tuple(int(k) for k in topk)
        assert len(self.topk) == 2, "HLA has two stages: topk = [k1, k2]"
        self.ratio = ratio
        self.rf_sizes = {int(k): float(v) for k, v in (rf_sizes or DEFAULT_RF_SIZES).items()}
        self.enforce_reg_range = enforce_reg_range

    def _rf(self, point_strides: torch.Tensor) -> torch.Tensor:
        """Receptive-field side per point. torch.where per stride: no host sync."""
        rf = torch.zeros_like(point_strides, dtype=torch.float32)
        for s, v in self.rf_sizes.items():
            rf = torch.where(point_strides == s, rf.new_tensor(v), rf)
        if not getattr(self, "_checked", False):          # once: every stride has a size
            missing = set(point_strides.unique().long().tolist()) - set(self.rf_sizes)
            assert not missing, f"no receptive-field size for strides {missing}: {self.rf_sizes}"
            self._checked = True
        return rf

    @staticmethod
    def _stage(sim: torch.Tensor, k: int) -> torch.Tensor:
        """Each GT takes its k best points; a point claimed twice keeps its best GT."""
        num_pts, num_gt = sim.shape
        k = min(k, num_pts)
        _, idx = sim.topk(k, dim=0)                                      # (k, G)
        claim = torch.zeros_like(sim, dtype=torch.bool)
        claim.scatter_(0, idx, True)
        claim &= sim > 0                                                 # never an excluded pair
        best = torch.where(claim, sim, sim.new_full((), -1.0)).argmax(1)
        keep = torch.zeros_like(claim)
        keep[torch.arange(num_pts, device=sim.device), best] = True
        return claim & keep

    @torch.no_grad()
    def forward(self, pred_scores: torch.Tensor, pred_boxes: torch.Tensor,
                points: torch.Tensor, gt_boxes: torch.Tensor,
                gt_labels: torch.Tensor, point_strides: torch.Tensor | None = None,
                reg_max: int | None = None) -> dict:
        num_pts, num_gt = points.shape[0], gt_boxes.shape[0]
        device = points.device
        if num_gt == 0:
            return {
                "fg_mask": torch.zeros(num_pts, dtype=torch.bool, device=device),
                "assigned_gt": torch.zeros(num_pts, dtype=torch.long, device=device),
                "assigned_labels": torch.full((num_pts,), -1, dtype=torch.long, device=device),
                "assigned_ious": torch.zeros(num_pts, device=device),
                "reg_weights": torch.zeros(num_pts, device=device),
            }
        assert point_strides is not None, "RFLA needs the stride of every point"
        rf = self._rf(point_strides)
        valid = torch.ones(num_pts, num_gt, dtype=torch.bool, device=device)
        if self.enforce_reg_range and reg_max is not None:
            ltrb = torch.cat([points[:, None, :] - gt_boxes[None, :, :2],
                              gt_boxes[None, :, 2:] - points[:, None, :]], -1)
            valid = (ltrb / point_strides[:, None, None]).max(-1).values <= reg_max

        sim1 = receptive_field_distance(points, rf, gt_boxes) * valid       # RFD > 0 always
        pos1 = self._stage(sim1, self.topk[0])
        fg1 = pos1.any(1)

        sim2 = receptive_field_distance(points, rf * self.ratio, gt_boxes) * valid
        sim2 = sim2 * (~fg1)[:, None]                                        # stage 2: negatives only
        mask = pos1 | self._stage(sim2, self.topk[1])

        fg = mask.any(1)
        assigned_gt = mask.float().argmax(1)
        labels = torch.full((num_pts,), -1, dtype=torch.long, device=device)
        labels[fg] = gt_labels[assigned_gt[fg]]
        iou = bbox_iou(pred_boxes, gt_boxes).clamp_min(0)                    # (L, G)
        assigned_ious = torch.where(fg, iou.gather(1, assigned_gt[:, None]).squeeze(1),
                                    iou.new_zeros(()))
        reg_weights = torch.where(fg, pred_scores.max(1).values, pred_scores.new_zeros(()))
        return {
            "fg_mask": fg,
            "assigned_gt": assigned_gt,
            "assigned_labels": labels,
            "assigned_ious": assigned_ious,     # QFL target: IoU(pred, gt), as in GFL
            "reg_weights": reg_weights,         # box-loss weight: max class score, as in GFL
        }
