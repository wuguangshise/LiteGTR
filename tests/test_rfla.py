"""RFLA (ECCV'22) label assignment, ported into the GFL detector."""
import torch

from assigners import build_assigner
from assigners.rfla_assigner import RFLAAssigner, receptive_field_distance
from assigners.task_aligned_assigner import TaskAlignedAssigner
from models.build import build_model, load_config
from models.head.gfl_head import GFLHead

STRIDES = (4, 8, 16, 32)


def _points(size=160):
    feats = [torch.zeros(1, 1, size // s, size // s) for s in STRIDES]
    return GFLHead.make_points(feats, STRIDES, "cpu", torch.float32)


def _run(gt, labels=None):
    pts, st = _points()
    n = pts.shape[0]
    labels = labels if labels is not None else torch.zeros(len(gt), dtype=torch.long)
    pred = torch.cat([pts - 4, pts + 4], 1)
    return RFLAAssigner()(torch.full((n, 3), 0.1), pred, pts, gt, labels, point_strides=st, reg_max=16), pts, st


def test_rfd_is_one_for_identical_gaussians_and_prefers_matching_size():
    p = torch.tensor([[50.0, 50.0]])
    gt = torch.tensor([[40.0, 40.0, 60.0, 60.0]])
    assert torch.allclose(receptive_field_distance(p, torch.tensor([20.0]), gt), torch.ones(1, 1))
    near = receptive_field_distance(p, torch.tensor([17.5]), gt)
    far = receptive_field_distance(p, torch.tensor([213.5]), gt)
    assert near > far


def test_tiny_gt_between_grid_points_gets_positives():
    """A 3-px box containing no stride-4 point centre: RFLA still assigns it."""
    # point centres: P2 4i+2, P3 8i+4, P4 16i+8, P5 32i+16 -> none lies in [22.3, 23.9]
    gt = torch.tensor([[22.3, 22.3, 23.9, 23.9]])
    res, pts, st = _run(gt)
    assert int(res["fg_mask"].sum()) >= 3                   # stage 1 alone gives topk[0] = 3
    plain = TaskAlignedAssigner(stal_size=0)(torch.full((len(pts), 3), 0.1), torch.cat([pts - 4, pts + 4], 1),
                                             pts, gt, torch.zeros(1, dtype=torch.long),
                                             point_strides=st, reg_max=16)
    assert not plain["fg_mask"].any()                       # the inside-the-box rule gives none


def test_levels_follow_object_size():
    gt = torch.tensor([[10.0, 10.0, 16.0, 16.0],              # 6 px   -> P2
                       [40.0, 40.0, 140.0, 140.0]])           # 100 px -> a coarser level
    res, _, st = _run(gt)
    fg, g = res["fg_mask"], res["assigned_gt"]
    assert set(st[fg & (g == 0)].tolist()) == {4}
    assert st[fg & (g == 1)].min() >= 8


def test_each_point_has_one_gt_and_targets_are_gfl_style():
    gt = torch.tensor([[20.0, 20.0, 26.0, 26.0], [22.0, 20.0, 28.0, 26.0], [60.0, 60.0, 90.0, 80.0]])
    res, pts, _ = _run(gt, torch.tensor([0, 1, 2]))
    fg = res["fg_mask"]
    for g in range(3):                                        # every GT is represented
        assert (fg & (res["assigned_gt"] == g)).any()
    assert (res["assigned_labels"][fg] >= 0).all() and (res["assigned_labels"][~fg] == -1).all()
    assert torch.allclose(res["reg_weights"][fg], torch.full((int(fg.sum()),), 0.1))
    assert (res["assigned_ious"][~fg] == 0).all() and (res["assigned_ious"][fg] >= 0).all()


def test_no_assignment_beyond_the_dfl_range():
    """A 600-px box must never land on P2, whose DFL reaches only 16 x 4 px."""
    pts, st = _points(640)
    gt = torch.tensor([[10.0, 10.0, 610.0, 610.0]])
    n = pts.shape[0]
    res = RFLAAssigner()(torch.full((n, 1), 0.1), torch.cat([pts - 4, pts + 4], 1), pts, gt,
                         torch.zeros(1, dtype=torch.long), point_strides=st, reg_max=16)
    assert res["fg_mask"].any() and (st[res["fg_mask"]] >= 32).all()


def test_empty_gt():
    pts, st = _points()
    res = RFLAAssigner()(torch.rand(len(pts), 3), torch.cat([pts - 4, pts + 4], 1), pts,
                         torch.zeros(0, 4), torch.zeros(0, dtype=torch.long), point_strides=st, reg_max=16)
    assert not res["fg_mask"].any()


def test_configs_select_the_assigner():
    assert isinstance(build_assigner(load_config("configs/models/model_main.yaml")["assigner"]), RFLAAssigner)
    stal = build_assigner(load_config("configs/ablation/assigner_stal.yaml")["assigner"])
    assert isinstance(stal, TaskAlignedAssigner) and stal.stal_size == 8
    assert isinstance(build_assigner({"topk": 13}), TaskAlignedAssigner)      # pre-RFLA configs


def test_main_model_trains_with_rfla():
    cfg = load_config("configs/datasets/visdrone_rgb.yaml", "configs/models/model_main.yaml")
    cfg["model"]["num_classes"] = 10
    m = build_model(cfg).train()
    imgs = torch.rand(2, 3, 256, 256)
    tg = [{"boxes": torch.tensor([[10.0, 10.0, 14.0, 13.0], [100.0, 80.0, 180.0, 200.0]]),
           "labels": torch.tensor([0, 3])},
          {"boxes": torch.zeros(0, 4), "labels": torch.zeros(0, dtype=torch.long)}]
    losses = m.loss(imgs, tg)
    total = sum(losses.values())
    assert torch.isfinite(total)
    total.backward()
    assert losses["loss_giou"] > 0 and losses["loss_dfl"] > 0
