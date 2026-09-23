"""Tiny GTs must be assignable; duplicate suppression options must do exactly what they say."""
import numpy as np
import pytest

torch = pytest.importorskip("torch")

from assigners.task_aligned_assigner import TaskAlignedAssigner  # noqa: E402
from models.detector import postprocess  # noqa: E402
from models.head.gfl_head import GFLHead  # noqa: E402
from tools.diagnose_predictions import box_types, coverage  # noqa: E402


def _points(size=160):
    feats = [torch.zeros(1, 1, size // s, size // s) for s in (4, 8, 16, 32)]
    return GFLHead.make_points(feats, (4, 8, 16, 32), "cpu", torch.float32)


# ------------------------------------------------------------ tiny fallback
def test_tiny_gt_between_grid_centres_gets_a_positive_only_with_fallback():
    pts, st = _points()
    gt = torch.tensor([[2.2, 2.2, 3.8, 3.8], [40.0, 40.0, 60.0, 80.0]])   # 1.6 px box: no centre inside
    lab = torch.tensor([0, 3])
    pred = torch.cat([pts - 32, pts + 32], 1)
    sc = torch.full((len(pts), 10), 0.01)
    assert not bool(coverage(gt, pts, st, 16)[0])
    old = TaskAlignedAssigner(tiny_fallback=False)(sc, pred, pts, gt, lab, point_strides=st, reg_max=16)
    new = TaskAlignedAssigner(tiny_fallback=True)(sc, pred, pts, gt, lab, point_strides=st, reg_max=16)
    assert int((old["assigned_gt"][old["fg_mask"]] == 0).sum()) == 0
    pos = new["fg_mask"] & (new["assigned_gt"] == 0)
    assert int(pos.sum()) == 1
    assert pts[pos][0].tolist() == [2.0, 2.0]             # nearest P2 centre
    assert st[pos][0] == 4
    assert float(new["assigned_ious"][pos][0]) >= 0.1     # trainable target / regression weight


def test_fallback_leaves_normal_gts_untouched():
    pts, st = _points()
    gt = torch.tensor([[40.0, 40.0, 60.0, 80.0], [100.0, 20.0, 130.0, 70.0]])
    lab = torch.tensor([3, 0])
    torch.manual_seed(0)
    pred = torch.cat([pts - 10, pts + 10], 1) + torch.rand(len(pts), 4)
    sc = torch.rand(len(pts), 10)
    a = TaskAlignedAssigner(tiny_fallback=False)(sc, pred, pts, gt, lab, point_strides=st, reg_max=16)
    b = TaskAlignedAssigner(tiny_fallback=True)(sc, pred, pts, gt, lab, point_strides=st, reg_max=16)
    for k in a:
        assert torch.equal(a[k], b[k]), k


def test_main_config_enables_the_fallback():
    from models.build import build_model
    from tests._variants import variant_cfg
    assert build_model(variant_cfg("main")).assigner.tiny_fallback


# ------------------------------------------------------------ post-processing
def _scores(rows):
    s = torch.zeros(len(rows), 10)
    for i, (c, v) in enumerate(rows):
        s[i, c] = v
    return s


def test_classwise_nms_keeps_a_cross_class_duplicate_agnostic_does_not():
    boxes = torch.tensor([[10.0, 10, 30, 60], [11, 10, 31, 61]])
    s = _scores([(0, 0.8), (1, 0.6)])                       # pedestrian and people, same person
    assert len(postprocess(s, boxes, (640, 640))["boxes"]) == 2
    assert len(postprocess(s, boxes, (640, 640), agnostic=True)["boxes"]) == 1


def test_containment_removes_a_nested_box_that_plain_nms_keeps():
    outer = [10.0, 10, 40, 70]                               # 30 x 60
    inner = [14.0, 12, 36, 55]                               # inside, IoU ~0.53 < 0.6
    boxes = torch.tensor([outer, inner])
    s = _scores([(0, 0.8), (0, 0.5)])
    assert len(postprocess(s, boxes, (640, 640))["boxes"]) == 2
    kept = postprocess(s, boxes, (640, 640), containment=0.8)
    assert len(kept["boxes"]) == 1 and float(kept["scores"][0]) == pytest.approx(0.8)


def test_containment_keeps_two_separate_objects():
    boxes = torch.tensor([[10.0, 10, 30, 60], [34, 10, 54, 60]])
    s = _scores([(0, 0.8), (0, 0.7)])
    assert len(postprocess(s, boxes, (640, 640), containment=0.8, agnostic=True)["boxes"]) == 2


# ------------------------------------------------------------ diagnosis
def test_box_types():
    gt_b = np.array([[10, 10, 30, 60], [100, 100, 120, 150]], np.float32)
    gt_l = np.array([0, 3])
    dt_b = np.array([[10, 10, 30, 60],      # correct
                     [11, 10, 31, 61],      # duplicate (same class, already found)
                     [100, 100, 120, 150],  # wrong class on the car
                     [10, 10, 60, 120],     # poorly localised
                     [400, 400, 420, 420]], np.float32)
    dt_s = np.array([0.9, 0.8, 0.7, 0.6, 0.5])
    dt_l = np.array([0, 0, 1, 0, 0])
    t = box_types(dt_b, dt_s, dt_l, gt_b, gt_l, conf=0.25)
    assert t == {"correct": 1, "duplicate": 1, "wrong_class": 1, "localisation": 1, "background": 1}


def test_cli_runs_end_to_end(tmp_path):
    import subprocess
    import sys

    import cv2
    split = tmp_path / "VisDrone2019-DET-val"
    (split / "images").mkdir(parents=True)
    (split / "annotations").mkdir()
    for i in range(2):
        cv2.imwrite(str(split / "images" / f"{i:07d}.jpg"), np.full((300, 400, 3), 120, np.uint8))
        (split / "annotations" / f"{i:07d}.txt").write_text("10,20,30,40,1,4,0,0\n100,100,4,4,1,1,0,0\n")
    ds_cfg = tmp_path / "ds.yaml"
    ds_cfg.write_text(f"data: {{name: visdrone, root: '{tmp_path.as_posix()}', train_split: VisDrone2019-DET-val, "
                      "val_split: VisDrone2019-DET-val, img_size: 640}\n")
    from models.build import build_model
    from tests._variants import variant_cfg
    ck = tmp_path / "best.pt"
    sd = build_model(variant_cfg("main")).state_dict()
    torch.save({"model": sd, "model_ema": sd}, ck)
    r = subprocess.run([sys.executable, "tools/diagnose_predictions.py", "--config", str(ds_cfg),
                        "configs/models/model_main.yaml", "--weights", str(ck), "--out", str(tmp_path / "d"),
                        "--device", "cpu"], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    text = (tmp_path / "d" / "diagnosis.txt").read_text(encoding="utf-8")
    assert "uncovered" in text and "class-agnostic NMS 0.6" in text
    assert (tmp_path / "d" / "postprocess_sweep.csv").exists()
