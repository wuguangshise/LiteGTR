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


# ------------------------------------------------------------ STAL
def test_tiny_gt_between_grid_centres_gets_candidates_only_with_stal():
    pts, st = _points()
    gt = torch.tensor([[2.2, 2.2, 3.8, 3.8], [40.0, 40.0, 60.0, 80.0]])   # 1.6 px box: no centre inside
    lab = torch.tensor([0, 3])
    pred = torch.cat([pts - 32, pts + 32], 1)
    sc = torch.full((len(pts), 10), 0.01)
    assert not bool(coverage(gt, pts, st, 16)[0])
    assert bool(coverage(gt, pts, st, 16, stal_size=8)[0])
    old = TaskAlignedAssigner(stal_size=0)(sc, pred, pts, gt, lab, point_strides=st, reg_max=16)
    new = TaskAlignedAssigner(stal_size=8)(sc, pred, pts, gt, lab, point_strides=st, reg_max=16)
    assert int((old["assigned_gt"][old["fg_mask"]] == 0).sum()) == 0
    pos = new["fg_mask"] & (new["assigned_gt"] == 0)
    # the 8x8 selection window around (3, 3) holds P2 centres (2|6, 2|6) and the P3 centre (4, 4)
    assert int(pos.sum()) == 5
    assert set(map(tuple, pts[pos].tolist())) == {(2.0, 2.0), (2.0, 6.0), (6.0, 2.0), (6.0, 6.0), (4.0, 4.0)}


def test_stal_leaves_gts_larger_than_its_size_untouched():
    pts, st = _points()
    gt = torch.tensor([[40.0, 40.0, 60.0, 80.0], [100.0, 20.0, 130.0, 70.0]])
    lab = torch.tensor([3, 0])
    torch.manual_seed(0)
    pred = torch.cat([pts - 10, pts + 10], 1) + torch.rand(len(pts), 4)
    sc = torch.rand(len(pts), 10)
    a = TaskAlignedAssigner(stal_size=0)(sc, pred, pts, gt, lab, point_strides=st, reg_max=16)
    b = TaskAlignedAssigner(stal_size=8)(sc, pred, pts, gt, lab, point_strides=st, reg_max=16)
    for k in a:
        assert torch.equal(a[k], b[k]), k


def test_main_config_enables_stal():
    from models.build import build_model
    from tests._variants import variant_cfg
    assert build_model(variant_cfg("main")).assigner.stal_size == 8


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


def test_multi_label_emits_every_class_above_threshold():
    boxes = torch.tensor([[10.0, 10, 30, 60]])
    s = torch.zeros(1, 10)
    s[0, 0], s[0, 1] = 0.5, 0.3                               # pedestrian 0.5, people 0.3
    assert postprocess(s, boxes, (640, 640), score_thr=0.001)["labels"].tolist() == [0]
    multi = postprocess(s, boxes, (640, 640), score_thr=0.001, multi_label=True)
    assert sorted(multi["labels"].tolist()) == [0, 1]


def test_containment_keeps_two_separate_objects():
    boxes = torch.tensor([[10.0, 10, 30, 60], [34, 10, 54, 60]])
    s = _scores([(0, 0.8), (0, 0.7)])
    assert len(postprocess(s, boxes, (640, 640), containment=0.8, agnostic=True)["boxes"]) == 2


@pytest.mark.parametrize("agnostic,multi,max_det", [(False, True, 300), (True, False, 300), (False, True, 20)])
def test_containment_on_a_prefix_matches_the_full_matrix(agnostic, multi, max_det):
    """postprocess only builds the containment matrix over a score-sorted prefix; the
    result must equal filtering every NMS survivor and truncating afterwards."""
    from torchvision.ops import batched_nms

    from models.detector import _not_contained

    g = torch.Generator().manual_seed(0)
    s = torch.rand(800, 10, generator=g) ** 6
    xy = torch.rand(800, 2, generator=g) * 600
    boxes = torch.cat([xy, xy + torch.rand(800, 2, generator=g) * 40 + 2], 1)
    got = postprocess(s, boxes, (640, 640), 0.001, 0.7, max_det, 30000, agnostic, 0.8, multi)
    ref = postprocess(s, boxes, (640, 640), 0.001, 0.7, 10 ** 9, 30000, agnostic, None, multi)
    groups = torch.zeros_like(ref["labels"]) if agnostic else ref["labels"]
    k = batched_nms(ref["boxes"], ref["scores"], groups, 0.7)   # already NMS'd: identity
    k = k[_not_contained(ref["boxes"][k], groups[k], 0.8)][:max_det]
    assert len(got["scores"]) == max_det
    for key in ("boxes", "scores", "labels"):
        assert torch.equal(got[key], ref[key][k])


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
    assert "uncovered" in text and "STAL 8px" in text and "RemDet/Ultralytics" in text
    assert (tmp_path / "d" / "postprocess_sweep.csv").exists()


# ------------------------------------------------------------ drawing
def test_display_filter_draws_one_box_per_object():
    from engine.evaluator import display_filter, vis_cfg
    from models.build import load_config

    vis = vis_cfg(load_config("configs/models/model_main.yaml"))
    boxes = np.array([[10.0, 10, 30, 60],    # pedestrian on one person
                      [11, 10, 31, 61],      # people on the same person (multi-label)
                      [14, 12, 28, 40],      # nested, same person, lower score
                      [100, 100, 120, 150],  # a second person
                      [200, 200, 220, 250]])  # below the drawing threshold
    scores = np.array([0.6, 0.4, 0.35, 0.5, 0.1])
    labels = np.array([0, 1, 0, 0, 0])
    b, s, l = display_filter(boxes, scores, labels, **vis)
    assert s.tolist() == [0.6, 0.5] and l.tolist() == [0, 0]
    assert len(display_filter(boxes[:0], scores[:0], labels[:0], **vis)[0]) == 0


def test_vis_defaults_apply_to_configs_without_a_vis_block():
    from engine.evaluator import vis_cfg

    assert vis_cfg({}) == vis_cfg({"vis": {"score_thr": 0.3, "nms_iou": 0.6, "agnostic": True,
                                           "containment": 0.8, "label": "class"}})


def test_drawings_omit_ground_truth_by_default():
    """Grey GT boxes under the predictions made small-object figures unreadable."""
    import numpy as np

    from engine.evaluator import vis_cfg
    from models.build import load_config
    from utils.plots import draw_predictions

    assert vis_cfg(load_config("configs/models/model_main.yaml"))["show_gt"] is False
    assert vis_cfg({})["show_gt"] is False
    img = np.zeros((64, 64, 3), np.uint8)
    gt = np.array([[10, 10, 40, 40]], np.float32)
    none = np.zeros((0, 4)), np.zeros(0), np.zeros(0)
    assert not draw_predictions(img, *none, gt, ["a"]).any()
    assert draw_predictions(img, *none, gt, ["a"], show_gt=True).any()
