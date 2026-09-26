"""The evaluation protocol: original pixels, top 100 detections per image, AI-TOD buckets."""
import numpy as np
import pytest

from datasets.metrics import COCOMeanAP
from engine.evaluator import to_original


def _grid(n, size=10.0, step=20.0):
    xy = np.array([[(i % 40) * step, (i // 40) * step] for i in range(n)], dtype=np.float64)
    return np.hstack([xy, xy + size])


def test_only_the_top_100_detections_per_image_count():
    """As RemDet / LEAF-YOLO / D-FINE: 150 perfect boxes, only 100 scored -> recall 100/150."""
    gt = _grid(150)
    m = COCOMeanAP(["a"])
    m.add(0, 1000, 1000, "day", gt, np.zeros(150), gt, np.linspace(0.9, 0.5, 150), np.zeros(150))
    r = m.evaluate()
    for k in ("mAP50", "mAP50_95", "AP_small"):
        assert r[k] < 0.7, k                 # capped well below 1.0 by the 100-box limit


def test_size_buckets_and_aitod_ranges():
    gt = np.array([[0, 0, 5, 5], [100, 100, 112, 112], [200, 200, 250, 250], [400, 400, 600, 600]], float)
    m = COCOMeanAP(["a"])
    m.add(0, 1000, 1000, "day", gt, np.zeros(4), gt, np.array([0.9, 0.8, 0.7, 0.6]), np.zeros(4))
    r = m.evaluate()
    for k in ("AP_small", "AP_medium", "AP_large", "AP_vt", "AP_t"):
        assert r[k] == pytest.approx(1.0, abs=1e-3), k


def test_to_original_maps_predictions_and_uses_original_gt():
    from datasets.transforms import LetterBox

    ori_hw = (765, 1360)
    gt = np.array([[100.0, 200.0, 130.0, 260.0]], np.float32)
    _, gt_in = LetterBox(640)(np.zeros((*ori_hw, 3), np.uint8), gt.copy())
    meta = {"ori_shape": ori_hw, "ori_boxes": gt, "ori_labels": np.array([3])}
    h, w, g, l, d = to_original(meta, (640, 640), gt_in, np.array([3]), gt_in)
    assert (h, w) == ori_hw and l.tolist() == [3]
    assert np.allclose(g, gt) and np.allclose(d, gt, atol=1e-3)
    # no metadata -> input space, unchanged
    assert to_original({}, (640, 640), gt_in, np.array([3]), gt_in)[:2] == (640, 640)


def test_visdrone_eval_images_are_not_painted_by_default():
    from datasets import builder

    seen = {}

    class Fake:
        def __init__(self, **kw):
            seen[kw["train"]] = kw["ignore_mode"]

    import datasets.visdrone as vd
    orig = vd.VisDroneDataset
    vd.VisDroneDataset = Fake
    try:
        cfg = {"data": {"name": "visdrone", "root": ".", "train_split": "t", "val_split": "v",
                        "ignore_mode": "mask"}}
        builder.build_dataset(cfg, "train", train=True)
        builder.build_dataset(cfg, "val", train=False)
    finally:
        vd.VisDroneDataset = orig
    assert seen == {True: "mask", False: "drop"}
