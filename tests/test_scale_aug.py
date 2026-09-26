"""Scale jitter (datasets/transforms.py random_scale) and the CIoU 2.0 box loss --
both part of the default recipe -- with their off-switch ablations."""
import random

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from datasets.base import DetectionDataset  # noqa: E402
from datasets.builder import build_dataset  # noqa: E402
from datasets.transforms import random_scale  # noqa: E402
from models.build import build_model, load_config  # noqa: E402


def _fix(monkeypatch, s, off=0):
    monkeypatch.setattr(random, "uniform", lambda a, b: s)
    monkeypatch.setattr(random, "randint", lambda a, b: min(off, b))


def test_zoom_in_maps_boxes_and_keeps_the_canvas_size(monkeypatch):
    _fix(monkeypatch, 1.5, off=0)
    img = np.zeros((100, 200, 3), np.uint8)
    out, b, keep = random_scale(img, np.array([[10, 20, 30, 40]], np.float32), 0.5)
    assert out.shape == img.shape
    np.testing.assert_allclose(b, [[15, 30, 45, 60]])
    assert keep.tolist() == [0]


def test_zoom_out_pads_and_shifts(monkeypatch):
    _fix(monkeypatch, 0.5, off=10)
    img = np.full((100, 200, 3), 7, np.uint8)
    out, b, keep = random_scale(img, np.array([[20, 20, 60, 40]], np.float32), 0.5)
    assert out.shape == img.shape
    assert (out[0, 0] == 114).all() and (out[15, 15] == 7).all()
    np.testing.assert_allclose(b, [[20, 20, 40, 30]])       # *0.5, then +10 px


def test_boxes_cropped_away_or_too_small_are_dropped(monkeypatch):
    _fix(monkeypatch, 1.5, off=0)
    img = np.zeros((100, 100, 3), np.uint8)
    boxes = np.array([[10, 10, 30, 30],       # kept
                      [95, 95, 99, 99],       # pushed outside the canvas
                      [10, 50, 11, 51]],      # 1.5 px after zoom: too small
                     np.float32)
    _, b, keep = random_scale(img, boxes, 0.5)
    assert keep.tolist() == [0] and len(b) == 1


def test_no_boxes(monkeypatch):
    _fix(monkeypatch, 0.8)
    out, b, keep = random_scale(np.zeros((64, 64, 3), np.uint8), np.zeros((0, 4), np.float32), 0.5)
    assert out.shape == (64, 64, 3) and b.shape == (0, 4) and len(keep) == 0


class _Fake(DetectionDataset):
    classes = ["a", "b"]

    def __len__(self):
        return 4

    def load_raw(self, i):
        rng = np.random.default_rng(i)
        img = rng.integers(0, 255, (300, 400, 3), dtype=np.uint8)
        boxes = np.array([[50, 50, 90, 80], [200, 100, 260, 190], [300, 20, 330, 60]], np.float32)
        return img, boxes, np.array([0, 1, 0]), {"image_id": i}


@pytest.mark.parametrize("mosaic", [0.0, 1.0])
def test_dataset_applies_it_and_keeps_labels_aligned(mosaic):
    random.seed(0)
    np.random.seed(0)
    ds = _Fake(img_size=128, train=True, mosaic_prob=mosaic, scale_aug=0.5)
    for i in range(len(ds)):
        img, t = ds[i]
        assert img.shape == (3, 128, 128)
        assert len(t["boxes"]) == len(t["labels"])
        if len(t["boxes"]):
            assert (t["boxes"] >= 0).all() and (t["boxes"] <= 128).all()


def test_off_by_default_and_never_at_evaluation():
    assert _Fake(img_size=128, train=True).scale_aug == 0.0
    assert _Fake(img_size=128, train=False, scale_aug=0.5).scale_aug == 0.0


def test_default_recipe_uses_both():
    import train_litegtr as T

    assert T.SCALE_AUG == 0.5
    main = load_config("configs/models/model_main.yaml")
    assert (main["loss"]["iou_type"], main["loss"]["iou_weight"], main["loss"]["nwd_weight"]) == \
        ("ciou", 2.0, 1.0)
    for ds in ("configs/datasets/visdrone_rgb.yaml", "configs/datasets/dronevehicle_rgb.yaml"):
        assert load_config(ds)["data"]["scale_aug"] == 0.5        # tools/train.py path agrees


def test_off_switch_configs_change_one_thing_each():
    main = load_config("configs/models/model_main.yaml")
    no_sa = load_config("configs/ablation/no_scale_aug.yaml")
    c1 = load_config("configs/ablation/ciou1.yaml")
    assert no_sa["data"]["scale_aug"] == 0.0
    assert no_sa["model"] == main["model"] and no_sa["loss"] == main["loss"]
    assert c1["model"] == main["model"] and "data" not in c1
    assert {k: v for k, v in c1["loss"].items() if k != "iou_weight"} == \
        {k: v for k, v in main["loss"].items() if k != "iou_weight"}
    c1["model"]["num_classes"] = 10
    m = build_model(c1)
    assert m.box_iou.loss_weight == 1.0 and m.nwd.loss_weight == 1.0


def test_yaml_path_passes_it_to_the_dataset(tmp_path):
    root = tmp_path / "VisDrone2019-DET-train"
    (root / "images").mkdir(parents=True)
    (root / "annotations").mkdir()
    import cv2
    cv2.imwrite(str(root / "images" / "a.jpg"), np.zeros((64, 64, 3), np.uint8))
    cfg = load_config("configs/datasets/visdrone_rgb.yaml", "configs/ablation/no_scale_aug.yaml")
    cfg["data"]["root"] = str(tmp_path)
    assert build_dataset(cfg, "train", True).scale_aug == 0.0          # the ablation wins
    cfg["data"]["scale_aug"] = 0.5
    assert build_dataset(cfg, "train", True).scale_aug == 0.5
