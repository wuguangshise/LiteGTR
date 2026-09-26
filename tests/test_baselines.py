"""The official-repository baselines (train_baselines.py + baselines/): data conversion,
generated configs and the dry run. Nothing here trains or needs the third-party repos."""
import json
from pathlib import Path

import pytest

yaml = pytest.importorskip("yaml")
PIL = pytest.importorskip("PIL.Image")

import train_baselines as TB  # noqa: E402
from baselines import configs  # noqa: E402
from baselines.prepare_visdrone import CLASSES, expected_paths, prepare  # noqa: E402

PATHS = {"train_images": "/d/train/images", "val_images": "/d/val/images",
         "train_coco": "/o/train.json", "val_coco": "/o/val.json"}


def _split(root: Path, name: str, lines: list[str], size=(100, 50)) -> None:
    (root / name / "images").mkdir(parents=True)
    (root / name / "annotations").mkdir(parents=True)
    PIL.new("RGB", size).save(root / name / "images" / "a.jpg")
    (root / name / "annotations" / "a.txt").write_text("\n".join(lines), encoding="utf-8")


def test_deim_schedule_reproduces_the_official_160_epochs():
    # configs/deim_dfine/deim_hgnetv2_n_coco.yml: policy [4, 78, 148], mixup [4, 78], stop 148
    assert configs.deim_schedule(160) == {"stop": 148, "policy": [4, 78, 148], "mixup": [4, 78]}


def test_deim_config_scales_with_batch_and_parses_as_numbers(tmp_path):
    cfg = configs.deim_config(tmp_path / "deim", tmp_path / "c.yml", tmp_path / "out", PATHS,
                              epochs=200, batch=8, workers=4)
    c = yaml.safe_load(cfg.read_text(encoding="utf-8"))
    k = 8 / 32
    assert c["optimizer"]["lr"] == pytest.approx(0.0008 * k)
    assert all(g.get("lr", 0) == pytest.approx(0.0004 * k)
               for g in c["optimizer"]["params"][:2])
    assert c["ema"] == {"decay": pytest.approx(1 - 0.0001 * k), "warmups": 4000}
    assert c["warmup_iter"] == 8000
    assert c["epoches"] == 200 and c["num_classes"] == 10
    assert c["remap_mscoco_category"] is False
    assert c["train_dataloader"]["total_batch_size"] == 8
    assert c["train_dataloader"]["collate_fn"]["stop_epoch"] == 188
    assert Path(c["__include__"][0]).is_absolute()


def test_remdet_config_overrides_data_schedule_and_batch(tmp_path):
    cfg = configs.remdet_config("rel/remdet", tmp_path / "r.py", tmp_path / "w", PATHS,
                                epochs=200, batch=8, workers=4, seed=0)
    text = cfg.read_text(encoding="utf-8")
    base = text.split("_base_ = '")[1].split("'")[0]
    assert Path(base).is_absolute() and base.endswith("remdet_tiny-300e_coco.py")
    compile(text, str(cfg), "exec")
    for s in ("max_epochs = 200", "train_batch_size_per_gpu = 8", "seed=0",
              "base_batch_size=256", "ann_file='/o/train.json'", "img='/d/val/images/'"):
        assert s in text


def test_prepare_visdrone_formats(tmp_path):
    ann = ["10,5,20,10,1,4,0,0",      # car -> class 3
           "90,40,20,20,1,1,0,0",     # pedestrian, runs off the image -> clipped in YOLO
           "0,0,30,30,0,0,0,0",       # ignored region: not an object
           "0,0,30,30,1,11,0,0",      # others: not an object
           "5,5,0,8,1,2,0,0"]         # degenerate
    _split(tmp_path, "train", ann)
    _split(tmp_path, "val", ann)
    p = prepare(tmp_path, "train", "val", tmp_path / "out")
    assert p == expected_paths(tmp_path, "train", "val", tmp_path / "out")

    rows = [list(map(float, r.split())) for r in
            (p["train_labels"] / "a.txt").read_text().splitlines()]
    assert rows[0] == pytest.approx([3, 0.2, 0.2, 0.2, 0.2])
    assert rows[1] == pytest.approx([0, 0.95, 0.9, 0.1, 0.2])
    assert len(rows) == 2

    coco = json.loads(p["val_coco"].read_text())
    assert [c["name"] for c in coco["categories"]] == CLASSES
    assert [a["category_id"] for a in coco["annotations"]] == [3, 0]
    # pycocotools treats a matched GT id of 0 as "unmatched"
    assert min(a["id"] for a in coco["annotations"]) >= 1
    assert min(i["id"] for i in coco["images"]) >= 1
    assert coco["annotations"][1]["bbox"] == [90, 40, 20, 20]    # COCO GT keeps the raw box

    y = yaml.safe_load(p["ultralytics_yaml"].read_text(encoding="utf-8"))
    assert y["names"] == dict(enumerate(CLASSES)) or y["names"] == CLASSES


def test_dry_run_commands_write_nothing(tmp_path, monkeypatch):
    monkeypatch.setattr(TB, "OUT", tmp_path / "runs")
    monkeypatch.setattr(TB, "THIRD_PARTY", tmp_path / "tp")
    rc = {"epochs": 200, "batch": 8, "imgsz": 640, "seed": 0, "workers": 4, "device": "0"}
    paths = expected_paths(tmp_path / "data", "train", "val", tmp_path / "runs" / "data")
    for name, fw, model in TB.BASELINES:
        cmd = TB.command(name, fw, model, rc, paths, write=False)
        assert cmd and all(isinstance(c, str) for c in cmd)
    assert not (tmp_path / "runs").exists() and not (tmp_path / "tp").exists()
    assert not (tmp_path / "data").exists()


def test_gpu_is_selected_by_visible_devices():
    assert TB.env_for("remdet", "1")["CUDA_VISIBLE_DEVICES"] == "1"
    assert TB.env_for("deim", "cpu")["CUDA_VISIBLE_DEVICES"] == ""


def test_environment_check_flags_versions_that_break_training():
    assert "torchvision" in TB.version_warning("deim", "torch 2.14.0 torchvision 0.29.0+cu130 cuda True")
    assert TB.version_warning("deim", "torch 2.5.1 torchvision 0.20.1+cu124 cuda True") == ""
    assert "numpy" in TB.version_warning("remdet", "mmcv 2.2.0 torch 2.2.0 numpy 2.4.6 cuda True")
    assert TB.version_warning("remdet", "mmcv 2.2.0 torch 2.2.0 numpy 1.26.4 cuda True") == ""


def test_remdet_runs_its_own_mmdet_from_the_repository():
    env = TB.env_for("remdet", "0")
    assert env["PYTHONPATH"].split(__import__("os").pathsep)[0] == str(TB.THIRD_PARTY / "remdet")
    assert "PYTHONPATH" not in TB.env_for("deim", "0") or \
        str(TB.THIRD_PARTY / "remdet") not in TB.env_for("deim", "0")["PYTHONPATH"]
