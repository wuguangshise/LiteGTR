"""Native XML reading, border handling and the label cache.

Builds a tiny synthetic DroneVehicle tree so the round trip is tested without
the real 28k-image dataset: padded image in, correctly-shifted boxes out.
"""
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")
cv2 = pytest.importorskip("cv2")

from datasets.dronevehicle import DroneVehicleDataset, norm_class, parse_obj  # noqa: E402
from datasets.label_cache import poly_to_hbb  # noqa: E402

BORDER = 100
PADDED = (712, 840)      # h, w  -- content is 512x640


def _write_xml(path: Path, objects, w=840, h=712):
    root = ET.Element("annotation")
    size = ET.SubElement(root, "size")
    ET.SubElement(size, "width").text = str(w)
    ET.SubElement(size, "height").text = str(h)
    for name, (x1, y1, x2, y2) in objects:
        obj = ET.SubElement(root, "object")
        ET.SubElement(obj, "name").text = name
        poly = ET.SubElement(obj, "polygon")
        for i, (px, py) in enumerate([(x1, y1), (x2, y1), (x2, y2), (x1, y2)], start=1):
            ET.SubElement(poly, f"x{i}").text = str(px)
            ET.SubElement(poly, f"y{i}").text = str(py)
    ET.ElementTree(root).write(path)


@pytest.fixture
def tiny_root(tmp_path):
    base = tmp_path / "train"
    img_dir, ann_dir = base / "rgb", base / "rgb_xml"
    img_dir.mkdir(parents=True)
    ann_dir.mkdir(parents=True)
    img = np.full((PADDED[0], PADDED[1], 3), 255, dtype=np.uint8)
    img[BORDER:PADDED[0] - BORDER, BORDER:PADDED[1] - BORDER] = 120   # content region
    for i in range(3):
        cv2.imwrite(str(img_dir / f"{i:04d}.jpg"), img)
        # box at (150,150)-(200,200) in PADDED coords -> (50,50)-(100,100) cropped
        _write_xml(ann_dir / f"{i:04d}.xml", [("car", (150, 150, 200, 200)),
                                              ("feright_car", (300, 300, 340, 360))])
    return tmp_path


def test_class_aliases_are_normalised():
    assert norm_class("feright_car") == "freight_car"
    assert norm_class("Freight Car") == "freight_car"
    assert norm_class("CAR") == "car"
    assert norm_class("bicycle") is None


def test_border_is_removed_losslessly_and_boxes_shift_with_it(tiny_root):
    ds = DroneVehicleDataset(root=str(tiny_root), split="train", train=False, mosaic_prob=0.0)
    img, boxes, labels, meta = ds.load_raw(0)
    assert img.shape[:2] == (512, 640), "content size after border crop"
    assert (img == 255).sum() == 0, "white margin should be gone"
    assert len(boxes) == 2
    assert np.allclose(boxes[0], [50, 50, 100, 100], atol=1e-3)
    assert labels[0] == 0                       # car
    assert labels[1] == 4                       # freight_car via the alias


def test_cache_is_reused_on_second_construction(tiny_root):
    DroneVehicleDataset(root=str(tiny_root), split="train", train=False, mosaic_prob=0.0)
    cached = list((tiny_root / "train").glob(".litegtr_cache_*.npz"))
    assert len(cached) == 1
    ds2 = DroneVehicleDataset(root=str(tiny_root), split="train", train=False, mosaic_prob=0.0)
    assert len(ds2) == 3


def test_oriented_boxes_survive_in_the_cache(tiny_root):
    ds = DroneVehicleDataset(root=str(tiny_root), split="train", train=False, mosaic_prob=0.0)
    obb = ds.obb(0)
    assert obb.shape == (2, 4, 2), "four corners kept, not reduced to xyxy"


def test_rotated_box_hbb_is_the_circumscribed_box():
    obj = ET.fromstring(
        '<object><name>car</name><robndbox>'
        '<cx>100</cx><cy>100</cy><w>40</w><h>20</h><angle>0.0</angle>'
        '</robndbox></object>')
    pts = parse_obj(obj)
    hbb = poly_to_hbb(pts.reshape(1, 8))[0]
    assert np.allclose(hbb, [80, 90, 120, 110], atol=1e-3)


def test_no_images_are_rewritten(tiny_root):
    before = {p: p.stat().st_mtime_ns for p in (tiny_root / "train" / "rgb").iterdir()}
    ds = DroneVehicleDataset(root=str(tiny_root), split="train", train=False, mosaic_prob=0.0)
    ds.load_raw(0)
    after = {p: p.stat().st_mtime_ns for p in (tiny_root / "train" / "rgb").iterdir()}
    assert before == after, "source images must never be modified"
