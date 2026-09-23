"""tools/visualize_tokens.py draws what routing actually saw -- smoke test on random weights."""
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("cv2")

from models.build import build_model  # noqa: E402
from tests._variants import variant_cfg  # noqa: E402
from tools.visualize_tokens import compose, render_row, token_maps  # noqa: E402


def test_maps_panels_and_figure():
    model = build_model(variant_cfg("main")).eval()
    image = torch.rand(3, 256, 256)
    maps = token_maps(model, image)
    assert set(maps["score"]) == set(model.token_levels)
    for lv in model.token_levels:
        s = maps["score"][lv]
        assert s.min() >= 0 and s.max() <= 1                          # sigmoid probabilities
        assert maps["tokens"][lv].shape == (model.selector.selectors[lv].num_tokens, 2)
    assert set(maps["delta"]) == set(model.writeback.blocks)
    rows = [render_row(image, [[10, 10, 40, 40]], maps, "a", delta=True),
            render_row(image, [], maps, "b", norm="minmax")]
    assert {"input", "score_P3", "delta_P3"} <= set(rows[0])
    fig = compose(rows, "abs")
    assert fig.ndim == 3 and fig.shape[0] == 2 * rows[0]["input"].shape[0]


def _fake_visdrone(tmp_path):
    cv2 = pytest.importorskip("cv2")
    import numpy as np

    split = tmp_path / "VisDrone2019-DET-val"
    (split / "images").mkdir(parents=True)
    (split / "annotations").mkdir()
    cv2.imwrite(str(split / "images" / "0000001_a.jpg"), np.full((300, 400, 3), 200, np.uint8))
    cv2.imwrite(str(split / "images" / "0000002_b.jpg"), np.full((300, 400, 3), 50, np.uint8))
    # two objects, one ignored region (class 0), one "others" (class 11)
    (split / "annotations" / "0000002_b.txt").write_text(
        "10,20,30,40,1,4,0,0\n100,100,50,20,1,1,0,0\n200,200,40,40,0,0,0,0\n300,10,20,20,1,11,0,0\n")
    return split / "images" / "0000002_b.jpg"


def test_load_image_uses_the_visdrone_annotations(tmp_path):
    from tools.visualize_tokens import load_image

    image, gt = load_image(_fake_visdrone(tmp_path), img_size=640)
    assert image.shape == (3, 640, 640)
    assert gt.shape == (2, 4)                     # ignored region and "others" dropped
    # 400x300 letterboxed to 640: scale 1.6, vertical pad 80 -> first box (10,20,40,60)
    assert gt[0].tolist() == pytest.approx([16.0, 112.0, 64.0, 176.0], abs=1.0)


def test_cli_writes_figure_and_raw_maps(tmp_path):
    import subprocess
    import sys

    img = _fake_visdrone(tmp_path)
    model = build_model(variant_cfg("main"))
    ck = tmp_path / "best.pt"
    torch.save({"model": model.state_dict(), "model_ema": model.state_dict()}, ck)
    out = tmp_path / "vis"
    r = subprocess.run([sys.executable, "tools/visualize_tokens.py", "--image", str(img),
                        "--weights", str(ck), str(ck), "--labels", "a", "b", "--out", str(out),
                        "--device", "cpu"], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert (out / "0000002_b.png").exists()
    import numpy as np

    z = np.load(out / "0000002_b.npz")
    assert z["gt_boxes"].shape == (2, 4) and "a/score_P3" in z and "b/delta_P5" in z
