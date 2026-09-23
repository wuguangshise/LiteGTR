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
