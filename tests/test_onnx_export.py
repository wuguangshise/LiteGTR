"""P1-8: the deployment claim requires a STATIC graph.

If the token path ever acquires data-dependent shapes, this test fails -- which
is the point. A fixed budget that exports dynamically buys nothing on TensorRT.
"""
import pytest

torch = pytest.importorskip("torch")

from models.build import build_model, load_config  # noqa: E402


class _Wrapper(torch.nn.Module):
    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, t):
        cls, reg, feats = self.m(t)
        c, _, boxes, _, _ = self.m.head.decode(cls, reg, feats)
        return c.sigmoid(), boxes


def test_export_is_static(tmp_path):
    onnx = pytest.importorskip("onnx")
    cfg = load_config("configs/models/model_main.yaml")
    cfg["model"]["num_classes"] = 10
    model = build_model(cfg).eval()
    model.ema_router = None
    path = tmp_path / "m.onnx"
    torch.onnx.export(_Wrapper(model), torch.randn(1, 3, 256, 256), str(path),
                      input_names=["images"], output_names=["scores", "boxes"],
                      opset_version=13, do_constant_folding=True, dynamic_axes=None)
    m = onnx.load(str(path))
    onnx.checker.check_model(m)
    dyn = [d.dim_param for vi in list(m.graph.input) + list(m.graph.output)
           for d in vi.type.tensor_type.shape.dim if d.dim_param]
    assert not dyn, f"dynamic dimensions leaked into the graph: {dyn}"
