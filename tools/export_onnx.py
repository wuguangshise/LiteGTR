"""Export to ONNX with STATIC shapes.

P1-8: the whole point of a fixed token budget is that ``top-k`` has a constant
``k`` and no dynamic control flow, so TensorRT does not fall back to a slower
path.  This script deliberately exports with ``dynamic_axes=None``; if that ever
fails, the token path has acquired data-dependent shapes and the deployment
claim in the paper is no longer true.

The EMA teacher is stripped before export -- it is training-only and must not
appear in deployment params/FLOPs.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from models.build import build_model, load_config  # noqa: E402
from utils.export import onnx_export  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", nargs="+", required=True)
    ap.add_argument("--weights", default=None)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--opset", type=int, default=13)
    ap.add_argument("--out", default="runs/export/litegtr.onnx")
    ap.add_argument("--simplify", action="store_true")
    a = ap.parse_args()

    cfg = load_config(*a.config)
    model = build_model(cfg).eval()
    if a.weights:
        ck = torch.load(a.weights, map_location="cpu", weights_only=False)
        model.load_state_dict(ck["model"], strict=False)
    model.ema_router = None          # training-only, never exported

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    x = torch.randn(a.batch, 3, a.imgsz, a.imgsz)

    class Wrapper(torch.nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m

        def forward(self, t):
            cls, reg, feats = self.m(t)
            c, r, boxes, _, _ = self.m.head.decode(cls, reg, feats)
            return c.sigmoid(), boxes

    onnx_export(
        Wrapper(model), x, str(out),
        input_names=["images"], output_names=["scores", "boxes"],
        opset_version=a.opset, do_constant_folding=True,
        dynamic_axes=None,           # STATIC on purpose -- see the docstring
    )
    print(f"exported {out} (static {a.batch}x3x{a.imgsz}x{a.imgsz})")

    try:
        import onnx
        m = onnx.load(str(out))
        onnx.checker.check_model(m)
        dyn = [d for vi in list(m.graph.input) + list(m.graph.output)
               for d in vi.type.tensor_type.shape.dim if d.dim_param]
        print(f"onnx check OK | dynamic dims in io: {len(dyn)} (expected 0)")
        if a.simplify:
            from onnxsim import simplify
            ms, ok = simplify(m)
            if ok:
                onnx.save(ms, str(out))
                print("simplified")
    except ImportError:
        print("[warn] onnx not installed -- skipped validation")


if __name__ == "__main__":
    main()
