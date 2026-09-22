"""Measure real latency / FPS / peak memory on the target device.

The paper claims "edge deployment"; theoretical MACs are not evidence.  Sparse
token ops in particular can be slower in practice than their FLOP count implies,
which is exactly why the budget is fixed and the graph static (P1-8).
"""
from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from models.build import build_model, load_config  # noqa: E402


def bench_torch(model, x, warmup: int, iters: int, device) -> dict:
    model.eval()
    with torch.no_grad():
        for _ in range(warmup):
            model(x)
        if device.type == "cuda":
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        ts = []
        for _ in range(iters):
            t0 = time.perf_counter()
            model(x)
            if device.type == "cuda":
                torch.cuda.synchronize()
            ts.append((time.perf_counter() - t0) * 1000)
    out = {"mean_ms": statistics.mean(ts), "median_ms": statistics.median(ts),
           "p90_ms": sorted(ts)[int(0.9 * len(ts)) - 1], "fps": 1000.0 / statistics.mean(ts)}
    if device.type == "cuda":
        out["peak_mem_MB"] = torch.cuda.max_memory_allocated() / 1e6
    return out


def bench_onnx(path: str, x, warmup: int, iters: int) -> dict:
    import numpy as np
    import onnxruntime as ort
    sess = ort.InferenceSession(path, providers=ort.get_available_providers())
    name = sess.get_inputs()[0].name
    arr = x.cpu().numpy().astype(np.float32)
    for _ in range(warmup):
        sess.run(None, {name: arr})
    ts = []
    for _ in range(iters):
        t0 = time.perf_counter()
        sess.run(None, {name: arr})
        ts.append((time.perf_counter() - t0) * 1000)
    return {"mean_ms": statistics.mean(ts), "median_ms": statistics.median(ts),
            "fps": 1000.0 / statistics.mean(ts), "providers": sess.get_providers()}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", nargs="*", default=[])
    ap.add_argument("--onnx", default=None)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--iters", type=int, default=100)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--half", action="store_true")
    a = ap.parse_args()

    device = torch.device(a.device)
    x = torch.randn(a.batch, 3, a.imgsz, a.imgsz, device=device)
    print(f"device={device} torch={torch.__version__} "
          f"gpu={torch.cuda.get_device_name(0) if device.type == 'cuda' else 'n/a'}")

    if a.onnx:
        print("onnxruntime:", bench_onnx(a.onnx, x, a.warmup, a.iters))
    if a.config:
        cfg = load_config(*a.config)
        model = build_model(cfg).to(device)
        model.ema_router = None
        if a.half and device.type == "cuda":
            model = model.half()
            x = x.half()
        print(f"torch ({'fp16' if a.half else 'fp32'}):", bench_torch(model, x, a.warmup, a.iters, device))


if __name__ == "__main__":
    main()
