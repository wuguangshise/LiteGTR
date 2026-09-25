"""Main vs. writeback-to-P2 (configs/ablation/writeback_p2.yaml), side by side.

    python tools/check_writeback_p2.py                 # 640 input, CUDA if available
    python tools/check_writeback_p2.py --device cpu --iters 20

For each model: forward / backward on a dummy batch (does it train at all),
deployment params, MACs including the writeback attention that thop skips, and
latency (batch 1, fp32, forward only). No weights, no dataset needed.
"""
from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from models.build import build_model, count_deploy_params, load_config  # noqa: E402

CONFIGS = {
    "main": "configs/models/model_main.yaml",
    "writeback_p2": "configs/ablation/writeback_p2.yaml",
}


def build(cfg_path: str):
    cfg = load_config(cfg_path)
    cfg["model"]["num_classes"] = 10
    return build_model(cfg)


def train_step_ok(model) -> bool:
    """One loss + backward on a dummy batch; also checks P2's writeback gets a gradient."""
    model.train()
    x = torch.randn(2, 3, 256, 256)
    targets = [{"boxes": torch.tensor([[20.0, 20.0, 26.0, 26.0], [100.0, 90.0, 130.0, 140.0]]),
                "labels": torch.tensor([0, 3])},
               {"boxes": torch.zeros(0, 4), "labels": torch.zeros(0, dtype=torch.long)}]
    total = sum(model.loss(x, targets).values())
    if not torch.isfinite(total):
        return False
    total.backward()
    for lv, blk in model.writeback.blocks.items():
        # gamma starts at 0, so it is the one writeback weight that always gets a gradient
        if blk.gamma.grad is None or not torch.isfinite(blk.gamma.grad).all():
            print(f"  [fail] no gradient into the {lv} writeback")
            return False
    return True


def macs(model, imgsz: int) -> tuple[float, float]:
    """(thop MACs, writeback attention MACs thop does not count), per image."""
    x = torch.randn(1, 3, imgsz, imgsz)
    try:
        from thop import profile
        conv = profile(model.eval(), inputs=(x,), verbose=False)[0]
    except ImportError:
        conv = float("nan")
        print("  [warn] thop not installed -- conv MACs not measured (pip install thop)")
    n = model.selector.num_tokens
    stride = dict(zip(model.levels, model.strides))
    attn = sum(2 * (imgsz // stride[lv]) ** 2 * n * model.writeback.blocks[lv].dh * model.writeback.blocks[lv].h
               for lv in model.writeback.levels)             # q@k^T and attn@v per level
    return conv, attn


@torch.no_grad()
def latency_ms(model, imgsz: int, device, warmup: int, iters: int) -> float:
    model.eval().to(device)
    x = torch.randn(1, 3, imgsz, imgsz, device=device)
    for _ in range(warmup):
        model(x)
    ts = []
    for _ in range(iters):
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        model(x)
        if device.type == "cuda":
            torch.cuda.synchronize()
        ts.append((time.perf_counter() - t0) * 1000)
    return statistics.mean(ts)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--warmup", type=int, default=30)
    ap.add_argument("--iters", type=int, default=200)
    a = ap.parse_args()
    device = torch.device(a.device)

    rows = {}
    for name, path in CONFIGS.items():
        print(f"{name}: {path}")
        ok = train_step_ok(build(path))
        model = build(path)
        conv, attn = macs(model, a.imgsz)
        rows[name] = {"ok": ok, "writeback": model.writeback.levels,
                      "params": count_deploy_params(model), "macs": conv + attn,
                      "ms": latency_ms(model, a.imgsz, device, a.warmup, a.iters)}
        print(f"  train step {'OK' if ok else 'FAILED'}, writeback levels {model.writeback.levels}")

    print(f"\ninput 1x3x{a.imgsz}x{a.imgsz}, {device.type}, fp32, forward only")
    print(f"{'model':14s}{'params':>10s}{'MACs':>9s}{'latency':>11s}")
    for name, r in rows.items():
        print(f"{name:14s}{r['params'] / 1e6:9.3f}M{r['macs'] / 1e9:8.2f}G{r['ms']:9.2f}ms")
    m, p = rows["main"], rows["writeback_p2"]
    print(f"{'delta':14s}{(p['params'] - m['params']) / 1e3:+9.1f}K{(p['macs'] - m['macs']) / 1e9:+8.2f}G"
          f"{p['ms'] - m['ms']:+9.2f}ms")
    if not all(r["ok"] for r in rows.values()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
