"""STEP 1 of the build order: make the budget real before writing experiments.

Named profile_model.py, not profile.py: a script called profile.py shadows the
standard-library ``profile`` module. Running it puts tools/ on sys.path, and
torchvision -> torch._dynamo -> cProfile then imports THIS file instead, failing
with "module 'profile' has no attribute 'run'".

The original plan targeted 3-5M params and ~5G MACs @640.  The analytic sweep
(``--search``, no torch needed) showed the proposed backbone alone was 3.76M /
4.95G -- i.e. the entire MAC budget was consumed before the neck, tokens and
head existed.  Run this whenever a structural knob changes.

Usage
-----
    python tools/profile_model.py --search                       # analytic backbone sweep
    python tools/profile_model.py --config configs/models/model_main.yaml
    python tools/profile_model.py --config ... --imgsz 640 --out runs/profile
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils.budget import PRESETS, backbone_macs, backbone_params  # noqa: E402

TARGET_PARAMS = (3.0e6, 5.0e6)


def human(n: float) -> str:
    return f"{n / 1e6:.2f}M" if n < 1e9 else f"{n / 1e9:.2f}G"


def run_search() -> None:
    print(f"{'preset':20s}{'channels':22s}{'depths':16s}{'params':>9s}{'MACs@640':>11s}")
    print("-" * 78)
    for name, (ch, dp) in PRESETS.items():
        p = backbone_params(ch, dp)
        m = backbone_macs(ch, dp)
        stages = [round(x / 1e6, 2) for x in p["per_stage"]]
        print(f"{name:20s}{str(ch):22s}{str(dp):16s}{human(p['total']):>9s}{human(m):>11s}   stages={stages}")
    print("\nBackbone only -- neck, token path and head are on top of these numbers.")


def group_params(model) -> dict[str, int]:
    groups: dict[str, int] = {}
    for name, p in model.named_parameters():
        if not p.requires_grad and name.startswith("ema_router"):
            key = "ema_router (train-only)"
        else:
            key = name.split(".")[0]
        groups[key] = groups.get(key, 0) + p.numel()
    return groups


def run_profile(cfg_paths: list[str], imgsz: int, out: str | None, batch: int) -> None:
    import torch
    from models.build import build_model, load_config

    cfg = load_config(*cfg_paths)
    model = build_model(cfg).eval()

    groups = group_params(model)
    deploy = sum(v.numel() for k, v in model.state_dict().items() if not k.startswith("ema_router."))
    total = sum(p.numel() for p in model.parameters())

    x = torch.randn(batch, 3, imgsz, imgsz)
    with torch.no_grad():
        cls, reg, feats = model(x)

    macs = None
    try:
        from thop import profile as thop_profile

        class _Fwd(torch.nn.Module):
            def __init__(self, m):
                super().__init__()
                self.m = m

            def forward(self, t):
                return self.m(t)

        macs, _ = thop_profile(_Fwd(model), inputs=(x,), verbose=False)
        macs /= batch
    except ImportError:
        print("[warn] thop not installed -- MACs not measured (pip install thop)")

    lines = []
    lines.append(f"config          : {', '.join(cfg_paths)}")
    lines.append(f"input           : {batch}x3x{imgsz}x{imgsz}")
    lines.append(f"levels          : {model.levels}  strides={model.strides}")
    lines.append(f"token budget    : {getattr(model.selector, 'num_tokens', 0)} "
                 f"(levels {model.token_levels})" if model.use_token else "token path   : DISABLED")
    lines.append("")
    lines.append(f"{'module':28s}{'params':>12s}{'share':>9s}")
    lines.append("-" * 49)
    for k, v in sorted(groups.items(), key=lambda kv: -kv[1]):
        lines.append(f"{k:28s}{v:12,d}{100.0 * v / total:8.1f}%")
    lines.append("-" * 49)
    lines.append(f"{'TOTAL (train)':28s}{total:12,d}")
    lines.append(f"{'TOTAL (deploy, no EMA)':28s}{deploy:12,d}   -> {human(deploy)}")
    if macs:
        lines.append(f"{'MACs':28s}{macs:12,.0f}   -> {human(macs)}  (~{human(2 * macs)} FLOPs)")
    lines.append("")
    lines.append("feature map shapes:")
    for lv, f in zip(model.levels, feats):
        lines.append(f"  {lv}: {tuple(f.shape)}")

    lo, hi = TARGET_PARAMS
    verdict = "OK" if lo <= deploy <= hi else ("UNDER" if deploy < lo else "OVER BUDGET")
    lines.append("")
    lines.append(f"param budget [{human(lo)}, {human(hi)}] -> {verdict}")
    if macs:
        lines.append(f"MACs @{imgsz}: {human(macs)}  (thop excludes the attention matmuls "
                     f"q@k and attn@v; add ~0.06G at 640 for Main)")

    text = "\n".join(lines)
    print(text)
    if out:
        d = Path(out)
        d.mkdir(parents=True, exist_ok=True)
        (d / "flops_params.txt").write_text(text, encoding="utf-8")
        (d / "model_summary.txt").write_text(str(model), encoding="utf-8")
        print(f"\nwritten to {d}/flops_params.txt and model_summary.txt")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--search", action="store_true", help="analytic backbone sweep (no torch)")
    ap.add_argument("--config", nargs="*", default=[])
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    if a.search or not a.config:
        run_search()
        if not a.config:
            return
    run_profile(a.config, a.imgsz, a.out, a.batch)


if __name__ == "__main__":
    main()
