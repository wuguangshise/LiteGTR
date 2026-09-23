"""Is the global token path actually being used? One command, a definite answer.

Reading the residual gate gamma alone is not enough: the writeback adds
``gamma * proj(attn_out)``, so a small gamma can be compensated by large ``proj``
weights. This measures what the path really contributes, with forward hooks on
real validation images -- no model code is modified.

Reported per writeback level:

* ``gamma``            the learned residual gate (starts at 0)
* ``contribution``     ||f' - f|| / ||f||  -- how much the global path actually
                       changes the feature map. This is the number to trust.
* ``sigma``            the spatial extent each token predicts for itself, in
                       pixels. If every token predicts the same sigma, the
                       geometric prior has collapsed into a fixed blur and is not
                       doing what the paper claims.

Also reads the EMA-routing health from the saved run, since ``loss_token``
sitting near zero means that mechanism is inert.

    python tools/inspect_tokens.py --config configs/datasets/visdrone_rgb.yaml \\
        configs/models/model_main.yaml \\
        --weights runs/train/litegtr_visdrone/weights/last.pt \\
        --data-root D:/dataset/VisDrone/LiteGTR
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from datasets.base import collate_fn  # noqa: E402
from datasets.builder import build_dataset  # noqa: E402
from models.build import build_model, load_config  # noqa: E402


def verdict_contribution(r: float) -> str:
    if r < 0.01:
        return "OFF      -- the network has learned to ignore this path"
    if r < 0.05:
        return "WEAK     -- present, but a small share of the signal"
    if r < 0.20:
        return "ACTIVE   -- meaningfully reshaping the features"
    return "STRONG   -- a large share of the feature signal"


@torch.no_grad()
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", nargs="+", required=True)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--data-root", default=None, help="override data.root from the YAML")
    ap.add_argument("--split", default="val")
    ap.add_argument("--num-batches", type=int, default=10)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--raw", action="store_true",
                    help="inspect raw weights instead of the EMA weights training evaluated")
    a = ap.parse_args()

    cfg = load_config(*a.config)
    if a.data_root:
        cfg["data"]["root"] = a.data_root
    ck = torch.load(a.weights, map_location="cpu", weights_only=False)

    # ---------------- gamma straight from the checkpoint ----------------
    print("=" * 70)
    print(f"checkpoint : {a.weights}   (epoch {ck.get('epoch', '?')})")
    print("=" * 70)
    print("\n[1] residual gate gamma, straight from the state dict")
    for tag in ("model", "model_ema"):
        sd = ck.get(tag)
        if not sd:
            continue
        gammas = {k: float(v) for k, v in sd.items() if "writeback" in k and k.endswith(".gamma")}
        if gammas:
            print(f"    {tag}:")
            for k, v in gammas.items():
                print(f"      {k.replace('writeback.blocks.', ''):14s} gamma = {v:+.5f}")
    if not any("writeback" in k for k in ck["model"]):
        print("    no writeback parameters -- this checkpoint has the token path disabled")
        return

    # ---------------- effective contribution, via hooks ----------------
    ds = build_dataset(cfg, a.split, train=False)
    cfg["model"]["num_classes"] = len(ds.classes)
    model = build_model(cfg)
    sd = ck["model"] if (a.raw or not ck.get("model_ema")) else ck["model_ema"]
    model.load_state_dict(sd, strict=False)
    device = torch.device(a.device)
    model.to(device).eval()
    img_size = cfg["data"].get("img_size", 640)

    stats: dict[str, dict[str, list]] = {}

    def make_hook(lv: str):
        def hook(module, inputs, output):
            feat, tokens, _ = inputs
            delta = (output - feat).float()
            f = feat.float()
            ratio = (delta.flatten(1).norm(dim=1) / f.flatten(1).norm(dim=1).clamp_min(1e-12))
            sigma = torch.nn.functional.softplus(module.sigma(tokens.float())) + module.sigma_min
            s = stats.setdefault(lv, {"ratio": [], "sigma": []})
            s["ratio"].append(ratio.cpu())
            s["sigma"].append((sigma * img_size).reshape(-1, 2).cpu())   # normalised -> pixels
        return hook

    handles = [blk.register_forward_hook(make_hook(lv)) for lv, blk in model.writeback.blocks.items()]
    loader = DataLoader(ds, batch_size=a.batch, shuffle=False, collate_fn=collate_fn, num_workers=0)
    n_img = 0
    for bi, (images, _) in enumerate(loader):
        if bi >= a.num_batches:
            break
        model(images.to(device))
        n_img += images.shape[0]
    for h in handles:
        h.remove()

    print(f"\n[2] effective contribution  ||f' - f|| / ||f||   over {n_img} {a.split} images"
          f"   ({'raw' if sd is ck['model'] else 'EMA'} weights)")
    print(f"    mode = {model.writeback.blocks[next(iter(model.writeback.blocks))].mode}")
    overall = []
    for lv in model.writeback.blocks:
        r = torch.cat(stats[lv]["ratio"])
        overall.append(float(r.mean()))
        print(f"      {lv}:  mean {float(r.mean()):.4f}   min {float(r.min()):.4f}   "
              f"max {float(r.max()):.4f}   -> {verdict_contribution(float(r.mean()))}")

    print(f"\n[3] per-token spatial extent sigma (pixels @ {img_size})  -- the geometric prior")
    collapsed = []
    for lv in model.writeback.blocks:
        sg = torch.cat(stats[lv]["sigma"])
        mean, std = sg.mean(0), sg.std(0)
        cv = float((std / mean.clamp_min(1e-9)).mean())
        collapsed.append(cv < 0.05)
        print(f"      {lv}:  sx {float(mean[0]):6.1f} ± {float(std[0]):5.1f}   "
              f"sy {float(mean[1]):6.1f} ± {float(std[1]):5.1f}   "
              f"spread(CV) {cv:.3f}{'   <- COLLAPSED' if cv < 0.05 else ''}")

    # ---------------- EMA routing health from the run log ----------------
    run_dir = Path(a.weights).resolve().parent.parent
    print(f"\n[4] EMA routing health  ({run_dir.name})")
    try:
        rows = list(csv.DictReader(open(run_dir / "results.csv", encoding="utf-8")))
        lt = [float(r["train/loss_token"]) for r in rows if r.get("train/loss_token")]
        if lt:
            print(f"      loss_token: first {lt[0]:.2e}  last {lt[-1]:.2e}  "
                  f"({'INERT -- the consistency loss contributes no gradient' if lt[-1] < 1e-3 else 'active'})")
    except FileNotFoundError:
        print("      results.csv not found next to the weights")
    try:
        rows = list(csv.DictReader(open(run_dir / "token_stats.csv", encoding="utf-8")))
        if rows and rows[-1].get("ema_agreement"):
            ag = float(rows[-1]["ema_agreement"])
            print(f"      ema_agreement (last epoch): {ag:.4f}  "
                  f"{'<- teacher tracks the student almost exactly' if ag > 0.98 else ''}")
    except FileNotFoundError:
        pass

    # ---------------- bottom line ----------------
    m = sum(overall) / len(overall)
    print("\n" + "=" * 70)
    print("BOTTOM LINE")
    if m < 0.01:
        print("  The global path is effectively OFF. Making the Transformer bigger will not")
        print("  help -- first work out why the network routes around it.")
    elif m < 0.05:
        print("  The global path is WEAK. Try MORE TOKENS (budget sweep) before a bigger")
        print("  Transformer: the bottleneck is more likely information than capacity.")
    else:
        print("  The global path is ACTIVE. Scaling it is worth testing -- decouple the token")
        print("  width from neck.channels so P2 does not absorb the extra cost.")
    if any(collapsed):
        print("  sigma has COLLAPSED on some level: every token predicts the same extent, so")
        print("  the geometric prior is acting as a fixed blur rather than a per-token scale.")
    print("=" * 70)


if __name__ == "__main__":
    main()
