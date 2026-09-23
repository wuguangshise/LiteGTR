"""Token-routing heatmaps for the paper: where does the budget go?

For each image, one row per checkpoint:

    input + GT | P3 score | P4 score | P5 score | [P3/P4/P5 write-back change]

* **score**   ``sigmoid(score)`` of the scorer -- the value the score gate
  multiplies each token by -- overlaid on the input, with the tokens top-k
  actually selected drawn as dots.
* **change**  ``||f' - f||`` per position (``--delta``): how much the global
  path rewrote the feature map there. This is the effect, the score is the cause.

Pass several ``--weights`` to compare runs side by side -- e.g. the unsupervised
run (``ablation/no_routing_supervision``) against the main model: flat maps with
scattered dots versus maps peaked on the objects.

Scores are drawn on an ABSOLUTE 0..1 scale by default. Do not switch to
``--norm minmax`` for that comparison: min-max stretching turns a flat map with
a 0.01 ripple into a full-contrast picture and hides exactly the collapse the
figure is meant to show.

Besides the PNGs, every image gets an ``.npz`` with the raw maps, token
coordinates and GT boxes, so the figure can be redrawn in any plotting tool.

    python tools/visualize_tokens.py --config configs/datasets/visdrone_rgb.yaml \\
        configs/models/model_main.yaml \\
        --weights runs/train/abl_no_routing_sup/weights/best.pt runs/train/main/weights/best.pt \\
        --labels "w/o supervision" "ours" \\
        --data-root D:/dataset/VisDrone/LiteGTR --num 12 --delta --out runs/token_vis
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from models.build import build_model, load_config  # noqa: E402


@torch.no_grad()
def token_maps(model, image: torch.Tensor) -> dict:
    """Score maps, selected token positions and write-back change for ONE image.

    ``image`` is ``(3, H, W)`` in [0, 1], as the dataset returns it. Mirrors
    ``LiteGTR.extract_feats`` so the maps are exactly what routing saw.
    """
    x = image.unsqueeze(0).to(next(model.parameters()).device)
    feats = model.neck(model.backbone(x))
    if model.local_path is not None:
        feats = model.local_path(feats)
    fmap = dict(zip(model.levels, feats))
    sel = model.selector({lv: fmap[lv] for lv in model.token_levels})

    out: dict = {"score": {}, "tokens": {}, "delta": {}}
    level_ids = sel["level_ids"].long().cpu()
    coords = sel["coords"][0].cpu()
    for i, lv in enumerate(model.token_levels):
        out["score"][lv] = torch.sigmoid(sel["score_maps"][lv][0, 0].float()).cpu().numpy()
        out["tokens"][lv] = coords[level_ids == i].numpy()             # (k, 2) normalised x, y

    mixed = model.mixer(sel["tokens"], sel["coords"], sel["level_ids"])
    new = model.writeback(dict(fmap), mixed, sel["coords"])
    for lv in model.writeback.blocks:
        out["delta"][lv] = (new[lv] - fmap[lv])[0].float().norm(dim=0).cpu().numpy()
    return out


def _to_bgr(image: torch.Tensor) -> np.ndarray:
    img = (image.clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    return np.ascontiguousarray(img[:, :, ::-1])                        # RGB -> BGR


def _overlay(base: np.ndarray, m: np.ndarray, lo: float, hi: float, alpha: float) -> np.ndarray:
    h, w = base.shape[:2]
    m = np.clip((m - lo) / max(hi - lo, 1e-12), 0, 1)
    m = cv2.resize(m.astype(np.float32), (w, h), interpolation=cv2.INTER_NEAREST)
    color = cv2.applyColorMap((m * 255).astype(np.uint8), cv2.COLORMAP_JET)
    return cv2.addWeighted(base, 1 - alpha, color, alpha, 0)


def _caption(img: np.ndarray, text: str) -> np.ndarray:
    bar = np.full((28, img.shape[1], 3), 255, np.uint8)
    cv2.putText(bar, text, (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1, cv2.LINE_AA)
    return np.vstack([bar, img])


def _colorbar(height: int, lo: float, hi: float) -> np.ndarray:
    ramp = np.linspace(255, 0, height).astype(np.uint8)[:, None].repeat(18, 1)
    bar = cv2.applyColorMap(ramp, cv2.COLORMAP_JET)
    canvas = np.full((height, 70, 3), 255, np.uint8)
    canvas[:, 4:22] = bar
    for y, v in ((12, hi), (height - 4, lo)):
        cv2.putText(canvas, f"{v:.2g}", (26, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)
    return canvas


def render_row(image: torch.Tensor, gt_boxes: np.ndarray, maps: dict, label: str,
               norm: str = "abs", alpha: float = 0.5, delta: bool = False) -> dict[str, np.ndarray]:
    """Panels for one checkpoint on one image, keyed by name (``input``, ``score_P3`` ...)."""
    base = _to_bgr(image)
    h, w = base.shape[:2]
    panels: dict[str, np.ndarray] = {}

    gt = base.copy()
    for x0, y0, x1, y1 in np.asarray(gt_boxes).reshape(-1, 4).astype(int):
        cv2.rectangle(gt, (x0, y0), (x1, y1), (0, 255, 0), 1)
    panels["input"] = _caption(gt, "input + GT")

    for lv, s in maps["score"].items():
        lo, hi = (0.0, 1.0) if norm == "abs" else (float(s.min()), float(s.max()))
        p = _overlay(base, s, lo, hi, alpha)
        for tx, ty in maps["tokens"][lv]:
            c = (int(tx * w), int(ty * h))
            cv2.circle(p, c, 4, (0, 0, 0), -1, cv2.LINE_AA)
            cv2.circle(p, c, 3, (255, 255, 255), -1, cv2.LINE_AA)
        panels[f"score_{lv}"] = _caption(p, f"{label} | {lv} score ({len(maps['tokens'][lv])} tok)")

    if delta:
        for lv, d in maps["delta"].items():
            p = _overlay(base, d, 0.0, float(d.max()) or 1.0, alpha)
            panels[f"delta_{lv}"] = _caption(p, f"{label} | {lv} change")
    return panels


def compose(rows: list[dict[str, np.ndarray]], norm: str) -> np.ndarray:
    lines = []
    for panels in rows:
        imgs = list(panels.values())
        line = np.hstack(imgs)
        if norm == "abs":
            line = np.hstack([line, _colorbar(line.shape[0], 0.0, 1.0)])
        lines.append(line)
    width = max(l.shape[1] for l in lines)
    lines = [np.pad(l, ((0, 0), (0, width - l.shape[1]), (0, 0)), constant_values=255) for l in lines]
    return np.vstack(lines)


def load_checkpoint_model(cfg: dict, path: str, raw: bool, device: torch.device):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    model = build_model(cfg)
    sd = ck["model"] if (raw or not ck.get("model_ema")) else ck["model_ema"]
    missing, _ = model.load_state_dict(sd, strict=False)
    real = [k for k in missing if not k.startswith("ema_router.")]
    if real:
        raise RuntimeError(f"{path}: weights do not match the model config, missing {real[:5]} ...")
    if not model.use_token:
        raise RuntimeError(f"{path}: the token path is disabled in this config -- nothing to draw")
    return model.to(device).eval()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", nargs="+", required=True)
    ap.add_argument("--weights", nargs="+", required=True, help="one or more checkpoints, one row each")
    ap.add_argument("--labels", nargs="+", default=None, help="row labels, one per checkpoint")
    ap.add_argument("--data-root", default=None, help="override data.root from the YAML")
    ap.add_argument("--split", default="val")
    ap.add_argument("--num", type=int, default=8, help="first N images of the split")
    ap.add_argument("--indices", type=int, nargs="+", default=None, help="explicit image indices")
    ap.add_argument("--norm", choices=["abs", "minmax"], default="abs")
    ap.add_argument("--alpha", type=float, default=0.5, help="heatmap opacity")
    ap.add_argument("--delta", action="store_true", help="also draw the write-back change maps")
    ap.add_argument("--raw", action="store_true", help="raw weights instead of the EMA weights")
    ap.add_argument("--out", default="runs/token_vis")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()

    from datasets.builder import build_dataset

    labels = a.labels or [Path(w).parent.parent.name for w in a.weights]
    if len(labels) != len(a.weights):
        raise SystemExit("--labels needs one entry per --weights")
    cfg = load_config(*a.config)
    if a.data_root:
        cfg["data"]["root"] = a.data_root
    ds = build_dataset(cfg, a.split, train=False)
    cfg["model"]["num_classes"] = len(ds.classes)
    device = torch.device(a.device)
    models = [load_checkpoint_model(cfg, w, a.raw, device) for w in a.weights]

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    indices = a.indices if a.indices is not None else list(range(min(a.num, len(ds))))
    for idx in indices:
        image, target = ds[idx]
        gt = target["boxes"].numpy()
        rows, arrays = [], {"gt_boxes": gt, "image": _to_bgr(image)}
        for model, label in zip(models, labels):
            maps = token_maps(model, image)
            rows.append(render_row(image, gt, maps, label, a.norm, a.alpha, a.delta))
            tag = label.replace(" ", "_")
            for kind in ("score", "tokens", "delta"):
                for lv, v in maps[kind].items():
                    arrays[f"{tag}/{kind}_{lv}"] = v
        cv2.imwrite(str(out / f"{idx:05d}.png"), compose(rows, a.norm))
        np.savez_compressed(out / f"{idx:05d}.npz", **arrays)
        print(f"  {idx:5d}  {len(gt):4d} objects  -> {out / f'{idx:05d}.png'}")
    print(f"\n{len(indices)} images written to {out}/  (PNG figure + NPZ raw maps each)")


if __name__ == "__main__":
    main()
