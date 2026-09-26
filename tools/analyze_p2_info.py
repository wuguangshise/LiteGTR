"""Where can the P2 path of LiteGTR take in more tiny-object detail? Forward passes only.

No training: images go through the model once and closed-form linear probes (ridge
regression) measure how much of the image detail each feature map still carries.
Detail = the high-pass of the input (image - AvgPool5x5(image)), as the 4x4x3 patch
behind each stride-4 cell. Probes are fitted on one set of images and scored (R^2) on
held-out images, inside ground-truth boxes of small (<32 px) and tiny (<16 px) objects.

Three questions:

  A. flow     R^2 at stem_s2 -> C2 -> lat(C2) -> P2 after FPN -> P2 into the head.
              The biggest drop is where detail is lost.
  B. gain     R^2 of [P2 into the head + source] minus R^2 of P2 alone, per source
              (stem_s2, C2, lat(C2)): the detail a source would ADD to P2 if injected.
  C. routing  the same gain split by the routing mask M (dilated sigmoid of the P3 score
              map, what the token path already computes): is the recoverable detail where
              M is high, and how much of the image does M cover? Reported for the model's
              own score map and for the GT-centre heatmap the scorer is trained towards.
  D. capacity (weights-free) PCA of the detail patches in small-object cells: how many
              dimensions they need vs the 32 channels of C2 / 64 of P2.

    python tools/analyze_p2_info.py --data <VisDrone root> --split val \
        --weights runs/train/<run>/weights/best.pt [--config configs/models/model_main.yaml]

Without --weights it probes the initialisation, which only shows structural bottlenecks.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from datasets.base import collate_fn  # noqa: E402
from datasets.visdrone import VisDroneDataset  # noqa: E402
from losses.token_routing import center_heatmap  # noqa: E402
from models.build import build_model, load_config  # noqa: E402

SOURCES = ("stem_s2", "C2", "lat_C2")
POINTS = SOURCES + ("P2_neck", "P2_head")
TAU = 0.3          # M > TAU counts as "routed"


def high_pass(img):
    return img - F.avg_pool2d(img, 5, stride=1, padding=2, count_include_pad=False)


def box_mask(targets, size, max_side):
    m = torch.zeros(len(targets), 1, size, size)
    for i, t in enumerate(targets):
        for x1, y1, x2, y2 in t["boxes"].tolist():
            if max(x2 - x1, y2 - y1) < max_side:
                m[i, 0, max(int(y1), 0):max(int(y2) + 1, 0), max(int(x1), 0):max(int(x2) + 1, 0)] = 1
    return F.max_pool2d(m, 4).flatten().bool()        # stride-4 cells touching such a box


def dilate_up(m, size):
    m = F.interpolate(m, size=size, mode="nearest")
    return F.max_pool2d(m, 3, stride=1, padding=1)


def load_weights(model, path):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    for key in ("ema", "model_ema", "model", "state_dict"):
        if isinstance(ck, dict) and key in ck:
            ck = ck[key]
            break
    sd = ck.state_dict() if hasattr(ck, "state_dict") else ck
    missing, unexpected = model.load_state_dict(sd, strict=False)
    print(f"loaded {path}: {len(missing)} missing, {len(unexpected)} unexpected keys")


class Tap:
    """Feature maps at the probe points, all brought to stride 4."""

    def __init__(self, model):
        self.model, self.store = model, {}
        s = self.store
        model.backbone.stem[2].register_forward_hook(lambda m, i, o: s.__setitem__("stem_s2", o))
        model.backbone.register_forward_hook(lambda m, i, o: s.__setitem__("C2", o[0]))
        model.neck.lateral[0].register_forward_hook(lambda m, i, o: s.__setitem__("lat_C2", o))
        model.neck.register_forward_hook(lambda m, i, o: s.__setitem__("P2_neck", o[0]))

    @torch.no_grad()
    def __call__(self, imgs):
        feats = self.model.extract_feats(imgs)
        out = {k: self.store[k] for k in ("C2", "lat_C2", "P2_neck")}
        out["stem_s2"] = F.pixel_unshuffle(self.store["stem_s2"], 2)
        out["P2_head"] = feats[0]
        score = self.model._last_student_maps
        m = torch.sigmoid(score["P3"]) if score else None
        return out, m


def unfold3(x):
    return F.unfold(x, 3, padding=1).transpose(1, 2).reshape(-1, 9 * x.shape[1])


class Ridge:
    def __init__(self):
        self.xx = self.xy = None

    def add(self, x, y):
        x = torch.cat([x, torch.ones(len(x), 1, dtype=x.dtype)], 1)
        xx, xy = x.T @ x, x.T @ y
        self.xx = xx if self.xx is None else self.xx + xx
        self.xy = xy if self.xy is None else self.xy + xy

    def solve(self):
        d = self.xx.shape[0]
        a = self.xx + 1e-3 * torch.trace(self.xx) / d * torch.eye(d, dtype=self.xx.dtype)
        self.w = torch.linalg.solve(a, self.xy)

    def predict(self, x):
        return torch.cat([x, torch.ones(len(x), 1, dtype=x.dtype)], 1) @ self.w


def split(ds, n_test=100, seed=0):
    seq = [p.name.split("_")[0] for p in ds.images]
    order = sorted(set(seq))
    random.Random(seed).shuffle(order)
    held, n = set(), 0
    for s in order:
        if n >= n_test:
            break
        held.add(s)
        n += seq.count(s)
    return [i for i, s in enumerate(seq) if s not in held], [i for i, s in enumerate(seq) if s in held]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--split", default="val")
    ap.add_argument("--config", default="configs/models/model_main.yaml")
    ap.add_argument("--weights", default=None)
    ap.add_argument("--fit", type=int, default=200, help="images to fit the probes on")
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", default=None, help="write the results as JSON here")
    args = ap.parse_args()

    cfg = load_config(args.config)
    ds = VisDroneDataset(args.data, split=args.split, train=False, mosaic_prob=0.0, ignore_mode="drop")
    cfg["model"]["num_classes"] = len(ds.classes)
    torch.manual_seed(0)
    model = build_model(cfg)
    if args.weights:
        load_weights(model, args.weights)
    model.eval().to(args.device)
    tap = Tap(model)
    tr, te = split(ds)
    fit_idx = random.Random(1).sample(tr, min(args.fit, len(tr)))

    probes = {p: Ridge() for p in POINTS}
    probes.update({f"P2_head+{s}": Ridge() for s in SOURCES})

    def batches(idx):
        for i in range(0, len(idx), args.batch):
            imgs, tgts = collate_fn([ds[j] for j in idx[i:i + args.batch]])
            yield imgs.to(args.device), tgts

    def design(feats):
        x = {p: unfold3(f).double().cpu() for p, f in feats.items()}
        for s in SOURCES:
            x[f"P2_head+{s}"] = torch.cat([x["P2_head"], x[s]], 1)
        return x

    def target(imgs):
        return F.pixel_unshuffle(high_pass(imgs), 4).permute(0, 2, 3, 1).reshape(-1, 48).double().cpu()

    # ---- fit
    patches = []
    for imgs, tgts in batches(fit_idx):
        feats, _ = tap(imgs)
        sel = box_mask(tgts, imgs.shape[-1], 32)
        y = target(imgs)[sel]
        patches.append(y)
        for p, x in design(feats).items():
            probes[p].add(x[sel], y)
    for r in probes.values():
        r.solve()

    # ---- held-out
    groups = ("small", "tiny", "small&M_model", "small&!M_model", "small&M_gt", "small&!M_gt")
    acc = {(p, g): [0.0, 0.0, 0.0, 0] for p in probes for g in groups}
    cover = {"M_model": [0, 0, 0, 0], "M_gt": [0, 0, 0, 0]}   # small hit, small all, area hit, area all
    for imgs, tgts in batches(te):
        feats, m_model = tap(imgs)
        size = imgs.shape[-1] // 4
        small, tiny = box_mask(tgts, imgs.shape[-1], 32), box_mask(tgts, imgs.shape[-1], 16)
        s3 = imgs.shape[-1] // 8
        m_gt = torch.stack([center_heatmap(t["boxes"], s3, s3, 8.0) for t in tgts])[:, None]
        masks = {"M_gt": dilate_up(m_gt, (size, size)).flatten().cpu() > TAU}
        masks["M_model"] = (dilate_up(m_model, (size, size)).flatten().cpu() > TAU
                            if m_model is not None else torch.zeros_like(masks["M_gt"]))
        sel = {"small": small, "tiny": tiny}
        for k, m in masks.items():
            sel[f"small&{k}"], sel[f"small&!{k}"] = small & m, small & ~m
            c = cover[k]
            c[0] += int((small & m).sum()); c[1] += int(small.sum()); c[2] += int(m.sum()); c[3] += m.numel()
        y = target(imgs)
        for p, x in design(feats).items():
            pred = probes[p].predict(x)
            for g, s in sel.items():
                if not s.any():
                    continue
                a = acc[(p, g)]
                e, t = pred[s] - y[s], y[s]
                a[0] += float((e ** 2).sum()); a[1] += float((t ** 2).sum()); a[2] += float(t.sum()); a[3] += t.numel()

    def r2(p, g):
        sse, s2, s1, n = acc[(p, g)]
        return 1 - sse / (s2 - s1 * s1 / n) if n else float("nan")

    res = {"weights": args.weights or "init"}
    print(f"\nweights: {res['weights']}\n\nA. flow -- R^2 of the image detail, held-out images")
    print(f"  {'point':10s} {'small':>7s} {'tiny':>7s}")
    for p in POINTS:
        res[f"flow/{p}/small"], res[f"flow/{p}/tiny"] = r2(p, "small"), r2(p, "tiny")
        print(f"  {p:10s} {r2(p, 'small'):7.3f} {r2(p, 'tiny'):7.3f}")

    print("\nB/C. gain -- R^2 added to P2 by each source (small objects), split by the routing mask")
    hdr = ["all", "tiny", "M_model", "!M_model", "M_gt", "!M_gt"]
    print(f"  {'source':8s} " + " ".join(f"{h:>9s}" for h in hdr))
    gmap = dict(zip(hdr, ("small", "tiny", "small&M_model", "small&!M_model", "small&M_gt", "small&!M_gt")))
    for s in SOURCES:
        row = []
        for h in hdr:
            g = gmap[h]
            v = r2(f"P2_head+{s}", g) - r2("P2_head", g)
            res[f"gain/{s}/{h}"] = v
            row.append(v)
        print(f"  {s:8s} " + " ".join(f"{v:9.3f}" for v in row))
    for k, (hit, n_small, area, n_all) in cover.items():
        res[f"cover/{k}/small_cells"] = hit / max(n_small, 1)
        res[f"cover/{k}/image_area"] = area / max(n_all, 1)
        print(f"  {k}: covers {hit / max(n_small, 1):.1%} of small-object cells "
              f"with {area / max(n_all, 1):.1%} of the image (M > {TAU})")

    y = torch.cat(patches)
    y = y - y.mean(0)
    ev = torch.linalg.eigvalsh(y.T @ y).flip(0).clamp_min(0)
    cum = ev.cumsum(0) / ev.sum()
    print("\nD. capacity -- dimensions of the 48-d detail patch in small-object cells (weights-free)")
    for q in (0.9, 0.95, 0.99):
        k = int((cum < q).sum()) + 1
        res[f"pca/{q}"] = k
        print(f"  {q:.0%} of the energy: {k} dims")
    if args.out:
        Path(args.out).write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
