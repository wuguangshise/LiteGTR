"""Run the same config under N seeds and report mean +/- std (P2-14).

A single best run is not a result -- on DroneVehicle especially, where `car`
dominates the 5 classes, seed variance is large enough to swallow the effect
size of an ablation.
"""
from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
KEYS = ["mAP50_95", "mAP50", "AP_small", "AP_medium", "AP_large"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", nargs="+", required=True)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--name", default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    base = a.name or Path(a.config[-1]).stem
    runs = []
    for s in a.seeds:
        name = f"{base}_seed{s}"
        cmd = [sys.executable, str(ROOT / "tools" / "train.py"),
               "--config", *a.config, "--seed", str(s), "--name", name]
        if a.device:
            cmd += ["--device", a.device]
        print("=" * 70)
        print(" ".join(cmd))
        if a.dry_run:
            continue
        subprocess.run(cmd, check=True, cwd=ROOT)
        runs.append(ROOT / "runs" / "train" / name / "best_metrics.json")

    if a.dry_run:
        return
    rows = [json.loads(p.read_text(encoding="utf-8")) for p in runs if p.exists()]
    if not rows:
        print("no best_metrics.json found")
        return

    print("\n" + "=" * 70)
    print(f"{base}  ({len(rows)} seeds: {a.seeds})")
    print(f"{'metric':12s}{'mean':>10s}{'std':>9s}{'min':>9s}{'max':>9s}")
    print("-" * 49)
    summary = {}
    for k in KEYS:
        vals = [r[k] for r in rows if k in r]
        if not vals:
            continue
        m = statistics.mean(vals)
        sd = statistics.stdev(vals) if len(vals) > 1 else 0.0
        summary[k] = {"mean": m, "std": sd, "min": min(vals), "max": max(vals), "values": vals}
        print(f"{k:12s}{m:10.4f}{sd:9.4f}{min(vals):9.4f}{max(vals):9.4f}")

    out = ROOT / "runs" / "train" / f"{base}_seed_summary.json"
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nwritten {out}")
    print("Report mean +/- std in the paper, not the best run.")


if __name__ == "__main__":
    main()
