# LiteGTR

**Li**ghtweight **G**lobal-**T**oken **R**efinement for UAV object detection —
a single-modality (RGB) detector for edge deployment.

```
RGB ─► TinyNeXt ─► C2..C5 ─► projection (+FPN) ─► P2..P5
                                    │
                    ┌───────────────┴───────────────┐
              Local CNN path              fixed-budget token selection
                    │                      (P3/P4/P5, local routing)
                    │                              │
                    │                         Token Mixer
                    │                              │
                    └──── geometry-aware writeback (residual) ────┘
                                    │
                                GFL head
```

Design rationale, locked decisions and measured budgets: **[docs/DESIGN.md](docs/DESIGN.md)**.

## Install

```bash
pip install -r requirements.txt   # install torch separately to match your CUDA
```

## Quick start

**1. Check the budget before anything else.**

```bash
python tools/profile.py --search                              # analytic sweep, no torch needed
python tools/profile.py --config configs/models/model_main.yaml --out runs/profile
```

**2. Measure your data before trusting the token budget.**

```bash
python tools/analyze_dataset.py --config configs/datasets/visdrone_rgb.yaml \
    configs/models/model_main.yaml --split train
```

Prints objects-per-image percentiles, COCO size buckets and class balance, then
compares them against the configured token budget. If it reports fewer tokens
than objects on a typical image, sweep the budget before anything else.

**3. Prepare data.**

VisDrone needs no preprocessing — point `configs/datasets/visdrone_rgb.yaml` at it.

DroneVehicle needs **no preprocessing** — point the config at the original
download. Annotations are read natively from the XML and cached once; the 100-px
white margin is cropped losslessly at load time.

Optional, before the cross-illumination table:

```bash
python -m datasets.prepare.make_conditions \
    --img-dir <root>/val/rgb --out <root>/val/conditions.txt
```

Then check alignment by eye — for either dataset:

```bash
python tools/visualize_labels.py --config configs/datasets/dronevehicle_rgb.yaml \
    --split train --num 20 --out runs/label_check
```

Open `runs/label_check/`. Boxes must sit **on** the vehicles. A border offset that
was never applied does not raise an error — it just trains a wrong model.

**4. Sweep the token budget before the main experiment.**

```bash
for b in 56 128 256 512; do
  python tools/train.py --config configs/datasets/visdrone_rgb.yaml \
      configs/ablation/token_budget_$b.yaml --name vd_budget_$b
done
```

If accuracy is flat across budgets, the global path is not earning its place —
fix that before running anything else.

**5. Train, compare against baselines, evaluate, export.**

```bash
python tools/train.py --config configs/datasets/visdrone_rgb.yaml configs/models/model_main.yaml

# baselines run the SAME neck, head, losses, assigner, augmentation and schedule
python tools/train.py --config configs/datasets/visdrone_rgb.yaml configs/baselines/csp_n.yaml
python tools/train.py --config configs/datasets/visdrone_rgb.yaml configs/ablation/no_global_token.yaml

python tools/val.py   --config configs/datasets/dronevehicle_rgb.yaml configs/models/model_main.yaml \
                      --weights runs/train/<name>/weights/best.pt
python tools/run_seeds.py --config configs/datasets/visdrone_rgb.yaml configs/models/model_main.yaml \
                          --seeds 0 1 2
python tools/export_onnx.py --config configs/models/model_main.yaml --weights ... --simplify
python tools/benchmark_latency.py --config configs/models/model_main.yaml --imgsz 640
python tools/submit_visdrone.py --config configs/datasets/visdrone_rgb.yaml \
    configs/models/model_main.yaml --weights ... --split test --out runs/submit
```

## Layout

```
configs/    _base_ / datasets / models / ablation / baselines   — every variant is a YAML key
datasets/   base · builder · visdrone · dronevehicle · transforms · metrics · prepare/
models/     backbone (tinynext + builder) · baselines/ · neck · token · head · detector · build
losses/     qfl · giou · dfl · token_consistency
assigners/  task_aligned_assigner
engine/     trainer · evaluator (loop) · ema · checkpoint · recorder
tools/      profile · analyze_dataset · visualize_labels · train · val · test
            run_seeds · export_onnx · benchmark_latency · submit_visdrone
utils/      budget (analytic) · boxes (letterbox inverse, IoU) · plots
tests/      param-budget guard · static-ONNX guard · shape/backward · baselines · boxes · config
```

## Run output

```
runs/train/<name>/
├── weights/{best,last,epoch_xx}.pt
├── results.csv / results.png
├── results_by_condition.csv      # day / night / dark breakdown
├── token_stats.csv               # score entropy, EMA agreement
├── confusion_matrix.png / pr_curve.png / val_predictions/
├── args.yaml / model_summary.txt / flops_params.txt / training.log
```

## Notes

* Windows-native: `pathlib` throughout, no shell dependencies, main-guarded entry points.
* Dataset roots live in YAML, never in code.
* `pytest tests/` — `test_param_budget.py` fails if the model outgrows its budget;
  `test_onnx_export.py` fails if the token path acquires dynamic shapes.
