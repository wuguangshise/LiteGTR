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

**2. Prepare data.**

VisDrone needs no preprocessing — point `configs/datasets/visdrone_rgb.yaml` at it.

DroneVehicle needs two passes, then an **eyeball check** (see DESIGN.md P0-5):

```bash
python -m datasets.prepare.crop_dronevehicle_border \
    --src  RAW/train/rgb      --dst  PREP/train/rgb \
    --ann-src RAW/train/rgb_xml --ann-dst PREP/train/rgb_xml
python -m datasets.prepare.obb_to_hbb \
    --ann-dir PREP/train/rgb_xml --out-dir PREP/train/hbb_labels

python tools/visualize_labels.py --config configs/datasets/dronevehicle_rgb.yaml \
    --split train --num 20 --out runs/label_check
```

Open `runs/label_check/`. Boxes must sit **on** the vehicles. A missing
coordinate shift does not raise an error — it just trains a wrong model.

**3. Sweep the token budget before the main experiment.**

```bash
for b in 56 128 256 512; do
  python tools/train.py --config configs/datasets/visdrone_rgb.yaml \
      configs/ablation/token_budget_$b.yaml --name vd_budget_$b
done
```

If accuracy is flat across budgets, the global path is not earning its place —
fix that before running anything else.

**4. Train / evaluate / export.**

```bash
python tools/train.py --config configs/datasets/visdrone_rgb.yaml configs/models/model_main.yaml
python tools/val.py   --config configs/datasets/dronevehicle_rgb.yaml configs/models/model_main.yaml \
                      --weights runs/train/<name>/weights/best.pt
python tools/run_seeds.py --config configs/datasets/visdrone_rgb.yaml configs/models/model_main.yaml \
                          --seeds 0 1 2
python tools/export_onnx.py --config configs/models/model_main.yaml --weights ... --simplify
python tools/benchmark_latency.py --config configs/models/model_main.yaml --imgsz 640
```

## Layout

```
configs/    _base_ / datasets / models / ablation     — every ablation is a YAML key
datasets/   loaders, transforms, metric definitions, prepare/ scripts
models/     backbone · neck · token (selector, mixer, writeback, ema) · head · build
losses/     qfl · giou · dfl · token_consistency
assigners/  task_aligned_assigner
engine/     trainer · evaluator (loop) · checkpoint · recorder
tools/      profile · visualize_labels · train · val · test · run_seeds · export_onnx · benchmark_latency
tests/      param-budget guard · static-ONNX guard · shape/backward smoke
```

## Run output

```
runs/train/<name>/
├── weights/{best,last,epoch_xx}.pt
├── results.csv / results.png
├── results_by_condition.csv      # day / night / dark breakdown
├── token_stats.csv               # score entropy, EMA agreement
├── args.yaml / model_summary.txt / flops_params.txt / training.log
```

## Notes

* Windows-native: `pathlib` throughout, no shell dependencies, main-guarded entry points.
* Dataset roots live in YAML, never in code.
* `pytest tests/` — `test_param_budget.py` fails if the model outgrows its budget;
  `test_onnx_export.py` fails if the token path acquires dynamic shapes.
