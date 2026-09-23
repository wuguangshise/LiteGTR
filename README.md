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

## Training entry points

Two entry points, identical training logic (both drive `engine/trainer.py`):

| | how it is configured | use it for |
|---|---|---|
| **`train_litegtr.py`** | constants at the top of the file; edit and run | day-to-day experiments |
| `tools/train.py` | everything from YAML, passed on the command line | batch runs, scheduling scripts |

```bash
python train_litegtr.py        # set DATA_ROOT and MODEL_CONFIG at the top first
```

To run an ablation, change `MODEL_CONFIG` and `NAME` together; keep every training
constant identical across runs so differences are attributable to the model.
To resume, set `RESUME` to `runs/train/<NAME>/weights/last.pt` with all other
constants unchanged -- the model config must match, or resuming fails loudly.

## Quick start

**1. Check the budget before anything else.**

```bash
python tools/profile_model.py --search                              # analytic sweep, no torch needed
python tools/profile_model.py --config configs/models/model_main.yaml --out runs/profile
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

**4. Check the token budget early.**

```bash
python tools/train.py --config configs/datasets/visdrone_rgb.yaml \
    configs/ablation/token_budget_256.yaml --name vd_budget_256
```

Compare against the main model (56 tokens). If 256 is clearly better, the main
model should use more tokens — find that out before the rest of the ablations.
The five ablations the paper needs are indexed in `configs/ablation/README.md`.

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
losses/     qfl · giou · dfl · token_consistency · token_routing
assigners/  task_aligned_assigner
engine/     trainer · evaluator (loop) · ema · checkpoint · recorder
tools/      profile · analyze_dataset · visualize_labels · train · val · test
            run_seeds · export_onnx · benchmark_latency · submit_visdrone · inspect_tokens · visualize_tokens
train_litegtr.py   edit-and-run training entry (constants at the top)
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
