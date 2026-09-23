# LiteGTR — Design Decisions

Lightweight Global-Token Refinement for UAV object detection.
This file is the authoritative record of *why* the code looks the way it does.
Every `P0-x` / `P1-x` / `P2-x` tag below is referenced from the source.

---

## 0. One-line summary

TinyNeXt extracts efficient local features; a **fixed budget** of globally-mixed
tokens models cross-region relations; **geometry-aware writeback** returns that
context without smearing small objects; a GFL head detects. VisDrone (RGB)
carries the method; DroneVehicle-RGB tests robustness **across illumination
conditions**.

---

## 1. Measured findings that changed the plan

Run `python tools/profile.py --search` to reproduce.

| backbone | params | MACs @640 | per-stage params (M) |
|---|---|---|---|
| `[32,64,128,256] × [4,4,9,4]` (original) | **3.76M** | **4.95G** | 0.04 / 0.15 / 1.25 / **2.16** |
| `[32,64,128,192] × [2,4,8,2]` (**Main, locked**) | 2.03M | 3.60G | 0.02 / 0.15 / 1.11 / 0.61 |
| `[24,48,96,160] × [2,3,6,2]` (**Edge-S, locked**) | 1.07M | 1.73G | 0.01 / 0.06 / 0.48 / 0.43 |

Two things the original plan could not have known without running this:

1. **Params.** The backbone alone was 3.76M against a 4.1–4.8M target, leaving
   ~0.3–1.0M for neck + tokens + head — while the GFL head alone needs
   0.3–0.5M. The token path, i.e. the contribution, had no budget. Stage 4 was
   2.16M of it and matters least for small objects, so Stage 4 is what got cut.
2. **MACs (worse).** The backbone alone was **4.95G**, i.e. the entire "≈5G MACs"
   target was spent before the neck, token path and head existed. At 640 input
   P2 is 160×160, and a single 3×3 conv at 64 channels there costs ≈0.94G MACs;
   two stem convs plus a head on P2 approaches 2G on their own.

**Conclusion: the "≈5G MACs @640" target is not attainable with a genuine P2
branch.** Either state a realistic figure (≈8–10G MACs / 16–20 GFLOPs, still
YOLOv8n-class) or drop P2. The code keeps P2 and makes it cheap; `tools/profile.py`
prints the real number so the paper never quotes an aspirational one.

---

## 2. P0 — locked before implementation

### P0-1 Backbone re-balanced
`models/backbone/tinynext.py`, `utils/budget.py`. Stage 4 depth 4→2 and width
256→192. Enforced by `tests/test_param_budget.py`, which fails the build if the
backbone exceeds 60% of deployment parameters.

### P0-2 P2 is cheap by construction
`models/head/gfl_head.py` gives P2 **one** depthwise-separable stem conv
(`p2_stacked_convs: 1`) while P3–P5 get the full stack, and the final 1×1
predictors are **shared across levels**. P2 also feeds the local CNN path only —
it is excluded from token selection and writeback (`token.levels: [P3,P4,P5]`).
`configs/ablation/no_p2.yaml` switches the whole branch off.

### P0-3 Token budget is a variable, and it is swept first
`configs/ablation/token_budget_{56,128,256,512}.yaml`.

The risk being managed: VisDrone averages ~53 objects per image and exceeds 300
in dense scenes, and P3 at 640 input is 80×80 = 6400 positions. A 56-token
budget is a 200:1 compression — enough for scene-level semantics, probably not
for small-object context. If `no_global_token` barely moves the metric, the
paper has no core claim, and that must be discovered **before** the main
experiments, not after.

Selection uses **local candidate routing**: the map is split into a `G×G` grid
and the top-`k` of each region is kept (`k = budget / G²`), so tokens stay
spatially distributed instead of collapsing onto a few salient blobs.

### P0-4 Token routing must learn, and the EMA teacher must see a different view

`models/token/token_selector.py`, `models/token/ema_token_router.py`,
`losses/token_consistency.py`. Two defects in the first implementation, both
found from a real training log (`loss_token` flickering between `1e-05` and `0.0`
from epoch ~50):

**1. The scorer was never trained on the task.** `top-k` supplies only indices,
and indices are discrete. The selected tokens were plain `gather`s of the feature
map and the score values were never used downstream, so the detection loss sent
gradient into the features and **none into the scorer**. Routing was effectively
random within each grid region.

*Fix — score gating.* Each selected token is scaled by `sigmoid(score)`, which
gives the detection loss a path into the scorer. `top-k` itself stays a pure
forward op with a static `k`; the exported graph gains one sigmoid and one
multiply. `score_gate: false` is retained as the random-routing baseline.

**2. The EMA consistency loss constrained nothing.** The teacher (an EMA of the
student's scorers) received the student's *exact input*. The loss therefore
measured only the drift between the network and a ~1000-step-old copy of itself
on identical data, which decays to zero on its own as training converges and the
learning rate falls. Mean Teacher, BYOL and DINO work because of **asymmetry** —
different views, and a loss that says the output must not depend on the
difference. Without it there is nothing to be consistent about.

*Fix — asymmetric photometric view.* The student keeps its normal input, so the
detection loss is untouched and the consistency term is the only new ingredient.
The teacher sees a photometric perturbation of the same batch (brightness,
contrast, gamma, sensor noise). Geometry is unchanged, so the two score maps
align pixel-for-pixel. The perturbations model illumination change and low-light
noise, so the constraint reads *token selection should not depend on lighting* —
the property the DroneVehicle day/night evaluation measures (P1-10).

The teacher branch costs one extra backbone/neck forward (no backward). BatchNorm
running statistics are preserved across it, so the perturbed batch never leaks
into the statistics used at validation.

*The gradient question is unchanged:* only the continuous score maps are aligned
(spatial-softmax KL); `top-k` never enters the graph, train and inference run the
identical selection, and the teacher is stripped from `deploy_state_dict()`.

| config | score_gate | EMA view | what it isolates |
|---|---|---|---|
| `ablation/random_routing` | off | off | is learned selection better than random? |
| `ablation/no_ema_routing` | on | off | learned routing, no consistency |
| `models/model_main` | on | photometric | + illumination-invariant routing |
| `ablation/ema_same_view` | on | same | shows why the asymmetric view is needed |

### P0-5 DroneVehicle: read the XML, rewrite nothing

Three traps, all handled inside `datasets/dronevehicle.py` rather than by an
offline pipeline:

1. **100-px white border.** Images ship at 840×712 with 640×512 of content, and
   **annotations are in padded coordinates**. Cropping without shifting labels
   raises no error — it just trains a wrong model. The border is derived from the
   XML's declared size (no image decode needed for the known case) and applied as
   a **numpy slice at load time**.

   The earlier pipeline decoded and *re-encoded* every image to crop it. On a
   JPEG source that round trip adds compression artefacts at exactly the scale a
   12-px vehicle occupies — a bad trade in a paper about small objects, and
   unnecessary, since the slice is free and lossless.
   `datasets/prepare/crop_dronevehicle_border.py` is retained but deprecated.

2. **OBB → HBB.** Handled in memory from `polygon`, `robndbox` or `bndbox`, plus
   the released annotations' class-name typos (`feright_car` → `freight_car`).
   `datasets/prepare/obb_to_hbb.py` is now an optional export, not a required step.

3. **Class imbalance.** `car` dominates the five classes, so seed variance is
   large. The cache build prints the distribution; 3 seeds are mandatory.

**Why a cache instead of a conversion** (`datasets/label_cache.py`): parsing XML
per sample per epoch makes the DataLoader the bottleneck at ~28k images × 300
epochs, but converting to flat label files permanently discards the rotation
angle and creates a derived artefact that can drift out of sync. The cache keeps
the XML as the single source of truth, parses once, and stores the **oriented**
boxes — so changing the OBB→HBB policy, or adding an oriented head later, is a
code change rather than a data-regeneration job (`DroneVehicleDataset.obb()`).

---

## 3. P1 — paper-level decisions

### P1-6 Pretraining: unified protocol, not a ban
Fairness comes from **retraining every baseline under identical settings**, not
from forbidding pretraining. Published baselines are almost all reported with
COCO pretraining; a from-scratch number compared against those looks weak and
invites the obvious reviewer question. Main tables use one protocol for all
methods; a pretrained comparison goes in an appendix table.

### P1-7 Primary contribution: geometry-aware writeback
`models/token/geometric_writeback.py`. A token summarising a 12-px car must not
smear across the map; a token summarising scene context should. Each token
carries its source centre `c_i` from the selector and **predicts its own spatial
extent** `σ_i`; writeback attention adds an anisotropic Gaussian prior to the
content logits:

```
C(p,i) = <W_q f_p, W_k t_i> / sqrt(D)
G(p,i) = -0.5 * [ (qx_p-cx_i)² / sx_i² + (qy_p-cy_i)² / sy_i² ]
A(p,·) = softmax_i( C(p,i) + G(p,i) )
f'_p   = f_p + γ · Σ_i A(p,i) · W_v t_i
```

`γ` is initialised to **0**, so training starts from the pure local-CNN solution
and the global path phases in. The residual form is what protects small-object
detail: nothing is replaced, only added.

This lives at inference time and is ablatable three ways
(`writeback_mode: geometric | content | broadcast`), which is what EMA routing —
a training-time technique — could not provide as a headline claim. EMA routing
is demoted to a training-strategy section.

### P1-8 Edge deployment is a constraint, not a slogan
`top-k` and `gather` export poorly under dynamic shapes and fall back on
TensorRT — the classic way a sparse-token model loses in latency what it won in
FLOPs. So: fixed `k`, static shapes, no data-dependent control flow.
`tools/export_onnx.py` exports with `dynamic_axes=None` and
`tests/test_onnx_export.py` fails if any dynamic dim appears. This is the real
argument for a *fixed* budget — deployment friendliness, not just FLOP savings.
`tools/benchmark_latency.py` measures a real device; MACs are not evidence.

### P1-9 VisDrone evaluation protocol (locked)
Class 0 `ignored regions` are painted with the letterbox pad value
(`ignore_mode: mask`); class 11 `others` is dropped; classes 1–10 map to 0–9.
Metrics come from the **COCO API** on `val`. The official MATLAB toolkit gives
slightly different numbers — pick one and never mix them across tables.
`test-dev` requires online submission and is not used for ablations.

### P1-10 DroneVehicle = cross-illumination robustness
HBB numbers are not comparable to published OBB results, so a leaderboard framing
would be a dead end. Instead: VisDrone is daytime and general; DroneVehicle
contains substantial night and low-light imagery. Each image is tagged
`day`/`night`/`dark` (from `conditions.txt` if present, otherwise by luminance)
and `engine/evaluator.py` reports per-condition mAP.

The payoff is a **cross ablation**: in darkness RGB texture degrades, the local
CNN path loses signal, and global context should matter *more*. If
`no_global_token` drops further at night than by day, that is stronger evidence
for the core mechanism than a fractional mAP gain. `configs/datasets/dronevehicle_ir.yaml`
adds an IR-only control (dataloader change only) to quantify how hard night RGB
is and to answer "you had paired IR, why not use it?".

---

## 3b. Baselines are code in this repo, not quoted numbers

`models/baselines/csp_baseline.py`, `configs/baselines/`.

P1-6 claims fairness through a unified protocol. That claim is only credible if
the baselines are actually trained here. Published YOLO numbers come with COCO
pretraining, different input sizes and different NMS settings; comparing against
them proves nothing about the architecture.

Each baseline swaps **only the feature extractor** and reuses LiteGTR's neck,
head, losses, assigner, augmentation and schedule unchanged
(`models/backbone/builder.py` is the seam). `tests/test_baselines.py` asserts
the shared path — same head type, same `reg_max`, same assigner, same strides —
so a baseline cannot silently drift out of the protocol.

| config | what it isolates |
|---|---|
| `baselines/csp_n.yaml` | YOLOv8n-class extractor vs TinyNeXt, token path off |
| `baselines/csp_t.yaml` | smaller CSP, the Edge-S comparison point |
| `ablation/no_global_token.yaml` | TinyNeXt with the token path off: separates the backbone's gain from the token path's gain. One run serves both the baseline table and the ablation table. |

---

## 4. P2 — engineering

| | |
|---|---|
| **P2-11** | `datasets/metrics.py` holds metric *definitions*; `engine/evaluator.py` holds the *loop*. No more duplicate `evaluator.py`. |
| **P2-12** | `neck.use_fpn` defaults **true**. Pure 1×1 projection does no cross-scale fusion and costs small-object AP; `configs/ablation/no_fpn.yaml` measures it. |
| **P2-13** | `models/build.py` — YAML-driven construction with deep `_base_` merging. Ablations flip a key; no code is edited. |
| **P2-14** | `tools/run_seeds.py` — 3 seeds, reports mean ± std. |
| **P2-15** | `datasets/builder.py` — dataset factory extracted, so building VisDrone no longer imports the DroneVehicle module. |
| **P2-16** | `utils/boxes.py` — letterbox inverse. Internal evaluation is self-consistent in letterboxed space, but submissions and qualitative overlays need original pixels; getting this wrong fails silently. |
| **P2-17** | `engine/ema.py` — weight EMA, kept clearly separate from the token-routing EMA teacher. Both are ablated independently. |

**Windows**: `pathlib` throughout, no shell dependencies, dataset roots live in
YAML, every entry point is `if __name__ == "__main__"` guarded for spawn-based
DataLoader workers.

---

## 5. Build order

1. `tools/profile.py` — **done**, numbers above.
2. Lock token/projection dims against the measured budget, and run
   `tools/analyze_dataset.py` so the budget sweep starts from your data's real
   objects-per-image distribution rather than a quoted average.
3. Geometry writeback + EMA alignment maths — **done** (P1-7, P0-4).
4. Data adaptation, then **look at `tools/visualize_labels.py` output**.
5. Skeleton + unit tests + ONNX smoke — **done**.
6. **Token budget sweep** — before any main experiment.
7. Main → cross-illumination → full ablations → real-device latency.

## 6. Experiment matrix

Every ablation lives in `configs/ablation/` and changes exactly one variable relative to
`models/model_main.yaml`. `configs/ablation/README.md` indexes them by paper table with the
question each one answers and a suggested run order.

| group | config | question |
|---|---|---|
| Main vs baselines | `models/model_main.yaml` | accuracy at equal params/FLOPs |
| Token budget | `ablation/token_budget_*` | **does the curve have a knee at all?** |
| No global token | `ablation/no_global_token` | does global modelling matter |
| No local CNN | `ablation/no_local_cnn` | does the local path matter |
| No P2 | `ablation/no_p2` | is the high-res branch worth its MACs |
| Writeback mode | `no_geometric_writeback`, `broadcast_writeback` | **isolates the primary claim** |
| No FPN | `ablation/no_fpn` | cross-scale fusion value |
| Token source levels | `ablation/token_src_p5`, `token_src_p4p5` | do the finer levels need to contribute tokens? (budget and write-back fixed) |
| Routing locality | `ablation/global_topk_routing` | does local candidate routing matter vs. one global top-k? |
| Token interaction | `ablation/mixer_none`, `mixer_deep` | **does token-to-token attention matter**; is Transformer capacity the bottleneck? |
| Write-back into P2 | `ablation/writeback_p2` | does global context help at stride 4, where small objects live? |
| Random routing | `ablation/random_routing` | **is learned selection better than random?** |
| No EMA routing | `ablation/no_ema_routing` | value of the photometric consistency term |
| Same-view EMA | `ablation/ema_same_view` | why the asymmetric view is necessary |
| Cross-illumination | `datasets/dronevehicle_rgb` | day / night / dark breakdown |
| Token × condition | `no_global_token` on DroneVehicle | **does global context matter more at night?** |
| Seeds | `tools/run_seeds.py` | mean ± std over 3 seeds |
