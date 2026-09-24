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

Run `python tools/profile_model.py --search` to reproduce.

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

### Measured, full model (CI, `tools/profile_model.py`, thop, 640×640)

Parameters are learnable parameters excluding the training-only EMA teacher
(`models.build.count_deploy_params`); BatchNorm running stats are buffers and are
not counted.

| model | params | MACs | FLOPs | vs its paired baseline |
|---|---|---|---|---|
| `baselines/csp_n` | 2.25M | 3.57G | ~7.1G | -- |
| **Main** | **2.32M** | **4.93G** | ~9.9G | params +3%, **MACs +38%** |
| `baselines/csp_t` | 1.17M | 2.22G | ~4.4G | -- |
| **Edge-S** | 1.22M | 2.49G | ~5.0G | params +4%, MACs +12% |

Main's parameters by module: backbone 2.03M, head 74K, token selector 56K,
write-back 50K, neck 47K, local path 34K, mixer 34K. The EMA teacher (56K) is
training-only.

**What the baseline comparison demands.** At matched parameters Main costs 38%
more compute than csp_n (TinyNeXt's 7x7 depthwise and 4x expansion at high
resolution are compute-heavy per parameter). The accuracy gain has to be large
enough to justify that, and the FLOPs column must sit in the main table next to
params -- not be left for a reviewer to compute. Edge-S's overhead is only 12%,
which makes the small-scale comparison the easier one to win.

thop does not count the attention matmuls (`q@k`, `attn@v`). They add roughly
0.06G at 640 input -- about 5.0G MACs in total. State this in the paper next to
the FLOPs column.

**Correction.** An earlier version of this section concluded that ≈5G MACs was
unattainable with a genuine P2 branch and recommended quoting 8–10G. That was an
*estimate* -- it put the P2 head alone at ~2G. The measurement shows neck, head,
local path and token path together cost 1.33G: P2 has a single
depthwise-separable stem and the predictors are shared across levels. **The
original ≈5G MACs / ≈10 GFLOPs target is met.** For context, YOLOv8n is 3.2M
params / 8.7 GFLOPs; Main has fewer parameters and slightly more FLOPs.

---

## 2. P0 — locked before implementation

### P0-1 Backbone re-balanced
`models/backbone/tinynext.py`, `utils/budget.py`. Stage 4 depth 4→2 and width
256→192. `tests/test_param_budget.py` fails the build if the backbone grows past
2.5M, i.e. drifts back toward the original 3.76M design.

The backbone still accounts for ~87% of deployment parameters (2.03M of 2.33M):
neck, head and token path are deliberately light, and the Transformer part is
only ~4%. An earlier target of "at most 60%" was never met by the locked model.
If the global path proves worth scaling, the levers are more mixer layers and a
token width decoupled from `neck.channels` -- widening the neck would mostly
grow the CNN side, P2 in particular.

### P0-2 P2 is cheap by construction
`models/head/gfl_head.py` gives P2 **one** depthwise-separable stem conv
(`p2_stacked_convs: 1`) while P3–P5 get the full stack, and the final 1×1
predictors are **shared across levels**. P2 also feeds the local CNN path only —
it is excluded from token selection and writeback (`token.levels: [P3,P4,P5]`).
`use_p2: false` switches the whole branch off (exercised in `tests/_variants.py`).

### P0-3 Token budget is a variable, and it is swept first
`configs/ablation/token_budget_256.yaml`, compared against the main model's 56.

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

**3. The gate did not teach the scorer where to look, and the maps went flat.**
Measured on the first 200-epoch VisDrone run (commit `3abbba5`): `score_entropy`
rose monotonically from 0.99998 to 0.999997 over 135 epochs -- a score std of
about 0.01 across the map -- while `ema_agreement` read 0.999 and `loss_token`
sat at ~4e-6. The agreement was trivial: both maps were flat. Three causes stack:

* the `sigmoid(score)` gate is a per-token scale the mixer and write-back can
  absorb, so the detection loss carries almost no information about *which
  position* deserves a token;
* the spatial-softmax KL is zero for any two identical maps, flat ones included,
  so the consistency term actively admits the collapse;
* the scorer's output conv sat in the weight-decay group, which pulls every
  score toward the same value.

A 150-step synthetic check confirms the consequence: without supervision 1.7% of
the selected P3 tokens fall on objects against an object-area share (chance) of
1.5% -- random routing.

*Fix — routing supervision.* Each level's score map is supervised with a
Gaussian GT-centre heatmap under CenterNet's penalty-reduced focal loss
(`losses/token_routing.py`, `routing_sup`). Every level sees every object, so
selection follows object density -- on VisDrone, small objects. The scorer's
output conv is exempt from weight decay (`scorer_no_decay`). On the same
synthetic check, 25.5% of selected tokens land on objects after 150 steps, the
map entropy falls, and the EMA term becomes live (1e-4 -> 2e-2): with peaked
maps, "consistent under a lighting change" is no longer trivially satisfied.
Training-only; the exported graph is unchanged.

| setting | score_gate | EMA view | routing_sup | what it isolates |
|---|---|---|---|---|
| random routing *(test variant)* | off | off | off | is learned selection better than random? |
| `ablation/no_routing_supervision` | on | photometric | off | the first run exactly: gate + EMA alone collapse |
| `ablation/no_ema_routing` | on | off | on | supervised routing, no consistency |
| `models/model_main` | on | photometric | on | + illumination-invariant routing |
| same-view EMA *(test variant)* | on | same | on | shows why the asymmetric view is needed |

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
per sample per epoch makes the DataLoader the bottleneck at ~28k images × 200
epochs, but converting to flat label files permanently discards the rotation
angle and creates a derived artefact that can drift out of sync. The cache keeps
the XML as the single source of truth, parses once, and stores the **oriented**
boxes — so changing the OBB→HBB policy, or adding an oriented head later, is a
code change rather than a data-regeneration job (`DroneVehicleDataset.obb()`).

---

### P0-5b "Too many boxes": what the literature does, what we adopted
`assigners/task_aligned_assigner.py`, `models/detector.py::postprocess`,
`configs/_base_/schedule.yaml` (`assigner`, `test`), `tools/diagnose_predictions.py`.

The first run's `val_predictions/` looked cluttered (grey = GT, colour =
predictions at score >= 0.25). Zoomed in, two distinct things:

* **Nested same-class boxes** on one tall object -- IoU = small / large area slips
  under the NMS threshold -- and **pedestrian + people** on one person, which
  class-wise NMS keeps by design.
* Underneath, a training defect: **the smallest objects were never positives.**
  A GT is a candidate only at grid centres inside it. At 640 input the median
  VisDrone box is ~7.6 px; one narrower than the P2 stride can contain no centre.
  From public size statistics ~9.5% of all GT (43% of those under 4 px) never got
  a positive and were trained as background.

**How published detectors handle it, and what we took.**

| problem | literature | adopted |
|---|---|---|
| tiny GT gets no candidate | NWD (ISPRS'22): assignment is where NWD helps most -- IoU matching leaves 0.72 positives per tiny GT vs 1.05 with NWD. RFLA (ECCV'22): Gaussian receptive-field distance. **STAL** (Ultralytics YOLO26): widen tiny GTs for candidate selection only, regress the original box | **STAL**, `stal_size: 8` (Ultralytics' `stride[1]` for a P2-P5 head). Designed for exactly our assigner (TAL, ltrb/DFL head) and the default in Ultralytics' own `TaskAlignedAssigner` (`select_candidates_in_gts`), so it is also what YOLOv8/11 use when trained with current Ultralytics; GTs >= 8 px are assigned bit-identically. Replaces an ad-hoc nearest-point fallback tried first |
| duplicate boxes (checked in RemDet's repo) | RemDet has no dedicated mechanism: YOLOv8 head, TAL (topk 10, alpha 0.5, beta 6), class-wise NMS 0.7, multi-label; figures are not filtered separately | nothing to adopt. Our drawings use a separate `vis:` filter (agnostic NMS + containment); TAL topk/alpha differ from ours (13 / 1.0) -- a candidate ablation, not changed without evidence |
| evaluation post-processing | RemDet (AAAI'25, mmyolo) and Ultralytics val: score 0.001, **multi-label**, class-wise NMS **0.7**, 300 boxes; LEAF-YOLO: 0.01 / 0.5 | **same as RemDet / Ultralytics** (`test:` block). Our earlier 0.02 / single-label / 0.6 was stricter than every method in Table 1, understating our mAP |
| drawn figures cluttered | every detector's predict mode filters at score 0.25-0.35; evaluation boxes are not meant to be drawn | figures filter at 0.25; `tools/diagnose_predictions.py` counts what the drawn boxes are |
| duplicates at all | YOLOv10 (NeurIPS'24) / YOLO26: one-to-one head, NMS-free | **not adopted** -- see below |
| tiny-box regression | NWD / SAL-NWD loss (DroneScan-YOLO) | **not adopted** -- GIoU keeps a gradient for disjoint boxes; NWD's own ablation puts the gain in assignment, which STAL covers |

*Why not NMS-free.* A one-to-one head keeps a single prediction per object,
which is the literal fix for duplicates and would also remove NMS from the edge
graph (P1-8). But it adds a second head, changes the training objective, and
one-to-one matching is least proven exactly where VisDrone is hardest -- dense
crowds of tiny objects. Worth a dedicated experiment once the main results
exist, not a change to the main model mid-plan.

*Containment and class-agnostic NMS* stay available (`test.containment`,
`test.agnostic`) but off: they are not in the protocol of the methods we compare
with, and whether they help our mAP is measured by the diagnosis tool, not assumed.

Every run from before this fix, including the first 200-epoch run, is obsolete
as a table row; `run_experiments.py` retrains all of them.

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
Class 11 `others` is dropped; classes 1–10 map to 0–9. Class 0 `ignored regions`
are painted with the letterbox pad value for training (`ignore_mode: mask`) but
NOT for evaluation (`eval_ignore_mode: drop`).
Metrics come from the **COCO API** on `val`, matched to mmdet/mmyolo's
`CocoMetric` (RemDet, AAAI'25): boxes in original-image pixels against the original
annotations, `maxDets` 100/300/1000 (AP at 1000), COCO size buckets in original
pixels, plus AI-TOD's AP_vt / AP_t. An earlier version scored in letterboxed
640-px space with pycocotools' default 100 detections per image and painted
validation images: buckets shifted by the resize factor (a "small" object at
640 is up to ~3x larger in the original), recall was capped on dense images, and
ignored regions could never produce false positives. Those numbers are not
comparable and are not reported. The official MATLAB toolkit gives slightly
different numbers again — pick one and never mix them across tables.
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

| config | backbone params | paired with | what it isolates |
|---|---|---|---|
| `baselines/csp_n.yaml` | 2.07M (`[48,96,192,384]`) | Main, TinyNeXt-M 2.03M | CSP vs TinyNeXt at equal params, token path off |
| `baselines/csp_t.yaml` | 1.08M (`[40,80,160,248]`) | Edge-S, TinyNeXt-S 1.07M | the same comparison at the smaller scale |

**Baselines are parameter-matched.** The first version of these configs used
`[32,64,128,256]` (0.92M) and `[24,48,96,192]` (0.52M) -- roughly 0.45x the paired
TinyNeXt backbones. Any LiteGTR win would then have been explainable as "twice
the parameters". Neck and head are shared, so matching the backbone matches the
total. `tests/test_baselines.py` fails if a baseline drifts beyond 10%.

Params and FLOPs cannot be matched simultaneously: at equal parameters the CSP
costs about 2.37G MACs to TinyNeXt-M's 3.60G, because TinyNeXt spends more compute
per parameter (7x7 depthwise and 4x expansion at high resolution). Params are
matched and FLOPs are reported. That is the *stricter* test for LiteGTR -- the
baseline gets equal parameters and less compute -- so the FLOPs column has to be
in the main table, not hidden.

| config | what it isolates |
|---|---|
| `ablation/no_global_token.yaml` | TinyNeXt with the token path off: separates the backbone's gain from the token path's gain. One run serves both the baseline table and the ablation table. |

---

## 4. P2 — engineering

| | |
|---|---|
| **P2-11** | `datasets/metrics.py` holds metric *definitions*; `engine/evaluator.py` holds the *loop*. No more duplicate `evaluator.py`. |
| **P2-12** | `neck.use_fpn` defaults **true**. Pure 1×1 projection does no cross-scale fusion and costs small-object AP; `use_fpn: false` measures it (exercised in `tests/_variants.py`). |
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

1. `tools/profile_model.py` — **done**, numbers above.
2. Lock token/projection dims against the measured budget, and run
   `tools/analyze_dataset.py` so the budget sweep starts from your data's real
   objects-per-image distribution rather than a quoted average.
3. Geometry writeback + EMA alignment maths — **done** (P1-7, P0-4).
4. Data adaptation, then **look at `tools/visualize_labels.py` output**.
5. Skeleton + unit tests + ONNX smoke — **done**.
6. **Token budget sweep** — before any main experiment.
7. Main → cross-illumination → full ablations → real-device latency.

## 6. Experiment matrix

### The five ablations the paper needs

Each changes exactly one variable relative to `models/model_main.yaml`; one per claim,
plus one design check. They are the only files in `configs/ablation/`.

| # | config | proves | why it is essential |
|---|---|---|---|
| 1 | `ablation/no_global_token` | the token path works at all | the premise of the paper. Also serves as the TinyNeXt-without-tokens baseline row |
| 2 | `ablation/no_geometric_writeback` | the geometric prior (primary contribution) | drops only the Gaussian term, keeps content attention -- isolates the claim exactly |
| 3 | `ablation/no_ema_routing` | illumination-consistent routing | direct evidence for the asymmetric-view EMA constraint |
| 4 | `ablation/token_budget_256` | 56 tokens are enough | the largest known risk (P0-3). If 256 is clearly better, the MAIN model changes |
| 5 | `ablation/no_routing_supervision` | routing supervision | without it the score maps collapse to flat (P0-4, defect 3). Must be trained: the first 200-epoch run (`3abbba5`) predates STAL (P0-5b), so it differs from main by more than one variable |

### Other runs

| group | config | question |
|---|---|---|
| Main vs baselines | `baselines/csp_n`, `baselines/csp_t` | accuracy at equal params (FLOPs reported), identical protocol |
| Second scale point | `models/model_edge_s` | does the design hold at ~1M backbone params? |
| Cross-illumination | #1 and #3 on `datasets/dronevehicle_rgb` | **does global context / photometric consistency matter more at night?** No new config -- same ablations, second dataset |
| Seeds | `tools/run_seeds.py` | mean ± std over 3 seeds |

### Code paths without a shipped config

Removing P2, removing FPN, mixer depth, routing locality, token source levels,
write-back into P2, random routing and same-view EMA all remain supported and are
exercised by `tests/_variants.py`. To run one as an experiment, write a
three-line YAML inheriting `../models/model_main.yaml` with the matching override
from that file.
