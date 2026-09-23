"""EMA-guided token routing with an asymmetric photometric view (TRAINING ONLY).

What went wrong in the first version, and why it matters
--------------------------------------------------------
The teacher (an EMA of the student's scorers) was fed *exactly the same input*
as the student. The consistency loss then measured nothing but the drift between
the network and a ~1000-step-old copy of itself on identical data -- a quantity
that falls to zero on its own as training converges and the learning rate
decays. In practice the loss sat near 5e-6: it constrained nothing.

Mean Teacher, BYOL and DINO work because of **asymmetry**: teacher and student
see different views, and the loss says "this output should not depend on the
difference". Without that there is nothing to be consistent *about*.

The asymmetric design
---------------------
* The **student keeps its normal input** -- the standard augmentation pipeline --
  so the detection loss is untouched and the only new ingredient is the
  consistency term. That keeps the ablation clean.
* The **teacher sees a photometric perturbation** of the same batch: brightness,
  contrast, gamma and sensor noise. These change appearance but not geometry,
  so the two score maps align pixel-for-pixel and need no inverse warping.
* The perturbations model illumination change and low-light noise. The
  constraint therefore reads: *which tokens get selected should not depend on
  lighting* -- the property the DroneVehicle day/night evaluation measures.

The gradient question (unchanged)
---------------------------------
``top-k`` is discrete. Only the continuous score maps are aligned, via spatial
softmax KL in ``losses/token_consistency.py``; ``top-k`` never enters the graph.
Train and inference run the identical selection, so the exported graph is
unaffected, and the teacher is stripped from ``deploy_state_dict()``.

This is only meaningful if the scorer learns from the detection objective at
all -- see ``score_gate`` in ``token_selector.py``.
"""
from __future__ import annotations

import contextlib
import copy

import torch
import torch.nn as nn


class EMATokenRouter(nn.Module):
    def __init__(self, selector: nn.Module, momentum: float = 0.999, warmup_iters: int = 1000):
        super().__init__()
        self.momentum = momentum
        self.warmup_iters = warmup_iters
        self.register_buffer("num_updates", torch.zeros(1, dtype=torch.long))
        self.teacher = copy.deepcopy(selector)
        for p in self.teacher.parameters():
            p.requires_grad_(False)
        self.teacher.eval()

    @torch.no_grad()
    def update(self, selector: nn.Module) -> None:
        """Call once per optimiser step. Momentum ramps in over ``warmup_iters``."""
        self.num_updates += 1
        n = int(self.num_updates.item())
        m = min(self.momentum, (1 + n) / (10 + n)) if n < self.warmup_iters else self.momentum
        for tp, sp in zip(self.teacher.parameters(), selector.parameters()):
            tp.mul_(m).add_(sp.detach(), alpha=1 - m)
        for tb, sb in zip(self.teacher.buffers(), selector.buffers()):
            tb.copy_(sb)

    @torch.no_grad()
    def teacher_score_maps(self, feats: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        self.teacher.eval()
        return self.teacher(feats)["score_maps"]

    def train(self, mode: bool = True):
        super().train(mode)
        self.teacher.eval()   # teacher is never in train mode
        return self


@torch.no_grad()
def photometric_view(x: torch.Tensor, brightness: float = 0.4, contrast: float = 0.4,
                     gamma: tuple[float, float] = (0.7, 1.5), noise: float = 0.03) -> torch.Tensor:
    """Per-sample illumination / sensor perturbation of a [0,1] image batch.

    Geometry is untouched, so score maps stay pixel-aligned with the input.

    * brightness  multiply by U(1-b, 1+b)          -- exposure / time of day
    * contrast    scale around the mean by U(1-c, 1+c)
    * gamma       x ** U(lo, hi); >1 darkens mid-tones -- the low-light regime
    * noise       additive Gaussian, sigma ~ U(0, noise) -- sensor noise at low light
    """
    b = x.shape[0]
    dev, dt = x.device, x.dtype

    def u(lo: float, hi: float) -> torch.Tensor:
        return torch.empty(b, 1, 1, 1, device=dev, dtype=dt).uniform_(lo, hi)

    y = x * u(1 - brightness, 1 + brightness)
    mean = y.mean(dim=(1, 2, 3), keepdim=True)
    y = (y - mean) * u(1 - contrast, 1 + contrast) + mean
    y = y.clamp(0, 1).pow(u(gamma[0], gamma[1]))
    if noise > 0:
        y = y + torch.randn_like(y) * u(0.0, noise)
    return y.clamp(0, 1)


@contextlib.contextmanager
def preserved_bn_stats(*modules: nn.Module):
    """Run a forward in train mode without letting it touch BatchNorm running stats.

    The teacher branch re-runs the neck on the perturbed batch. In train mode that
    would fold the perturbed batch's statistics into the running mean/var used at
    validation. Batch statistics are still used for normalisation during the
    pass -- keeping it consistent with the student's -- but the buffers are
    restored afterwards.
    """
    saved = []
    for m in modules:
        if m is None:
            continue
        for bn in m.modules():
            if isinstance(bn, nn.modules.batchnorm._BatchNorm) and bn.track_running_stats:
                saved.append((bn, bn.running_mean.clone(), bn.running_var.clone(),
                              bn.num_batches_tracked.clone()))
    try:
        yield
    finally:
        for bn, rm, rv, nb in saved:
            bn.running_mean.copy_(rm)
            bn.running_var.copy_(rv)
            bn.num_batches_tracked.copy_(nb)
