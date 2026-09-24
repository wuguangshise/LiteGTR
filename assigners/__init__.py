"""Label assigners, selected by ``assigner.mode`` in the config."""
from __future__ import annotations


def build_assigner(cfg: dict | None):
    """``assigner:`` config block -> assigner module.

    ``mode: rfla`` -- RFLA (ECCV'22), the main model's assigner.
    ``mode: tal``  -- task-aligned assigner; with ``stal_size > 0`` it is TAL + STAL.
    A block without ``mode`` is a TAL config from before RFLA existed.
    """
    cfg = dict(cfg or {})
    mode = cfg.pop("mode", "tal")
    if mode == "rfla":
        from assigners.rfla_assigner import RFLAAssigner

        return RFLAAssigner(**cfg.get("rfla", {}))
    if mode == "tal":
        from assigners.task_aligned_assigner import TaskAlignedAssigner

        cfg.pop("rfla", None)
        return TaskAlignedAssigner(**cfg)
    raise ValueError(f"unknown assigner.mode: {mode!r} (expected 'rfla' or 'tal')")
