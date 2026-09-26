"""Model variants exercised by the tests, built in code from the main config.

These are NOT the paper's ablations -- those live in configs/ablation/. Keeping
the tests independent of which ablations the paper happens to run means the
ablation list can be trimmed or extended without ever losing test coverage of a
code path (no P2, no FPN, mixer depth, routing locality, score gate, ...).
"""
from models.build import _deep_update, load_config

MAIN = "configs/models/model_main.yaml"

VARIANTS: dict[str, dict] = {
    "main": {},
    "no_global_token": {"model": {"token": {"enabled": False}}},
    "no_local_cnn": {"model": {"use_local_cnn": False}},
    "no_p2": {"model": {"use_p2": False}},
    "no_fpn": {"model": {"neck": {"use_fpn": False}}},
    "patchify_stem": {"model": {"backbone": {"stem": "patchify"}}},
    "content_writeback": {"model": {"token": {"writeback_mode": "content"}}},
    "broadcast_writeback": {"model": {"token": {"writeback_mode": "broadcast"}}},
    "writeback_p2": {"model": {"token": {"writeback_levels": ["P2", "P3", "P4", "P5"]}}},
    "detail_enhance": {"model": {"detail_enhance": {"enabled": True, "mask": "score"}}},
    "detail_enhance_global": {"model": {"detail_enhance": {"enabled": True, "mask": "global"}}},
    "detail_enhance_token": {"model": {"detail_enhance": {"enabled": True, "mask": "token"}}},
    "detail_inject": {"model": {"detail_inject": {"enabled": True, "mask": "score"}}},
    "detail_inject_global": {"model": {"detail_inject": {"enabled": True, "mask": "global"}}},
    "detail_inject_before_head": {"model": {"detail_inject": {"enabled": True, "inject_at": "before_head"}}},
    "detail_inject_writeback_p2": {"model": {"detail_inject": {"enabled": True},
                                             "token": {"writeback_levels": ["P2", "P3", "P4", "P5"]}}},
    "detail_inject_no_token": {"model": {"detail_inject": {"enabled": True, "mask": "global"},
                                         "token": {"enabled": False}}},
    "p2_weighted_fusion": {"model": {"neck": {"p2_fusion": "weighted"}}},
    "budget_128": {"model": {"token": {"budget": {"P3": 64, "P4": 48, "P5": 16}}}},
    "mixer_none": {"model": {"token": {"mixer_layers": 0}}},
    "mixer_deep": {"model": {"token": {"mixer_layers": 2}}},
    "global_topk": {"model": {"token": {"grids": {"P3": 1, "P4": 1, "P5": 1}}}},
    "token_src_p5": {"model": {"token": {"levels": ["P5"], "budget": {"P5": 56},
                                         "writeback_levels": ["P3", "P4", "P5"]}}},
    # routing: the scorer's gradient sources
    "random_routing": {"model": {"token": {"score_gate": False, "ema": {"enabled": False},
                                           "routing_sup": {"enabled": False}}}},
    "no_ema": {"model": {"token": {"ema": {"enabled": False}, "routing_sup": {"enabled": False}}}},
    "no_routing_sup": {"model": {"token": {"routing_sup": {"enabled": False}, "scorer_no_decay": False}}},
    "routing_sup_only": {"model": {"token": {"score_gate": False, "ema": {"enabled": False}}}},
    "ema_same_view": {"model": {"token": {"ema": {"view": "same"}}}},
}


def variant_cfg(name: str, num_classes: int = 10) -> dict:
    cfg = _deep_update(load_config(MAIN), VARIANTS[name])
    cfg["model"]["num_classes"] = num_classes
    return cfg
