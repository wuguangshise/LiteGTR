"""Config plumbing. Runs without torch."""
from models.build import load_config


def test_base_inheritance_merges_deeply():
    cfg = load_config("configs/models/model_main.yaml")
    assert cfg["train"]["epochs"] == 200
    assert cfg["model"]["backbone"]["channels"] == [32, 64, 128, 192]


def test_ablation_overrides_only_the_named_key():
    cfg = load_config("configs/ablation/token_budget_256.yaml")
    assert cfg["model"]["token"]["budget"]["P3"] == 128
    assert cfg["model"]["token"]["ema"]["enabled"] is True     # untouched by the override
    assert cfg["model"]["backbone"]["depths"] == [2, 4, 8, 2]


def test_token_budget_divisible_by_grid_squared():
    """LocalTopKSelector requires budget % grid^2 == 0 -- catch it in CI, not at epoch 1."""
    import glob
    for path in glob.glob("configs/**/*.yaml", recursive=True):
        cfg = load_config(path)
        tok = cfg.get("model", {}).get("token", {})
        if not tok.get("enabled", True) or "budget" not in tok:
            continue
        for lv, n in tok["budget"].items():
            g = tok["grids"][lv]
            assert n % (g * g) == 0, f"{path}: {lv} budget {n} not divisible by grid^2 {g*g}"
