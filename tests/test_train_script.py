"""train_litegtr.py must stay in step with the repo it drives.

It names config files in comments and constants. When configs were renamed or
removed, the script kept pointing at files that no longer existed -- this has
happened twice. These checks need only PyYAML, not torch.
"""
import re
from pathlib import Path

import train_litegtr as T
from models.build import load_config

ROOT = Path(__file__).resolve().parents[1]


def test_every_config_path_named_in_the_script_exists():
    src = Path(T.__file__).read_text(encoding="utf-8")
    paths = sorted(set(re.findall(r"configs/[\w/.-]+\.yaml", src)))
    assert paths, "expected the script to name at least one config"
    missing = [p for p in paths if not (ROOT / p).exists()]
    assert not missing, f"script references configs that do not exist: {missing}"


def test_model_config_loads():
    cfg = load_config(ROOT / T.MODEL_CONFIG)
    assert "model" in cfg and "train" in cfg


def test_schedule_constants_are_consistent():
    assert T.WARMUP_EPOCHS < T.FLAT_EPOCHS < T.EPOCHS
    assert 0 <= T.NO_AUG_EPOCHS < T.EPOCHS
    assert T.SCHEDULER in ("flat_cosine", "cosine")


def test_repo_is_located_without_configuration():
    assert T.REPO_ROOT == "", "the script sits in the repo root; auto-detection should find it"
    assert T.locate_repo().resolve() == ROOT
