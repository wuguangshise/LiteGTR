"""train_writeback_p2.py launches train_litegtr.py with the P2-writeback config."""
import pytest

torch = pytest.importorskip("torch")

import train_litegtr as T  # noqa: E402
import train_writeback_p2 as W  # noqa: E402
from models.build import load_config  # noqa: E402


def test_config_exists_and_writes_back_into_p2():
    cfg = load_config(W.REPO / W.MODEL_CONFIG)
    assert cfg["model"]["token"]["writeback_levels"] == ["P2", "P3", "P4", "P5"]
    assert cfg["model"]["token"]["levels"] == ["P3", "P4", "P5"]      # selection unchanged


def test_output_dir_differs_from_main():
    assert W.NAME != T.NAME and W.NAME != "main"


def test_training_script_accepts_its_arguments(monkeypatch):
    for k in ("MODEL_CONFIG", "NAME", "RESUME", "SEED"):
        monkeypatch.setattr(T, k, getattr(T, k))          # restored after the test
    T.apply_cli_overrides(["--model-config", W.MODEL_CONFIG, "--name", W.NAME, "--seed", str(W.SEED)])
    assert (T.MODEL_CONFIG, T.NAME, T.SEED) == (W.MODEL_CONFIG, W.NAME, W.SEED)
