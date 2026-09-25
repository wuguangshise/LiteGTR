"""train_candidates.py trains the P2 candidates, alone or two at a time."""
import subprocess
import sys

import pytest

torch = pytest.importorskip("torch")

import train_candidates as C  # noqa: E402
import train_litegtr as T  # noqa: E402
from models.build import load_config  # noqa: E402


def test_two_candidates_with_distinct_output_dirs():
    names = [c[0] for c in C.CANDIDATES]
    assert len(names) == 2 and len(set(names)) == 2
    assert not {"main", T.NAME} & set(names)


def test_candidate_configs_are_what_they_claim():
    by = {n: load_config(C.REPO / cfg)["model"] for n, cfg, _, _ in C.CANDIDATES}
    assert by["cand_writeback_p2"]["token"]["writeback_levels"] == ["P2", "P3", "P4", "P5"]
    assert "detail_enhance" not in by["cand_writeback_p2"]
    de = by["cand_detail_enhance"]["detail_enhance"]
    assert de["enabled"] and de["mask"] == "score" and de["levels"] == ["P2"]
    assert by["cand_detail_enhance"]["token"]["writeback_levels"] == ["P3", "P4", "P5"]


def test_training_script_accepts_each_candidate(monkeypatch):
    for k in ("MODEL_CONFIG", "NAME", "RESUME", "SEED"):
        monkeypatch.setattr(T, k, getattr(T, k))          # restored after the test
    for name, cfg, seed, _ in C.CANDIDATES:
        T.apply_cli_overrides(["--model-config", cfg, "--name", name, "--seed", str(seed)])
        assert (T.MODEL_CONFIG, T.NAME, T.SEED) == (cfg, name, seed)


def test_dry_run_lists_both_and_trains_nothing(tmp_path, monkeypatch):
    out = subprocess.run([sys.executable, str(C.REPO / "train_candidates.py"), "--dry-run"],
                         cwd=str(C.REPO), capture_output=True, text=True, encoding="utf-8", timeout=120)
    assert out.returncode == 0, out.stderr
    for name, _, _, _ in C.CANDIDATES:
        assert name in out.stdout


def test_parallel_launches_every_candidate_and_collects_exit_codes(tmp_path, monkeypatch):
    """Two children run at once; each writes its own console.log; exit codes come back."""
    monkeypatch.setattr(C, "POLL_SECONDS", 0.2)
    todo = []
    for i, code in enumerate((0, 3)):
        run = tmp_path / f"c{i}"
        cmd = [sys.executable, "-c", f"import time; print('hi {i}'); time.sleep(0.5); raise SystemExit({code})"]
        todo.append((f"c{i}", cmd, run))
    rc = C.run_parallel(todo)
    assert rc == {"c0": 0, "c1": 3}
    assert "hi 0" in (tmp_path / "c0" / "console.log").read_text(encoding="utf-8")
    assert "hi 1" in (tmp_path / "c1" / "console.log").read_text(encoding="utf-8")
