"""train_candidates.py trains the P2 candidates one after another."""
import subprocess
import sys

import pytest

torch = pytest.importorskip("torch")

import train_candidates as C  # noqa: E402
import train_litegtr as T  # noqa: E402
from models.build import load_config  # noqa: E402


def test_candidates_in_order():
    names = [c[0] for c in C.CANDIDATES]
    assert names == ["cand_detail_inject", "cand_writeback_p2_detail_inject"]
    assert T.NAME not in names


def test_candidate_configs_are_what_they_claim():
    by = {n: load_config(C.REPO / cfg)["model"] for n, cfg, _, _ in C.CANDIDATES}
    for name in by:
        di = by[name]["detail_inject"]
        assert di["enabled"] and di["mask"] == "score" and di["source"] == "P3"
        assert di["inject_at"] == "before_local"
        assert "detail_enhance" not in by[name]
    assert by["cand_detail_inject"]["token"]["writeback_levels"] == ["P3", "P4", "P5"]
    assert by["cand_writeback_p2_detail_inject"]["token"]["writeback_levels"] == ["P2", "P3", "P4", "P5"]


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


def test_candidates_run_one_after_another(tmp_path):
    """The second candidate starts only after the first has exited; a failure does
    not stop the next one."""
    stamp = tmp_path / "order.txt"
    todo = []
    for i, code in enumerate((3, 0)):
        cmd = [sys.executable, "-c",
               f"import time; open(r'{stamp}', 'a').write('start{i} '); time.sleep(0.3); "
               f"open(r'{stamp}', 'a').write('end{i} '); raise SystemExit({code})"]
        todo.append((f"c{i}", cmd, tmp_path / f"c{i}"))
    rc = C.run_in_order(todo)
    assert rc == {"c0": 3, "c1": 0}
    assert stamp.read_text().split() == ["start0", "end0", "start1", "end1"]
