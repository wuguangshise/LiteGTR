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
    assert names == ["cand_bb_r2", "cand_bb_r3"]
    assert T.NAME not in names


def test_candidates_differ_only_in_the_backbone():
    by = {n: load_config(C.REPO / cfg)["model"] for n, cfg, _, _ in C.CANDIDATES}
    r2, r3 = by["cand_bb_r2"], by["cand_bb_r3"]
    assert (r2["backbone"]["channels"], r2["backbone"]["depths"]) == ([40, 80, 128, 176], [2, 6, 6, 2])
    assert (r3["backbone"]["channels"], r3["backbone"]["depths"]) == ([48, 96, 128, 160], [3, 6, 6, 2])
    for m in (r2, r3):
        assert m["neck"]["p2_fusion"] == "weighted"
        di = m["detail_inject"]
        assert di["enabled"] and di["mask"] == "score" and di["inject_at"] == "before_local"
        assert m["token"]["writeback_levels"] == ["P2", "P3", "P4", "P5"]
        assert m["backbone"]["stem"] == "conv"
    strip = lambda m: {**m, "backbone": {k: v for k, v in m["backbone"].items() if k not in ("channels", "depths")}}
    assert strip(r2) == strip(r3)


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
