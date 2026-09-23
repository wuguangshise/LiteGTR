"""run_experiments.py drives train_litegtr.py; the batch must stay resumable and in step."""
import pytest

torch = pytest.importorskip("torch")

import run_experiments as R  # noqa: E402
import train_litegtr as T  # noqa: E402


def test_every_experiment_config_exists_and_names_are_unique():
    names = [e[0] for e in R.EXPERIMENTS]
    assert len(set(names)) == len(names)
    for _, cfg, seed, _ in R.EXPERIMENTS:
        assert (R.REPO / cfg).exists(), cfg
        assert isinstance(seed, int)


def test_epochs_and_output_dir_come_from_the_training_script():
    tc = R.train_constants()
    assert tc["epochs"] == T.EPOCHS and tc["project"] == R.REPO / T.PROJECT


def test_state_is_read_from_last_pt(tmp_path):
    run = tmp_path / "exp"
    assert R.last_epoch(run) == 0 and R.best_metrics(run) == {}
    (run / "weights").mkdir(parents=True)
    torch.save({"epoch": 137, "best_metrics": {"mAP50_95": 0.17, "epoch": 135}}, run / "weights" / "last.pt")
    assert R.last_epoch(run) == 137
    assert R.best_metrics(run)["epoch"] == 135


def test_summary_marks_done_partial_and_pending(tmp_path, monkeypatch):
    monkeypatch.setattr(R, "EXPERIMENTS", [("a", "x.yaml", 0, ""), ("b", "y.yaml", 0, ""),
                                           ("c", "z.yaml", 0, "")])
    for name, ep in (("a", 200), ("b", 50)):
        (tmp_path / name / "weights").mkdir(parents=True)
        torch.save({"epoch": ep, "best_metrics": {"mAP50_95": 0.2, "AP_small": 0.15, "epoch": ep}},
                   tmp_path / name / "weights" / "last.pt")
    text = R.write_summary(tmp_path, 200, {}).read_text(encoding="utf-8-sig").splitlines()
    status = {row.split(",")[0]: row.split(",")[4] for row in text[1:]}
    assert status == {"a": "done", "b": "partial", "c": "pending"}
    assert "0.1500" in text[1]


def test_training_script_accepts_the_overrides(monkeypatch):
    for k in ("MODEL_CONFIG", "NAME", "RESUME", "SEED"):
        monkeypatch.setattr(T, k, getattr(T, k))          # restored after the test
    T.apply_cli_overrides(["--model-config", "configs/ablation/no_global_token.yaml",
                           "--name", "abl", "--resume", "runs/x/last.pt", "--seed", "2"])
    assert (T.MODEL_CONFIG, T.NAME, T.RESUME, T.SEED) == (
        "configs/ablation/no_global_token.yaml", "abl", "runs/x/last.pt", 2)


def test_no_arguments_leaves_the_constants_alone(monkeypatch):
    before = (T.MODEL_CONFIG, T.NAME, T.RESUME, T.SEED)
    T.apply_cli_overrides([])
    assert (T.MODEL_CONFIG, T.NAME, T.RESUME, T.SEED) == before


def _save_last(run, cfg_path, epoch=10):
    from models.build import load_config

    (run / "weights").mkdir(parents=True, exist_ok=True)
    cfg = load_config(R.REPO / cfg_path)
    cfg["model"]["num_classes"] = 10                     # written at runtime by the trainer
    torch.save({"epoch": epoch, "config": cfg}, run / "weights" / "last.pt")


def test_own_run_is_not_foreign(tmp_path):
    _save_last(tmp_path / "main", "configs/models/model_main.yaml")
    assert R.foreign_run(tmp_path / "main", "configs/models/model_main.yaml") == ""
    assert R.foreign_run(tmp_path / "empty", "configs/models/model_main.yaml") == ""


def test_run_with_another_config_is_refused(tmp_path):
    """The first 200-epoch run has the no_routing_supervision config: it must never be
    skipped as 'main is done' nor resumed as main."""
    _save_last(tmp_path / "main", "configs/ablation/no_routing_supervision.yaml", epoch=200)
    assert R.foreign_run(tmp_path / "main", "configs/models/model_main.yaml")


def test_results_without_checkpoint_is_refused(tmp_path):
    (tmp_path / "x").mkdir()
    (tmp_path / "x" / "results.csv").write_text("epoch\n1\n")
    assert R.foreign_run(tmp_path / "x", "configs/models/model_main.yaml")


def test_run_from_before_the_assigner_fix_is_refused(tmp_path):
    """Same model config, but trained before STAL (assigner.stal_size) existed."""
    from models.build import load_config

    run = tmp_path / "main"
    (run / "weights").mkdir(parents=True)
    cfg = load_config(R.REPO / "configs/models/model_main.yaml")
    cfg["model"]["num_classes"] = 10
    cfg["assigner"].pop("stal_size")
    torch.save({"epoch": 50, "config": cfg}, run / "weights" / "last.pt")
    assert "assigner" in R.foreign_run(run, "configs/models/model_main.yaml")
