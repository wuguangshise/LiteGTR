"""YOLO-style run recording: results.csv, curves, model summary, token stats."""
from __future__ import annotations

import csv
import json
import logging
from pathlib import Path


class Recorder:
    def __init__(self, out_dir: str | Path):
        self.dir = Path(out_dir)
        (self.dir / "weights").mkdir(parents=True, exist_ok=True)
        self.results_csv = self.dir / "results.csv"
        self.condition_csv = self.dir / "results_by_condition.csv"
        self.token_csv = self.dir / "token_stats.csv"
        self._headers: dict[Path, list[str]] = {}
        self.logger = self._make_logger()

    def _make_logger(self) -> logging.Logger:
        lg = logging.getLogger(f"litegtr.{self.dir.name}")
        lg.setLevel(logging.INFO)
        lg.handlers.clear()
        fh = logging.FileHandler(self.dir / "training.log", encoding="utf-8")
        sh = logging.StreamHandler()
        fmt = logging.Formatter("%(asctime)s %(message)s", "%Y-%m-%d %H:%M:%S")
        fh.setFormatter(fmt); sh.setFormatter(fmt)
        lg.addHandler(fh); lg.addHandler(sh)
        lg.propagate = False
        return lg

    def _append(self, path: Path, row: dict) -> None:
        if path not in self._headers:
            self._headers[path] = list(row.keys())
        write_header = not path.exists() or path.stat().st_size == 0
        with open(path, "a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=self._headers[path])
            if write_header:
                w.writeheader()
            w.writerow({k: row.get(k, "") for k in self._headers[path]})

    def log_epoch(self, row: dict) -> None:
        self._append(self.results_csv, row)

    def log_conditions(self, epoch: int, per_condition: dict) -> None:
        for cond, metrics in per_condition.items():
            self._append(self.condition_csv, {"epoch": epoch, "condition": cond, **metrics})

    def log_tokens(self, row: dict) -> None:
        self._append(self.token_csv, row)

    def save_text(self, name: str, text: str) -> None:
        (self.dir / name).write_text(text, encoding="utf-8")

    def save_json(self, name: str, obj) -> None:
        (self.dir / name).write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")

    def plot_curves(self) -> None:
        """results.png from results.csv. Best-effort: never let plotting kill a run."""
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            import pandas as pd
            if not self.results_csv.exists():
                return
            df = pd.read_csv(self.results_csv)
            cols = [c for c in df.columns if c not in ("epoch", "lr", "time")]
            if not cols:
                return
            n = len(cols)
            ncol = min(4, n)
            nrow = (n + ncol - 1) // ncol
            fig, axes = plt.subplots(nrow, ncol, figsize=(4 * ncol, 3 * nrow), squeeze=False)
            for ax, c in zip(axes.ravel(), cols):
                ax.plot(df["epoch"], df[c], marker=".")
                ax.set_title(c, fontsize=9); ax.grid(alpha=0.3)
            for ax in axes.ravel()[n:]:
                ax.axis("off")
            fig.tight_layout()
            fig.savefig(self.dir / "results.png", dpi=120)
            plt.close(fig)
        except Exception as e:  # pragma: no cover
            self.logger.warning(f"plot_curves failed: {e}")
