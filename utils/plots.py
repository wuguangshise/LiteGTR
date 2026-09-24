"""Run artefacts promised in docs/DESIGN.md: confusion matrix, PR curves,
qualitative prediction dumps. All best-effort -- plotting never kills a run."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from utils.boxes import iou_matrix


def _agg():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


class ConfusionMatrix:
    """(K+1) x (K+1) matrix; the extra row/column is background.

    On dense UAV scenes the background column is the informative part: it counts
    missed objects, which is where small-object methods actually differ.
    """

    def __init__(self, num_classes: int, conf: float = 0.25, iou_thr: float = 0.45):
        self.k = num_classes
        self.conf = conf
        self.iou_thr = iou_thr
        self.matrix = np.zeros((num_classes + 1, num_classes + 1), dtype=np.int64)

    def update(self, dt_boxes, dt_scores, dt_labels, gt_boxes, gt_labels) -> None:
        dt_boxes = np.asarray(dt_boxes).reshape(-1, 4)
        dt_scores = np.asarray(dt_scores).reshape(-1)
        dt_labels = np.asarray(dt_labels).reshape(-1).astype(int)
        gt_boxes = np.asarray(gt_boxes).reshape(-1, 4)
        gt_labels = np.asarray(gt_labels).reshape(-1).astype(int)

        keep = dt_scores >= self.conf
        dt_boxes, dt_labels = dt_boxes[keep], dt_labels[keep]

        if not len(gt_boxes):
            for c in dt_labels:
                self.matrix[c, self.k] += 1           # predicted c, was background
            return
        if not len(dt_boxes):
            for c in gt_labels:
                self.matrix[self.k, c] += 1           # missed
            return

        iou = iou_matrix(dt_boxes, gt_boxes)
        matched_gt: set[int] = set()
        matched_dt: set[int] = set()
        order = np.argsort(-iou.max(axis=1))
        for di in order:
            gi = int(iou[di].argmax())
            if iou[di, gi] >= self.iou_thr and gi not in matched_gt:
                self.matrix[dt_labels[di], gt_labels[gi]] += 1
                matched_gt.add(gi)
                matched_dt.add(int(di))
        for di in range(len(dt_boxes)):
            if di not in matched_dt:
                self.matrix[dt_labels[di], self.k] += 1
        for gi in range(len(gt_boxes)):
            if gi not in matched_gt:
                self.matrix[self.k, gt_labels[gi]] += 1

    def plot(self, path: str | Path, classes: list[str], normalize: bool = True) -> None:
        try:
            plt = _agg()
            names = list(classes) + ["background"]
            m = self.matrix.astype(np.float64)
            if normalize:
                m = m / np.clip(m.sum(axis=0, keepdims=True), 1, None)
            fig, ax = plt.subplots(figsize=(1 + 0.6 * len(names), 1 + 0.6 * len(names)))
            im = ax.imshow(m, cmap="Blues", vmin=0, vmax=m.max() or 1)
            ax.set_xticks(range(len(names)), names, rotation=45, ha="right", fontsize=7)
            ax.set_yticks(range(len(names)), names, fontsize=7)
            ax.set_xlabel("ground truth"); ax.set_ylabel("prediction")
            if len(names) <= 14:
                for i in range(len(names)):
                    for j in range(len(names)):
                        ax.text(j, i, f"{m[i, j]:.2f}" if normalize else int(m[i, j]),
                                ha="center", va="center", fontsize=6,
                                color="white" if m[i, j] > m.max() * 0.6 else "black")
            fig.colorbar(im, fraction=0.046)
            fig.tight_layout(); fig.savefig(path, dpi=140); plt.close(fig)
        except Exception as e:  # pragma: no cover
            print(f"[plots] confusion matrix failed: {e}")


def plot_pr_curves(per_class: dict[str, tuple[np.ndarray, np.ndarray]], path: str | Path,
                   title: str = "Precision-Recall @ IoU 0.5") -> None:
    """``per_class``: {class_name: (recall, precision)}."""
    try:
        plt = _agg()
        fig, ax = plt.subplots(figsize=(7, 5))
        for name, (rec, prec) in sorted(per_class.items()):
            ax.plot(rec, prec, linewidth=1.2, label=name)
        ax.set_xlabel("recall"); ax.set_ylabel("precision")
        ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.grid(alpha=0.3)
        ax.set_title(title)
        ax.legend(fontsize=7, loc="lower left", ncol=2)
        fig.tight_layout(); fig.savefig(path, dpi=140); plt.close(fig)
    except Exception as e:  # pragma: no cover
        print(f"[plots] PR curve failed: {e}")


def draw_predictions(img_bgr: np.ndarray, dt_boxes, dt_scores, dt_labels,
                     gt_boxes, classes: list[str], conf: float = 0.25,
                     label: str = "class_score") -> np.ndarray:
    """GT in grey, predictions in colour -- for ``val_predictions/``.

    ``label``: ``none`` (colour only), ``class`` or ``class_score``."""
    assert label in ("none", "class", "class_score"), label
    import cv2
    out = img_bgr.copy()
    for b in np.asarray(gt_boxes).reshape(-1, 4).astype(int):
        cv2.rectangle(out, (b[0], b[1]), (b[2], b[3]), (150, 150, 150), 1)
    palette = [(66, 135, 245), (66, 245, 135), (245, 135, 66), (245, 66, 135),
               (135, 66, 245), (245, 200, 66), (66, 245, 245), (200, 66, 245),
               (120, 200, 120), (200, 120, 120)]
    dt_boxes = np.asarray(dt_boxes).reshape(-1, 4)
    dt_scores = np.asarray(dt_scores).reshape(-1)
    dt_labels = np.asarray(dt_labels).reshape(-1).astype(int)
    for b, s, l in zip(dt_boxes, dt_scores, dt_labels):
        if s < conf:
            continue
        c = palette[int(l) % len(palette)]
        p1, p2 = (int(b[0]), int(b[1])), (int(b[2]), int(b[3]))
        cv2.rectangle(out, p1, p2, c, 1)
        if label != "none":
            text = classes[int(l)] if label == "class" else f"{classes[int(l)]} {s:.2f}"
            cv2.putText(out, text, (p1[0], max(p1[1] - 2, 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, c, 1, cv2.LINE_AA)
    return out
