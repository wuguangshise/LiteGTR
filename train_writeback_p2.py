"""
单独训练「token 写回 P2」版本 —— 直接运行即可（PyCharm 里点运行）

    python train_writeback_p2.py

模型：configs/ablation/writeback_p2.yaml（P3-P5 上选 token，额外写回到 P2）。
训练配方（数据路径、轮数、batch、学习率、增强……）全部取自 train_litegtr.py
顶部的常量区，和 main 完全一致，这样两者的差别只来自模型。要改配方或数据路径，
改 train_litegtr.py，不要改这里。

输出：runs/train/<NAME>/，和 main 的输出目录互不影响。
"""
import subprocess
import sys
from pathlib import Path

# ======================== 只有这几项 ========================
MODEL_CONFIG = r"configs/ablation/writeback_p2.yaml"
NAME = "cand_writeback_p2"      # 输出目录名，别和 main 同名
SEED = 0                        # 和 main 保持一致
RESUME = ""                     # 续训：填 runs/train/cand_writeback_p2/weights/last.pt
# ============================================================

REPO = Path(__file__).resolve().parent


def main() -> int:
    cmd = [sys.executable, str(REPO / "train_litegtr.py"),
           "--model-config", MODEL_CONFIG, "--name", NAME, "--seed", str(SEED)]
    if RESUME:
        cmd += ["--resume", RESUME]
    print("[train_writeback_p2]", " ".join(cmd), flush=True)
    return subprocess.call(cmd, cwd=str(REPO))


if __name__ == "__main__":
    sys.exit(main())
