"""
依次训练两个 P2 候选模型 —— 直接运行即可（PyCharm 里点运行）

    python train_candidates.py             按顺序训练：跑完一个再开始下一个
    python train_candidates.py --dry-run   只看每个候选的状态和将要执行的命令

默认按顺序训练：
    1. cand_detail_inject                configs/ablation/detail_inject.yaml
       路由细节注入：stem stride-2 的细节在打分图标出目标的地方注入 P2
    2. cand_writeback_p2_detail_inject   configs/ablation/writeback_p2_detail_inject.yaml
       路由细节注入 + token 全局信息写回 P2

对照用已经训练好的 main 和 cand_writeback_p2，四个一起构成 P2 上的 2x2（上下文 x 细节）。

一次只训练一个，显存和 CPU 占用和单独训练完全一样。

训练配方（数据路径、轮数、batch、学习率、增强……）全部取自 train_litegtr.py 顶部的
常量区，和 main 完全一致，候选之间的差别只来自模型。要改配方或数据路径，改
train_litegtr.py，不要改这里。

和 run_experiments.py 一样可以随时中断（Ctrl+C），再次运行会自动接着来：
  * 已经跑完的候选直接跳过
  * 跑到一半的候选从 last.pt 续训
  * runs/train/<NAME>/ 里是别的配置训练出来的：报 conflict，不动它，接着训练下一个
  * 某个候选出错：记下来，接着训练下一个
"""
import subprocess
import sys
import time
from pathlib import Path

try:                      # Windows 控制台默认 GBK
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

# ======================== 只改这里 ========================
# (输出目录名 NAME, 模型配置, 随机种子, 说明)；按这个顺序依次训练，NAME 别和 main 同名
CANDIDATES = [
    ("cand_detail_inject",              "configs/ablation/detail_inject.yaml",              0,
     "路由细节注入 P2"),
    ("cand_writeback_p2_detail_inject", "configs/ablation/writeback_p2_detail_inject.yaml", 0,
     "路由细节注入 P2 + 写回 P2"),
]
# ===========================================================

REPO = Path(__file__).resolve().parent
TRAIN = REPO / "train_litegtr.py"


def plan() -> list[tuple[str, list[str], Path]]:
    """每个候选要执行的命令，按 CANDIDATES 的顺序；已完成或有冲突的不在其中。"""
    sys.path.insert(0, str(REPO))
    from run_experiments import foreign_run, last_epoch, train_constants

    names = [c[0] for c in CANDIDATES]
    if len(set(names)) != len(names):
        raise SystemExit("CANDIDATES 里有重复的 NAME，会互相覆盖输出")
    tc = train_constants()
    epochs, project = tc["epochs"], tc["project"]
    todo = []
    for i, (name, cfg, seed, note) in enumerate(CANDIDATES, 1):
        if not (REPO / cfg).exists():
            raise SystemExit(f"{name}: 配置文件不存在 {cfg}")
        run = project / name
        head = f"[{i}/{len(CANDIDATES)}] {name:22s}"
        why = foreign_run(run, cfg, tc["token_budget"])
        if why:
            print(f"{head} !! 已存在同名目录，没有动它：{why}\n"
                  f"{'':29s}把 {run} 改名或移走后再运行")
            continue
        done = last_epoch(run)
        if done >= epochs:
            print(f"{head} 已完成（{done} 轮），跳过   {note}")
            continue
        cmd = [sys.executable, str(TRAIN), "--model-config", cfg, "--name", name, "--seed", str(seed)]
        if done:
            cmd += ["--resume", str(run / "weights" / "last.pt")]
        print(f"{head} {'从第 %d 轮续训' % done if done else '从头训练'}   {note}")
        print(f"{'':29s}{' '.join(cmd)}")
        todo.append((name, cmd, run))
    return todo


def run_in_order(todo) -> dict[str, int]:
    """一个接一个；Ctrl+C 停在当前候选，后面的不再开始。"""
    rc: dict[str, int] = {}
    for name, cmd, _ in todo:
        print(f"\n== 开始 {name}", flush=True)
        t0 = time.time()
        try:
            rc[name] = subprocess.call(cmd, cwd=str(REPO))
        except KeyboardInterrupt:
            rc[name] = 130
        print(f"== {name} 结束，返回码 {rc[name]}，用时 {(time.time() - t0) / 3600:.2f} h", flush=True)
        if rc[name] == 130:
            print("已中断：再次运行本脚本会从 last.pt 续训", flush=True)
            break
    return rc


def main() -> int:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="只列出状态和命令，不训练")
    a = ap.parse_args()

    print("=" * 72)
    todo = plan()
    print("=" * 72)
    if a.dry_run or not todo:
        return 0
    rc = run_in_order(todo)

    from run_experiments import best_metrics
    runs = {name: run for name, _, run in todo}
    print("=" * 72)
    for name, code in rc.items():
        m = best_metrics(runs[name]) if code == 0 else {}
        best = "  ".join(f"{k}={m[k]:.4f}" for k in ("mAP50_95", "AP_small") if isinstance(m.get(k), float))
        print(f"{name:22s} {'完成' if code == 0 else f'未完成（返回码 {code}）'}   {best}")
    print("=" * 72)
    return 0 if len(rc) == len(todo) and all(c == 0 for c in rc.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
