"""
一次训练两个候选模型 —— 直接运行即可（PyCharm 里点运行）

    python train_candidates.py             两个同时训练（PARALLEL = True）
    python train_candidates.py --dry-run   只看每个候选的状态和将要执行的命令

默认同时训练：
    cand_writeback_p2      configs/ablation/writeback_p2.yaml     token 全局信息写回 P2
    cand_detail_enhance    configs/ablation/detail_enhance.yaml   打分图引导的 P2 细节增强

训练配方（数据路径、轮数、batch、学习率、增强……）全部取自 train_litegtr.py 顶部的
常量区，和 main 完全一致，候选之间的差别只来自模型。要改配方或数据路径，改
train_litegtr.py，不要改这里。

和 run_experiments.py 一样可以随时中断（Ctrl+C），再次运行会自动接着来：
  * 已经跑完的候选直接跳过
  * 跑到一半的候选从 last.pt 续训（原来 train_writeback_p2.py 训到一半的
    cand_writeback_p2 也会自动续训）
  * runs/train/<NAME>/ 里是别的配置训练出来的：报 conflict，不动它

同时训练时两个进程共用一张卡：
  * 显存大约是单个训练的两倍。显存不够（CUDA out of memory）就把 PARALLEL 改成
    False，改为一个接一个地训练
  * 每个进程都有 train_litegtr.py 里 WORKERS 个数据加载进程，mosaic 很吃 CPU；
    CPU 占满、time/data 明显变大时，把 WORKERS 调小
  * 每个候选的控制台输出写到 runs/train/<NAME>/console.log，训练日志照常在
    training.log。这个窗口里只定时打印两个候选各跑到了第几轮
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
# (输出目录名 NAME, 模型配置, 随机种子, 说明)；NAME 别和 main 同名
CANDIDATES = [
    ("cand_writeback_p2",   "configs/ablation/writeback_p2.yaml",   0, "token 全局信息写回 P2"),
    ("cand_detail_enhance", "configs/ablation/detail_enhance.yaml", 0, "打分图引导的 P2 细节增强"),
]
PARALLEL = True           # True：同时训练；False：一个接一个
POLL_SECONDS = 300        # 同时训练时，每隔多少秒打印一次进度
# ===========================================================

REPO = Path(__file__).resolve().parent
TRAIN = REPO / "train_litegtr.py"


def plan(dry_run: bool = False) -> list[tuple[str, list[str], Path]]:
    """每个候选要执行的命令；已完成或有冲突的不在其中。"""
    sys.path.insert(0, str(REPO))
    from run_experiments import foreign_run, last_epoch, train_constants

    names = [c[0] for c in CANDIDATES]
    if len(set(names)) != len(names):
        raise SystemExit("CANDIDATES 里有重复的 NAME，会互相覆盖输出")
    tc = train_constants()
    epochs, project = tc["epochs"], tc["project"]
    todo = []
    for name, cfg, seed, note in CANDIDATES:
        if not (REPO / cfg).exists():
            raise SystemExit(f"{name}: 配置文件不存在 {cfg}")
        run = project / name
        why = foreign_run(run, cfg, tc["token_budget"])
        if why:
            print(f"{name:22s} !! 已存在同名目录，没有动它：{why}\n"
                  f"{'':22s}    把 {run} 改名或移走后再运行")
            continue
        done = last_epoch(run)
        if done >= epochs:
            print(f"{name:22s} 已完成（{done} 轮），跳过   {note}")
            continue
        cmd = [sys.executable, str(TRAIN), "--model-config", cfg, "--name", name, "--seed", str(seed)]
        if done:
            cmd += ["--resume", str(run / "weights" / "last.pt")]
        print(f"{name:22s} {'从第 %d 轮续训' % done if done else '从头训练'}   {note}")
        print(f"{'':22s} {' '.join(cmd)}")
        todo.append((name, cmd, run))
    return todo


def run_sequential(todo) -> dict[str, int]:
    rc = {}
    for name, cmd, _ in todo:
        print(f"\n== 开始 {name}", flush=True)
        rc[name] = subprocess.call(cmd, cwd=str(REPO))
        if rc[name] == 130:
            break
    return rc


def run_parallel(todo) -> dict[str, int]:
    from run_experiments import last_epoch

    procs = {}
    for name, cmd, run in todo:
        run.mkdir(parents=True, exist_ok=True)
        log = open(run / "console.log", "a", encoding="utf-8")
        procs[name] = (subprocess.Popen(cmd, cwd=str(REPO), stdout=log, stderr=subprocess.STDOUT), log, run)
        print(f"已启动 {name}（pid {procs[name][0].pid}），输出见 {run / 'console.log'}", flush=True)
    rc: dict[str, int] = {}
    try:
        while len(rc) < len(procs):
            time.sleep(POLL_SECONDS)
            for name, (p, log, run) in procs.items():
                if name not in rc and p.poll() is not None:
                    rc[name] = p.returncode
                    log.close()
            print(time.strftime("[%H:%M] ") + "   ".join(
                f"{n}: 第 {last_epoch(r)} 轮" + ("" if n not in rc else f"（已结束，返回码 {rc[n]}）")
                for n, (_, _, r) in procs.items()), flush=True)
    except KeyboardInterrupt:
        print("\n中断：正在停止所有训练进程……再次运行本脚本会从 last.pt 续训", flush=True)
        for name, (p, log, _) in procs.items():
            if p.poll() is None:
                p.terminate()
        for name, (p, log, _) in procs.items():
            p.wait()
            log.close()
            rc.setdefault(name, 130)
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
    rc = run_parallel(todo) if PARALLEL and len(todo) > 1 else run_sequential(todo)

    from run_experiments import best_metrics
    runs = {name: run for name, _, run in todo}
    print("=" * 72)
    for name, code in rc.items():
        m = best_metrics(runs[name]) if code == 0 else {}
        best = "  ".join(f"{k}={m[k]:.4f}" for k in ("mAP50_95", "AP_small") if isinstance(m.get(k), float))
        print(f"{name:22s} {'完成' if code == 0 else f'未完成（返回码 {code}）'}   {best}")
    print("=" * 72)
    return 0 if all(c == 0 for c in rc.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
