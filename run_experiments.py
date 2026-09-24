"""
LiteGTR 批量实验 —— 按顺序把论文需要的实验一次跑完

    python run_experiments.py              按下面 EXPERIMENTS 的顺序全部跑
    python run_experiments.py --dry-run    只列出每个实验的状态和将要执行的命令
    python run_experiments.py --only main abl_no_global_token   只跑指定的几个

每个实验都是单独调用一次 train_litegtr.py，只替换 MODEL_CONFIG / NAME / SEED，
其余训练常量（轮数、学习率、batch、数据路径……）全部取自 train_litegtr.py。
所以要改训练配方或数据路径，改 train_litegtr.py 顶部的常量，不要改这里。

可以随时中断（Ctrl+C），再次运行会自动接着来：
  * 已经跑完的实验（last.pt 已到最后一轮）直接跳过
  * 跑到一半的实验从 last.pt 续训
  * 还没开始的实验从头训练
某个实验出错时默认记下来、继续跑下一个（STOP_ON_ERROR 可改）。

每跑完一个实验，都会把所有实验的最佳指标汇总到 runs/train/experiments_summary.csv。

输出位置：每个实验写到 runs/train/<NAME>/（NAME 是下面列表的第一列），互不覆盖。
如果 runs/train/<NAME>/ 已经存在、但里面是别的配置训练出来的，本脚本不会跳过、
不会续训、也不会覆盖它，而是报 conflict 并跳到下一个实验。
"""
import csv
import subprocess
import sys
import time
from pathlib import Path

try:                      # Windows 控制台默认 GBK
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

# ======================== 实验列表（按顺序执行） ========================
# (输出目录名 NAME, 模型配置, 随机种子, 说明)
# 不想跑的实验在行首加 # 注释掉即可；顺序就是执行顺序
EXPERIMENTS = [
    # --- 完整模型：所有消融都和它比，先确认路由监督有效（score_entropy 明显 < 1）
    ("main",                        "configs/models/model_main.yaml",                   0, "完整模型"),
    # --- 消融（相对完整模型只改一个变量）
    ("abl_no_global_token",         "configs/ablation/no_global_token.yaml",            0, "① 去掉整条 token 路径"),
    ("abl_token_budget_256",        "configs/ablation/token_budget_256.yaml",           0, "② token 56 -> 256"),
    ("abl_no_geometric_writeback",  "configs/ablation/no_geometric_writeback.yaml",     0, "③ 去掉几何先验"),
    ("abl_no_ema_routing",          "configs/ablation/no_ema_routing.yaml",             0, "④ 去掉 EMA 光照一致路由"),
    ("abl_no_routing_supervision",  "configs/ablation/no_routing_supervision.yaml",     0, "⑤ 去掉路由监督"),
    ("abl_assigner_stal",           "configs/ablation/assigner_stal.yaml",              0, "⑥ 标签分配 RFLA -> TAL+STAL"),
    # --- 基线与第二个规模点
    ("base_csp_n",                  "configs/baselines/csp_n.yaml",                     0, "CSP 基线（对应 Main）"),
    ("edge_s",                      "configs/models/model_edge_s.yaml",                 0, "Edge-S 轻量版"),
    ("base_csp_t",                  "configs/baselines/csp_t.yaml",                     0, "CSP 基线（对应 Edge-S）"),
    # --- 多 seed（主表的均值 ± 标准差）：需要时取消注释
    # ("main_seed1",                "configs/models/model_main.yaml",                   1, "完整模型 seed 1"),
    # ("main_seed2",                "configs/models/model_main.yaml",                   2, "完整模型 seed 2"),
]

STOP_ON_ERROR = False     # True：某个实验出错就停下整批；False：记下来，接着跑下一个

SUMMARY_KEYS = ["mAP50_95", "mAP50", "mAP75", "AP_small", "AP_medium", "AP_large", "AP_vt", "AP_t"]


# ======================== 实现 ========================
REPO = Path(__file__).resolve().parent
TRAIN = REPO / "train_litegtr.py"


def train_constants() -> dict:
    """train_litegtr.py 的常量：轮数和输出目录以它为准，两边永远一致。"""
    sys.path.insert(0, str(REPO))
    import train_litegtr as t
    return {"epochs": t.EPOCHS, "project": REPO / t.PROJECT, "token_budget": t.TOKEN_BUDGET}


def _model_cfg(model: dict, token_budget=None) -> dict:
    """比较用的模型配置：去掉运行时才写入的 num_classes，套上 TOKEN_BUDGET 覆盖。"""
    import copy
    m = copy.deepcopy(model)
    m.pop("num_classes", None)
    if token_budget is not None and "token" in m:
        m["token"]["budget"] = dict(token_budget)
    return m


def foreign_run(run_dir: Path, cfg_path: str, token_budget=None) -> str:
    """同名目录里是不是别的实验。是的话返回原因，否则返回空字符串。

    last.pt 每一轮都会保存，所以只有 results.csv、没有 last.pt 的目录不可能是
    本实验跑到一半留下的；last.pt 里存的模型配置和本实验不同，说明是别的训练。
    这两种情况都不能跳过、不能续训、也不能在里面从头训练（results.csv 会被追加）。
    """
    last = run_dir / "weights" / "last.pt"
    if not last.exists():
        if (run_dir / "results.csv").exists():
            return "目录里有 results.csv 但没有 last.pt，不是本实验留下的"
        return ""
    import torch
    from models.build import load_config

    saved = torch.load(last, map_location="cpu", weights_only=False).get("config") or {}
    if "model" not in saved:
        return ""
    want = load_config(REPO / cfg_path)
    if _model_cfg(saved["model"]) != _model_cfg(want["model"], token_budget):
        return f"last.pt 的模型配置和 {cfg_path} 不一致，是别的实验"
    # 分配器和损失也算实验定义：修复前（没有 STAL）的训练不能被当成同一个实验续训
    for key in ("assigner", "loss"):
        if saved.get(key, {}) != want.get(key, {}):
            return f"last.pt 的 {key} 配置和当前代码不一致（多半是修复前的旧训练）"
    return ""


def last_epoch(run_dir: Path) -> int:
    """last.pt 里记录的轮数；没有 last.pt 返回 0。"""
    last = run_dir / "weights" / "last.pt"
    if not last.exists():
        return 0
    import torch
    return int(torch.load(last, map_location="cpu", weights_only=False).get("epoch", 0))


def best_metrics(run_dir: Path) -> dict:
    last = run_dir / "weights" / "last.pt"
    if not last.exists():
        return {}
    import torch
    ck = torch.load(last, map_location="cpu", weights_only=False)
    return dict(ck.get("best_metrics") or {})


def write_summary(project: Path, epochs: int, outcome: dict) -> Path:
    path = project / "experiments_summary.csv"
    project.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8-sig") as f:      # utf-8-sig：Excel 能直接打开
        w = csv.writer(f)
        w.writerow(["name", "config", "seed", "note", "status", "last_epoch", "best_epoch"] + SUMMARY_KEYS)
        for name, cfg, seed, note in EXPERIMENTS:
            run = project / name
            done = last_epoch(run)
            m = best_metrics(run)
            status = outcome.get(name) or ("done" if done >= epochs else ("partial" if done else "pending"))
            w.writerow([name, cfg, seed, note, status, done, m.get("epoch", "")]
                       + [f"{m[k]:.4f}" if isinstance(m.get(k), float) else "" for k in SUMMARY_KEYS])
    return path


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="只列出状态和命令，不训练")
    ap.add_argument("--only", nargs="+", help="只跑这些 NAME")
    a = ap.parse_args()

    names = [e[0] for e in EXPERIMENTS]
    if len(set(names)) != len(names):
        raise SystemExit("EXPERIMENTS 里有重复的 NAME，会互相覆盖输出")
    for name, cfg, _, _ in EXPERIMENTS:
        if not (REPO / cfg).exists():
            raise SystemExit(f"{name}: 配置文件不存在 {cfg}")
    if a.only:
        unknown = set(a.only) - set(names)
        if unknown:
            raise SystemExit(f"--only 里有不认识的 NAME: {sorted(unknown)}；可选: {names}")

    tc = train_constants()
    epochs, project = tc["epochs"], tc["project"]
    todo = [e for e in EXPERIMENTS if not a.only or e[0] in a.only]

    print("=" * 72)
    print(f"LiteGTR 批量实验：{len(todo)} 个，每个 {epochs} 轮，输出到 {project}")
    print("=" * 72)
    outcome: dict[str, str] = {}
    t_all = time.time()
    for i, (name, cfg, seed, note) in enumerate(todo, 1):
        run = project / name
        why = foreign_run(run, cfg, tc["token_budget"])
        if why:
            outcome[name] = "conflict"
            print(f"[{i}/{len(todo)}] {name:28s} !! 已存在同名目录，没有动它：{why}\n"
                  f"        把 {run} 改名或移走后再运行，本实验会从头训练")
            continue
        done = last_epoch(run)
        cmd = [sys.executable, str(TRAIN), "--model-config", cfg, "--name", name, "--seed", str(seed)]
        if done >= epochs:
            print(f"[{i}/{len(todo)}] {name:28s} 已完成（{done} 轮），跳过   {note}")
            continue
        if done:
            cmd += ["--resume", str(run / "weights" / "last.pt")]
            state = f"从第 {done} 轮续训"
        else:
            state = "从头训练"
        print(f"[{i}/{len(todo)}] {name:28s} {state}   {note}")
        print("        " + " ".join(cmd))
        if a.dry_run:
            continue

        t0 = time.time()
        try:
            rc = subprocess.call(cmd, cwd=str(REPO))
        except KeyboardInterrupt:
            rc = 130
        hours = (time.time() - t0) / 3600
        if rc == 130:
            print(f"\n已中断：{name} 停在第 {last_epoch(run)} 轮。再次运行 run_experiments.py 会从这里续训")
            outcome[name] = "interrupted"
            write_summary(project, epochs, outcome)
            sys.exit(130)
        if rc != 0 or last_epoch(run) < epochs:
            outcome[name] = f"failed (exit {rc})"
            print(f"\n!! {name} 出错（返回码 {rc}，用时 {hours:.2f} h），日志见 {run / 'training.log'}")
            write_summary(project, epochs, outcome)
            if STOP_ON_ERROR:
                sys.exit(rc or 1)
            continue
        outcome[name] = "done"
        m = best_metrics(run)
        print(f"\n== {name} 完成，用时 {hours:.2f} h   best: "
              + "  ".join(f"{k}={m[k]:.4f}" for k in SUMMARY_KEYS if isinstance(m.get(k), float)))
        path = write_summary(project, epochs, outcome)
        print(f"   汇总已更新: {path}\n")

    if not a.dry_run:
        path = write_summary(project, epochs, outcome)
        failed = [n for n, s in outcome.items() if s.startswith("failed") or s == "conflict"]
        print("=" * 72)
        print(f"全部结束，总用时 {(time.time() - t_all) / 3600:.2f} h   汇总: {path}")
        if failed:
            print(f"出错的实验: {failed}  —— 修好后再运行本脚本，只会重跑没完成的")
        print("=" * 72)


if __name__ == "__main__":
    main()
