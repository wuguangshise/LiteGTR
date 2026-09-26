"""
对比方法统一训练 —— 用各方法作者的官方仓库，依次训练，直接运行即可（PyCharm 里点运行）

    python train_baselines.py              按 BASELINES 的顺序依次训练
    python train_baselines.py --dry-run    只看每个方法的状态和将要执行的命令

训练设置和 LiteGTR（main 分支 train_litegtr.py）完全一致：数据路径、train/val 划分、
轮数、batch、输入尺寸、随机种子、workers、GPU 见下面的"训练设置"，两边改的时候要一起改。
其余超参数（优化器、学习率、增强、损失、评估）一律用各仓库的官方设置；batch 和官方不同时，
按各仓库自己规定的方式缩放（见 baselines/configs.py）。

4 个方法共用一个 conda 环境（env/ 里的配置文件，见 README.md）：在这个环境里运行本脚本即可。

第一次运行时：
  1. 把 RemDet、DEIM 的官方仓库克隆到 baselines/third_party/，并固定到下面记录的提交；
  2. 把 VisDrone 转成各仓库的格式（baselines/prepare_visdrone.py）；
  3. 生成每个方法的配置，依次训练。

可以随时中断（Ctrl+C），再次运行：已完成的跳过，跑到一半的从各自的断点续训。
输出：runs/baselines/<方法名>/
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
# 训练设置 —— 和 main 分支 train_litegtr.py 的同名常量保持一致
DATA_ROOT = r"D:\dataset\VisDrone\LiteGTR"   # 里面是 VisDrone 官方的划分目录（images/ + annotations/）
TRAIN_SPLIT = "VisDrone2019-DET-train"
VAL_SPLIT = "VisDrone2019-DET-val"
EPOCHS = 200
BATCH_SIZE = 8
IMG_SIZE = 640
SEED = 0
WORKERS = 8              # Windows 下若报多进程错误，改成 0
DEVICE = "0"             # "0" / "cpu"

# (输出目录名, 框架, 模型) —— 按这个顺序依次训练；不想跑的在行首加 # 注释掉
BASELINES = [
    ("yolov8n",     "ultralytics", "yolov8n.yaml"),
    ("yolo11n",     "ultralytics", "yolo11n.yaml"),
    ("remdet_tiny", "remdet",      "remdet_tiny"),
    ("deim_n",      "deim",        "deim_n"),
]
ONLY = []                 # 只跑其中几个（填输出目录名）；[] = 全部
# ===========================================================

# 官方仓库，固定到这些提交，保证结果可复现
REPOS = {
    "remdet": ("https://github.com/HZAI-ZJNU/RemDet.git", "8cf2667e9ff80122e89436a0ebc6558ff1cfdb93"),
    "deim": ("https://github.com/ShihuaHuang95/DEIM.git", "09d35d53d39ee3145a1e61e3a989b28b9468d1dd"),
}
ULTRALYTICS_VERSION = "8.4.163"

REPO = Path(__file__).resolve().parent
THIRD_PARTY = REPO / "baselines" / "third_party"
OUT = REPO / "runs" / "baselines"


def recipe() -> dict:
    return {"data_root": DATA_ROOT, "train_split": TRAIN_SPLIT, "val_split": VAL_SPLIT,
            "epochs": EPOCHS, "batch": BATCH_SIZE, "imgsz": IMG_SIZE, "seed": SEED,
            "workers": WORKERS, "device": DEVICE}


def python_for(framework: str) -> str:
    """All four methods share one environment: the Python running this script."""
    return sys.executable


def ensure_repo(name: str, fetch: bool = True) -> Path:
    url, commit = REPOS[name]
    dst = THIRD_PARTY / name
    if not fetch:
        return dst
    if not dst.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        print(f"  克隆 {url} -> {dst}")
        subprocess.check_call(["git", "clone", url, str(dst)])
    head = subprocess.run(["git", "-C", str(dst), "rev-parse", "HEAD"],
                          capture_output=True, text=True).stdout.strip()
    if head != commit:
        subprocess.check_call(["git", "-C", str(dst), "checkout", "-q", commit])
    return dst


def done_marker(name: str) -> Path:
    return OUT / name / "DONE"


def check_env(framework: str) -> str:
    """One line on the framework's environment, so a missing install shows up before training."""
    probe = {"ultralytics": "import ultralytics, torch; print('ultralytics', ultralytics.__version__, "
                            "'torch', torch.__version__, 'cuda', torch.cuda.is_available())",
             "remdet": "import mmengine, mmcv, mmdet, torch, numpy; print('mmengine', mmengine.__version__, "
                       "'mmcv', mmcv.__version__, 'torch', torch.__version__, 'numpy', numpy.__version__, "
                       "'cuda', torch.cuda.is_available())",
             "deim": "import torch, torchvision, faster_coco_eval; print('torch', torch.__version__, "
                     "'torchvision', torchvision.__version__, 'cuda', torch.cuda.is_available())"}[framework]
    if framework == "remdet" and not (THIRD_PARTY / "remdet").exists():
        probe = probe.replace(" mmdet,", "")          # RemDet's mmdet comes with its repository
    r = subprocess.run([python_for(framework), "-c", probe], capture_output=True, text=True,
                       env=env_for(framework, "cpu"))
    if r.returncode:
        return "!! 环境不可用：" + (r.stderr.strip().splitlines() or ["?"])[-1]
    line = r.stdout.strip()
    if framework == "ultralytics" and f"ultralytics {ULTRALYTICS_VERSION} " not in line + " ":
        line += f"   !! 建议 ultralytics=={ULTRALYTICS_VERSION}"
    return line + version_warning(framework, line)


def _version(line: str, package: str) -> tuple:
    words = line.split()
    v = words[words.index(package) + 1] if package in words else "0"
    return tuple(int(x) for x in v.split("+")[0].split(".")[:2] if x.isdigit())


def version_warning(framework: str, line: str) -> str:
    """Versions that install fine but break training (see README)."""
    if framework == "deim" and _version(line, "torchvision") >= (0, 21):
        return "   !! torchvision 必须 <= 0.20（DEIM 的数据增强在 0.21+ 会报 NotImplementedError）"
    if framework == "remdet" and _version(line, "numpy") >= (2,):
        return '   !! 需要 pip install "numpy<2"（torch 2.2 不兼容 numpy 2）'
    return ""


def command(name: str, framework: str, model: str, rc: dict, paths: dict,
            write: bool = True) -> list[str]:
    """The training command. ``write=False`` (dry run) clones nothing and writes no config."""
    from baselines import configs

    py, out = python_for(framework), OUT / name
    if framework == "ultralytics":
        return [py, str(REPO / "baselines" / "run_ultralytics.py"), "--model", model,
                "--data", str(paths["ultralytics_yaml"]), "--epochs", str(rc["epochs"]),
                "--batch", str(rc["batch"]), "--imgsz", str(rc["imgsz"]), "--seed", str(rc["seed"]),
                "--workers", str(rc["workers"]), "--device", str(rc["device"]),
                "--project", str(OUT), "--name", name]
    if framework == "remdet":
        repo = ensure_repo("remdet", fetch=write)
        cfg = OUT / "configs" / f"{name}.py"
        if write:
            configs.remdet_config(repo, cfg, out, paths, rc["epochs"], rc["batch"],
                                  rc["workers"], rc["seed"])
        cmd = [py, str(repo / "tools" / "train.py"), str(cfg), "--work-dir", str(out),
               "--auto-scale-lr"] + (["--amp"] if str(rc["device"]) != "cpu" else [])
        if (out / "last_checkpoint").exists():
            cmd.append("--resume")
        return cmd
    if framework == "deim":
        repo = ensure_repo("deim", fetch=write)
        cfg = OUT / "configs" / f"{name}.yml"
        if write:
            configs.deim_config(repo, cfg, out, paths, rc["epochs"], rc["batch"], rc["workers"])
        cpu = str(rc["device"]) == "cpu"
        cmd = [py, str(REPO / "baselines" / "run_deim.py"), str(repo), "-c", str(cfg),
               "--seed", str(rc["seed"]),
               "-d", "cpu" if cpu else "cuda:0"] + ([] if cpu else ["--use-amp"])
        if (out / "last.pth").exists():
            cmd += ["-r", str(out / "last.pth")]
        return cmd
    raise ValueError(f"未知框架: {framework}")


def env_for(framework: str, device) -> dict:
    """RemDet and DEIM always train on the first visible GPU, so the GPU chosen by DEVICE is
    selected by CUDA_VISIBLE_DEVICES. RemDet ships its own modified mmdet (pure Python):
    putting the repository on PYTHONPATH is what its `pip install -e .` would do."""
    import os

    env = dict(os.environ)
    if framework in REPOS:
        env["CUDA_VISIBLE_DEVICES"] = "" if str(device) == "cpu" else str(device)
    if framework == "remdet":
        env["PYTHONPATH"] = os.pathsep.join(
            [str(THIRD_PARTY / "remdet")] + [x for x in [env.get("PYTHONPATH")] if x])
    return env


def cwd_for(framework: str) -> Path:
    """RemDet and DEIM import their packages from the repository root."""
    return THIRD_PARTY / framework if framework in REPOS else REPO


def main() -> int:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="只列出状态和命令，不训练")
    a = ap.parse_args()

    names = [b[0] for b in BASELINES]
    unknown = set(ONLY) - set(names)
    if unknown:
        raise SystemExit(f"ONLY 里有不认识的名字 {sorted(unknown)}；可选：{names}")
    todo = [b for b in BASELINES if not ONLY or b[0] in ONLY]
    rc = recipe()

    print("=" * 72)
    print(f"对比方法：{len(todo)} 个   数据 {rc['data_root']}   {rc['epochs']} 轮  batch {rc['batch']}  "
          f"{rc['imgsz']}px  seed {rc['seed']}")
    print("=" * 72)
    from baselines.prepare_visdrone import expected_paths, prepare

    args = (rc["data_root"], rc["train_split"], rc["val_split"], OUT / "data")
    paths = expected_paths(*args) if a.dry_run else prepare(*args)
    print(f"环境 {sys.executable}")
    for fw in dict.fromkeys(b[1] for b in todo):
        print(f"  {fw:12s} {check_env(fw)}")
    print("-" * 72)

    results = {}
    for i, (name, fw, model) in enumerate(todo, 1):
        head = f"[{i}/{len(todo)}] {name:12s}"
        if done_marker(name).exists():
            print(f"{head} 已完成，跳过")
            continue
        cmd = command(name, fw, model, rc, paths, write=not a.dry_run)
        print(f"{head} {fw}\n{'':17s}{' '.join(cmd)}")
        if a.dry_run:
            continue
        t0 = time.time()
        try:
            code = subprocess.call(cmd, cwd=str(cwd_for(fw)), env=env_for(fw, rc["device"]))
        except KeyboardInterrupt:
            code = 130
        results[name] = code
        print(f"== {name} 结束，返回码 {code}，用时 {(time.time() - t0) / 3600:.2f} h", flush=True)
        if code == 0:
            done_marker(name).parent.mkdir(parents=True, exist_ok=True)
            done_marker(name).write_text(time.strftime("%Y-%m-%d %H:%M:%S"), encoding="utf-8")
        if code == 130:
            print("已中断：再次运行本脚本会从断点续训")
            break
    if results:
        print("=" * 72)
        for name, code in results.items():
            print(f"{name:12s} {'完成' if code == 0 else f'未完成（返回码 {code}）'}")
    return 0 if all(c == 0 for c in results.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
