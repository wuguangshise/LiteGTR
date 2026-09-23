"""
LiteGTR 训练脚本 —— VisDrone（官方原始标注），改常量即可运行

    python train_litegtr.py

仓库里有两个训练入口，训练逻辑完全相同（都调用 engine/trainer.py）：

    train_litegtr.py   参数写在顶部常量区，改完直接运行。适合日常实验
    tools/train.py     参数全部来自 YAML，命令行传入。适合批量跑、写脚本调度

训练开始前会做完整预检，路径或标注格式不对会立刻报错并说明原因，不会跑到一半才崩。

数据集结构：
    DATA_ROOT\\
    ├── VisDrone2019-DET-train\\{images\\, annotations\\}
    └── VisDrone2019-DET-val\\{images\\, annotations\\}
"""
import sys
import time
from pathlib import Path

# 顶层立即输出：文件被执行就一定看得到。
# 若运行后完全没有任何输出，说明 Python 执行的不是这个文件
# （文件为空 / 被截断 / 运行配置指向了别处）。
try:                      # Windows 控制台默认 GBK，中文输出会炸
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

# Windows 的 spawn 会在每个 DataLoader worker 里重新 import 本文件，
# __name__ 在那里是 '__mp_main__'。只有主进程才打印，否则会刷屏。
if __name__ == "__main__":
    print("[train_litegtr] 脚本已加载", flush=True)

# ======================== 配置参数 ========================
# 【代码仓库】本脚本就在仓库根目录，留空 "" 即可自动定位
# 只有把脚本拷到别处运行时才需要填，且别填成下面那个同名的数据集目录
REPO_ROOT = ""

# 【数据集】VisDrone 根目录，其下直接是 VisDrone2019-DET-* 各个 split
DATA_ROOT = r"D:\dataset\VisDrone\LiteGTR"
TRAIN_SPLIT = "VisDrone2019-DET-train"
VAL_SPLIT = "VisDrone2019-DET-val"

# 忽略区域处理（VisDrone class 0）
#   'mask' 涂成 padding 灰   'drop' 仅丢弃标注框
# 跑基线对比时必须保持同一设置，否则不公平（docs/DESIGN.md P1-6 / P1-9）
IGNORE_MODE = "mask"

# 模型配置（相对 REPO_ROOT）
MODEL_CONFIG = r"configs/models/model_main.yaml"
#   主模型 / 轻量版：
#     configs/models/model_main.yaml        主模型
#     configs/models/model_edge_s.yaml      轻量版（第二个规模点）
#   对比基线（同 neck/head/loss/调度，只换 backbone）：
#     configs/baselines/csp_n.yaml          YOLOv8n 量级 CSP
#     configs/baselines/csp_t.yaml          更小的 CSP，对应 Edge-S
#   消融实验（configs/ablation/，共 4 个，按建议顺序）：
#     configs/ablation/no_global_token.yaml         ① token 路径整体有效（最先跑）
#     configs/ablation/token_budget_256.yaml        ② 56 个 token 是否足够
#     configs/ablation/no_geometric_writeback.yaml  ③ 几何先验（主创新）
#     configs/ablation/no_ema_routing.yaml          ④ 光照一致路由
#     说明见 configs/ablation/README.md
#   换实验时记得同时改下面的 NAME，否则会覆盖上一次的输出

# 训练参数
EPOCHS = 300
BATCH_SIZE = 16          # 显存不够就降：16 / 8 / 4
VAL_BATCH_SIZE = 16
IMG_SIZE = 640
DEVICE = "0"             # "0" / "cpu"
WORKERS = 8              # Windows 下若报多进程错误，改成 0

# 优化器 —— AdamW，ConvNeXt 从零训练配方
#   backbone 是 ConvNeXt 结构（原论文即用 AdamW），token 路径含注意力和 LayerNorm
#   注意：D-FINE/DEIM 的 lr 2.5e-4 / wd 1.25e-4 / 裁剪 0.1 是"预训练 backbone +
#   72 轮微调"的参数，从零训练 300 轮照搬会训不动，故不采用
OPTIMIZER = "adamw"      # 'adamw' / 'sgd'
LR0 = 0.001              # AdamW 0.001；若改 SGD 用 0.01
WEIGHT_DECAY = 0.05      # AdamW 0.05；若改 SGD 用 0.0005
GRAD_CLIP = 10.0         # 安全阀（YOLO 同值）；0.1 是 DETR 专用

# 学习率调度 —— FlatCosine（DEIM, CVPR 2025）
#   warmup -> 峰值平台期 -> 余弦衰减。DEIM 平台期约占总轮数一半
SCHEDULER = "flat_cosine"  # 'flat_cosine' / 'cosine'
WARMUP_EPOCHS = 3
FLAT_EPOCHS = 150          # 峰值学习率保持到第几轮（建议 EPOCHS 的一半）
FINAL_LR_RATIO = 0.01      # 衰减终点 = LR0 * 该值

# 权重 EMA —— D-FINE (ICLR 2025) 与 Ultralytics YOLO 均用 0.9999
EMA_DECAY = 0.9999

# 数据增强
MOSAIC_PROB = 0.5
NO_AUG_EPOCHS = 15       # 最后 N 轮关闭 mosaic（YOLO close_mosaic=10，DEIM no_aug_epoch=8）

# Token 预算（None = 用模型配置里的值）
# 例：{"P3": 64, "P4": 48, "P5": 16}，必须能被对应 grid^2 整除
TOKEN_BUDGET = None

# 其他
PROJECT = "runs/train"
NAME = "litegtr_visdrone"
SEED = 0
AMP = True
VAL_INTERVAL = 1         # 每 N 轮验证一次
SAVE_PERIOD = 0          # >0 时额外保存 epoch_xx.pt
RESUME = ""              # 续训：填 runs/train/<NAME>/weights/last.pt
                         # 其余参数（尤其 MODEL_CONFIG、EPOCHS）必须和原训练一致


# ======================== 仓库定位 ========================
def locate_repo() -> Path:
    marker = Path("models") / "detector.py"
    if REPO_ROOT:
        repo = Path(REPO_ROOT)
        if not (repo / marker).exists():
            raise FileNotFoundError(
                f"REPO_ROOT 不是 LiteGTR 代码仓库: {repo}\n"
                f"  该目录下应存在 models\\detector.py\n"
                f"  （注意别填成数据集目录 {DATA_ROOT}）\n"
                f"  克隆：git clone https://github.com/wuguangshise/LiteGTR.git")
        return repo

    here = Path(__file__).resolve().parent
    for cand in (here, here.parent, Path.cwd(), Path.cwd().parent):
        if (cand / marker).exists():
            print(f"自动定位仓库: {cand}")
            return cand
    raise FileNotFoundError(
        "未能自动定位 LiteGTR 代码仓库。\n"
        "  请在脚本顶部设置 REPO_ROOT，例如：\n"
        '      REPO_ROOT = r"D:\\code\\LiteGTR"\n'
        "  已尝试: 脚本目录、其父目录、当前工作目录及其父目录。\n"
        "  克隆：git clone https://github.com/wuguangshise/LiteGTR.git")


# ======================== 预检 ========================
def preflight(repo: Path) -> dict:
    cfg_path = repo / MODEL_CONFIG
    if not cfg_path.exists():
        raise FileNotFoundError(
            f"模型配置不存在: {cfg_path}\n"
            f"  可用配置见 {repo / 'configs'}")

    root = Path(DATA_ROOT)
    if not root.is_dir():
        raise FileNotFoundError(f"数据集根目录不存在: {root}")

    problems, counts = [], {}
    for tag, split in (("train", TRAIN_SPLIT), ("val", VAL_SPLIT)):
        split_dir = root / split
        img_dir, ann_dir = split_dir / "images", split_dir / "annotations"
        if not split_dir.is_dir():
            sibs = sorted(p.name for p in root.iterdir() if p.is_dir())
            problems.append(f"  缺少 {split_dir}\n    该目录下实际有: {sibs}")
            continue
        if not img_dir.is_dir():
            problems.append(f"  缺少 {img_dir}")
            continue
        n_img = sum(1 for p in img_dir.iterdir() if p.suffix.lower() in {".jpg", ".png"})
        counts[tag] = n_img
        if n_img == 0:
            problems.append(f"  {img_dir} 里没有图片")
        if not ann_dir.is_dir():
            problems.append(
                f"  缺少 {ann_dir}\n"
                f"    如果只有 labels\\（YOLO 格式），忽略区域已经丢失，\n"
                f"    请解压官方原始包。")
            continue
        n_ann = sum(1 for _ in ann_dir.glob("*.txt"))
        if n_ann == 0:
            problems.append(f"  {ann_dir} 里没有 .txt 标注")
        elif n_ann != n_img:
            print(f"  提示: {split} 图片 {n_img} / 标注 {n_ann}（数量不一致，缺标注的按空图处理）")

    if problems:
        raise FileNotFoundError("数据集结构不对:\n" + "\n".join(problems))

    # 抽一行标注，确认是官方八字段格式而非 YOLO 格式
    sample = next((root / TRAIN_SPLIT / "annotations").glob("*.txt"), None)
    if sample is not None:
        for line in sample.read_text(encoding="utf-8", errors="ignore").splitlines():
            line = line.strip()
            if not line:
                continue
            parts = [v for v in line.replace(" ", "").split(",") if v]
            if len(parts) < 6:
                raise ValueError(
                    f"标注不像 VisDrone 官方格式: {sample.name}\n"
                    f"  读到: {line!r}\n"
                    f"  期望: x,y,w,h,score,category,truncation,occlusion\n"
                    f"  若这是 YOLO 格式（cls cx cy w h），请改用官方原始标注。")
            break
    return counts


# ======================== 主流程 ========================
def main():
    print("=" * 64)
    print("LiteGTR 训练")
    print("=" * 64)
    repo = locate_repo()
    sys.path.insert(0, str(repo))
    counts = preflight(repo)

    import torch
    from torch.utils.data import DataLoader

    from datasets.base import collate_fn
    from datasets.visdrone import VisDroneDataset
    from engine.trainer import Trainer
    from models.build import active_token_count, build_model, count_deploy_params, load_config

    # 固定随机种子
    import random
    import numpy as np
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.benchmark = True

    # --- 配置：以仓库 YAML 为底，脚本常量覆盖 ---
    cfg = load_config(str(repo / MODEL_CONFIG))
    cfg["train"].update(
        epochs=EPOCHS, batch_size=BATCH_SIZE, val_batch_size=VAL_BATCH_SIZE,
        optimizer=OPTIMIZER, lr=LR0, weight_decay=WEIGHT_DECAY,
        scheduler=SCHEDULER, warmup_epochs=WARMUP_EPOCHS, flat_epochs=FLAT_EPOCHS,
        final_lr_ratio=FINAL_LR_RATIO, no_aug_epochs=NO_AUG_EPOCHS,
        grad_clip=GRAD_CLIP, amp=AMP, val_interval=VAL_INTERVAL,
        save_period=SAVE_PERIOD,
    )
    cfg["train"].setdefault("model_ema", {})["decay"] = EMA_DECAY
    if SCHEDULER == "flat_cosine" and FLAT_EPOCHS >= EPOCHS:
        raise ValueError(f"FLAT_EPOCHS={FLAT_EPOCHS} 必须小于 EPOCHS={EPOCHS}，否则永远不会衰减")
    if NO_AUG_EPOCHS >= EPOCHS:
        raise ValueError(f"NO_AUG_EPOCHS={NO_AUG_EPOCHS} 必须小于 EPOCHS={EPOCHS}")
    cfg.setdefault("loader", {})["num_workers"] = WORKERS
    if TOKEN_BUDGET is not None:
        tk = cfg["model"]["token"]
        missing = [lv for lv in tk["levels"] if lv not in TOKEN_BUDGET]
        if missing:
            raise ValueError(f"TOKEN_BUDGET 缺少生效层 {missing}；当前配置的 token 层是 "
                             f"{tk['levels']}，每一层都要给出数量")
        tk["budget"] = dict(TOKEN_BUDGET)
        for lv in tk["levels"]:
            n, g = tk["budget"][lv], tk["grids"][lv]
            if n % (g * g):
                raise ValueError(f"TOKEN_BUDGET[{lv}]={n} 不能被 grid^2={g * g} 整除")

    device = torch.device("cpu" if DEVICE == "cpu" else f"cuda:{DEVICE}")
    if device.type == "cuda" and not torch.cuda.is_available():
        print("  警告: 未检测到 CUDA，回退 CPU（会非常慢）")
        device = torch.device("cpu")
        cfg["train"]["amp"] = False

    # --- 数据 ---
    train_ds = VisDroneDataset(root=DATA_ROOT, split=TRAIN_SPLIT, img_size=IMG_SIZE,
                               train=True, mosaic_prob=MOSAIC_PROB, ignore_mode=IGNORE_MODE)
    val_ds = VisDroneDataset(root=DATA_ROOT, split=VAL_SPLIT, img_size=IMG_SIZE,
                             train=False, mosaic_prob=0.0, ignore_mode=IGNORE_MODE)
    cfg["model"]["num_classes"] = len(train_ds.classes)

    pin = device.type == "cuda"
    persist = WORKERS > 0
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=WORKERS, collate_fn=collate_fn, pin_memory=pin,
                              drop_last=True, persistent_workers=persist)
    val_loader = DataLoader(val_ds, batch_size=VAL_BATCH_SIZE, shuffle=False,
                            num_workers=WORKERS, collate_fn=collate_fn, pin_memory=pin,
                            persistent_workers=persist)

    # --- 模型 ---
    model = build_model(cfg)
    n_train = sum(p.numel() for p in model.parameters())
    n_deploy = count_deploy_params(model)
    tk = cfg["model"]["token"]
    n_tok = active_token_count(tk)   # 只计生效层；合并后的 budget 可能带有未启用层的键
    out_dir = repo / PROJECT / NAME

    print(f"代码仓库  : {repo}")
    print(f"模型配置  : {MODEL_CONFIG}")
    print(f"参数量    : 训练 {n_train / 1e6:.2f}M / 部署 {n_deploy / 1e6:.2f}M（EMA teacher 不计入）")
    print(f"特征层    : {model.levels}  strides={model.strides}")
    print(f"Token     : {n_tok} 个（{'启用' if n_tok else '禁用'}）  写回={tk.get('writeback_mode', '-')}")
    if n_tok:
        ema = tk.get("ema", {})
        gate = tk.get("score_gate")
        view = ema.get("view", "-") if ema.get("enabled") else "关闭"
        print(f"路由      : 打分门控={gate}  EMA视图={view}")
        if gate is None:
            print("  !! 配置里没有 score_gate —— 仓库代码是旧的，请先 git pull")
    print(f"数据集    : {DATA_ROOT}")
    print(f"            train {counts.get('train', 0)} 张 / val {counts.get('val', 0)} 张"
          f"  忽略区域={IGNORE_MODE}")
    print(f"类别      : {len(train_ds.classes)} 类 {train_ds.classes}")
    print(f"训练      : {EPOCHS} epoch  batch {BATCH_SIZE}  imgsz {IMG_SIZE}  workers {WORKERS}")
    print(f"优化器    : {OPTIMIZER}  lr={LR0}  wd={WEIGHT_DECAY}  裁剪={GRAD_CLIP}")
    sched = (f"flat_cosine  warmup {WARMUP_EPOCHS} -> 平台至 {FLAT_EPOCHS} -> 衰减至 {LR0 * FINAL_LR_RATIO:g}"
             if SCHEDULER == "flat_cosine" else f"cosine  warmup {WARMUP_EPOCHS} -> 衰减至 {LR0 * FINAL_LR_RATIO:g}")
    print(f"调度      : {sched}")
    print(f"增强      : mosaic {MOSAIC_PROB}，最后 {NO_AUG_EPOCHS} 轮关闭   权重EMA {EMA_DECAY}")
    print(f"设备      : {device}"
          + (f"  {torch.cuda.get_device_name(device.index or 0)}"
             f"  {torch.cuda.get_device_properties(device.index or 0).total_memory / 1024 ** 3:.1f} GB"
             if device.type == "cuda" else ""))
    print(f"AMP       : {cfg['train']['amp']}")
    print(f"输出      : {out_dir}")
    print("=" * 64)

    # --- 训练 ---
    trainer = Trainer(model, train_loader, val_loader, cfg, device, out_dir, train_ds.classes)
    trainer.recorder.save_json("args.yaml", {
        "script": str(Path(__file__).resolve()), "seed": SEED, "device": str(device),
        "data_root": DATA_ROOT, "ignore_mode": IGNORE_MODE, "resolved_config": cfg,
    })
    if RESUME:
        # 恢复原始权重、EMA、优化器、调度器、best 指标；配置不一致会直接报错
        start = trainer.resume(RESUME)
        print(f"续训: 从 {RESUME} 继续，第 {start} 轮开始")

    t0 = time.time()
    best = trainer.fit()
    print("\n" + "=" * 64)
    print(f"训练完成，用时 {(time.time() - t0) / 3600:.2f} 小时")
    print(f"最佳: {best}")
    print(f"权重: {out_dir / 'weights' / 'best.pt'}")
    print(f"曲线: {out_dir / 'results.png'}")
    print("=" * 64)


if __name__ == "__main__":
    # Windows 下 DataLoader 多进程必须有这层保护
    try:
        main()
    except KeyboardInterrupt:
        print("\n训练被用户中断")
    except Exception as e:
        print(f"\n出错: {type(e).__name__}: {e}")
        raise
