# LiteGTR 对比实验：用官方仓库统一训练

这个分支只用来跑对比实验：`train_baselines.py` 按顺序训练 4 个对比方法。每个方法都用作者的官方代码，
训练设置和 LiteGTR（main 分支）一致：

| 方法 | 官方代码 | 固定版本 |
|---|---|---|
| YOLOv8n | ultralytics（pip 安装） | `ultralytics==8.4.163` |
| YOLO11n | ultralytics（pip 安装） | `ultralytics==8.4.163` |
| RemDet-Tiny | https://github.com/HZAI-ZJNU/RemDet | `8cf2667e` |
| DEIM-N（D-FINE-N + DEIM） | https://github.com/ShihuaHuang95/DEIM | `09d35d53` |

RemDet 和 DEIM 的仓库会在第一次运行时克隆到 `baselines/third_party/`，并固定到上表的提交。
这个目录已经加进 `.gitignore`，不会提交到本仓库；它们的许可证（GPL/AGPL/Apache）也就不会影响本仓库。

## 和 LiteGTR 一致的设置

下面这些值写在 `train_baselines.py` 顶部，和 main 分支 `train_litegtr.py` 的同名常量一一对应。
**改了 main 的训练设置，这里要一起改**：

| 项 | 常量 | 当前值 |
|---|---|---|
| 数据 | `DATA_ROOT` / `TRAIN_SPLIT` / `VAL_SPLIT` | VisDrone2019-DET train / val |
| 轮数 | `EPOCHS` | 200 |
| batch | `BATCH_SIZE` | 8 |
| 输入尺寸 | `IMG_SIZE` | 640 |
| 随机种子 | `SEED` | 0 |
| 数据加载进程 | `WORKERS` | 8 |
| GPU | `DEVICE` | "0" |

其余所有超参数（优化器、学习率、数据增强、损失、NMS、评估）都用各仓库的官方设置，
一个都不改。batch 和官方不同时，按各仓库自己规定的方式换算：

- **YOLOv8n / YOLO11n**：ultralytics 内部自带换算（nominal batch 64 的梯度累积和 weight decay 缩放），
  直接传 `batch=8`。模型从 yaml 从头训练（`pretrained=False`），和 LiteGTR 一样不用 COCO 预训练权重。
- **RemDet-Tiny**：官方是 8 卡 × 32 = 256。用 mmdet 自带的 `--auto-scale-lr`（线性缩放，
  `base_batch_size=256`）。最后 10 轮关闭 mosaic，和官方一致。
- **DEIM-N**：官方总 batch 32、160 轮。按 DEIM README "Customizing Batch Size" 的规则，
  k = 8/32：学习率 × k，EMA decay = 1 − (1 − 0.9999)·k，EMA 预热和学习率预热步数 ÷ k。
  轮数改成 200 时，数据增强的分段按官方比例换算：官方 160 = 148 + 12，分段为 [4, 78, 148]；
  200 轮换算为 [4, 98, 188]，最后 12 轮不做增强。骨干用官方的 ImageNet 预训练 HGNetv2-B0，
  第一次运行时自动下载到 `runs/baselines/pretrained/`。

生成的配置文件在 `runs/baselines/configs/`，打开就能看到所有改动。

## 评估

每个方法用**自己仓库的评估代码**：ultralytics 用自己的 mAP；RemDet 用它的 CocoMetric；
DEIM 用 faster-coco-eval。GT 都来自同一份标注，由 `prepare_visdrone.py` 转换：

- 类别 1–10 转成 0–9；类别 0（ignored regions）和 11（others）去掉；宽或高 ≤ 0 的框去掉。
  这和 LiteGTR 的数据读取规则相同。
- COCO json（RemDet、DEIM 用）保留原始框；YOLO txt（ultralytics 用）会把框裁剪到图像内。
- 转换结果：YOLO 标签写到数据集各划分目录下的 `labels/`（和 `images/` 并列，ultralytics 规定的位置），
  COCO json 写到 `runs/baselines/data/`。原始 `annotations/` 不会被改动。

## 环境（Windows + conda，4 个方法共用一个环境）

配置文件在 `env/`：

| 文件 | 作用 |
|---|---|
| `env/setup_env.bat` | 一键建环境：建 conda 环境 → 装 torch → 装 mmcv → 装其余依赖 → 检查 |
| `env/requirements.txt` | 全部依赖，版本都固定成实际跑通 4 个方法时的版本 |

在 **Anaconda Prompt** 里、本仓库根目录下运行：

```bat
env\setup_env.bat
```

它依次执行（出错会停下并提示）：

```bat
conda create -n litegtr-baselines -y python=3.10
conda activate litegtr-baselines
pip install torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cu121
pip install mmcv==2.1.0 --only-binary mmcv -f https://download.openmmlab.com/mmcv/dist/cu121/torch2.1/index.html
pip install -r env\requirements.txt
```

为什么是这些版本（三个仓库要装进同一个环境，每一条都是必须的）：

| 版本 | 原因 |
|---|---|
| torch 2.1.2 + cu121 | mmcv 带 CUDA 算子，必须和 torch 版本配套；torch 2.1 的 Windows 预编译 mmcv 最全 |
| mmcv 2.1.0 | RemDet 自带的 mmdet 要求 `2.0.0rc4 <= mmcv < 2.2.1`；`--only-binary` 表示只装预编译包，不在本机编译 |
| torchvision 0.16.2 | DEIM 的数据增强用的是 torchvision 旧接口，0.21 起会报 `NotImplementedError` |
| numpy 1.26.4 | torch 2.1 是用 numpy 1.x 编译的，numpy 2 会报错 |
| transformers 4.46.3 | DEIM 需要它；RemDet 的 mmdet 发现环境里有 transformers 就会导入，5.x 不支持 torch 2.1，导入时直接崩溃 |
| Python 3.10 | 三个仓库都支持，Windows 预编译包覆盖最全 |

RemDet 不需要 `pip install -e .`：它的 mmdet 是纯 Python，`train_baselines.py` 训练 RemDet 时
把它的仓库目录加进 `PYTHONPATH`，效果相同。RemDet 和 DEIM 的官方仓库在第一次运行时自动克隆。

显卡驱动要支持 CUDA 12.1（`nvidia-smi` 右上角 CUDA Version ≥ 12.1）。

装好以后先检查：

```bat
conda activate litegtr-baselines
python train_baselines.py --dry-run
```

每个方法一行版本信息；出现 `!!` 表示环境有问题。

**如果 mmcv 那一步失败**（`No matching distribution found for mmcv==2.1.0`，说明 OpenMMLab 没有
和你的 Python / CUDA 对应的预编译包）：打开
https://download.openmmlab.com/mmcv/dist/cu121/torch2.1/index.html ，搜 `win_amd64`，看有哪些 `cp3xx`，
按它把 `setup_env.bat` 里的 `python=3.10` 改成对应版本重来。还不行就用 WSL2（Linux 下这套环境已完整跑通）。

## 运行

```bash
python train_baselines.py --dry-run    # 先看：每个环境是否可用、将要执行的命令；不写任何文件
python train_baselines.py              # 依次训练（PyCharm 里直接点运行）
```

- 顺序由 `BASELINES` 决定；不想跑的在行首加 `#`，或者在 `ONLY` 里只填想跑的名字。
- 可以随时 Ctrl+C。再次运行时：已完成的（有 `DONE` 标记）跳过，没完成的从各自的断点续训
  （ultralytics `last.pt`，RemDet `last_checkpoint`，DEIM `last.pth`）。
- 某个方法失败不会影响后面的方法，最后会汇总每个方法的返回码。
- 输出：`runs/baselines/<方法名>/`。

## 说明

- RemDet 和 DEIM 的训练脚本总是使用第一块可见 GPU，所以 `DEVICE` 是通过 `CUDA_VISIBLE_DEVICES` 传过去的。
- ultralytics 检测到环境里装了 albumentations（RemDet 需要它）时，会启用自己内置的几种轻微增强
  （模糊、灰度、CLAHE 等，每种概率 1%）。这是 ultralytics 的默认行为，不需要处理。
- DEIM 官方用 `torchrun` 启动。这里用 `run_deim.py` 在单进程里直接运行官方 `train.py`，不建进程组，
  DEIM 走它自带的单进程模式（不包 DDP、BN 不转 SyncBN）。单卡时这和 `torchrun --nproc_per_node=1`
  在数值上等价，而且不依赖 NCCL（Windows 版 PyTorch 没有 NCCL）。DEIM 仓库里的文件一个都不改。
