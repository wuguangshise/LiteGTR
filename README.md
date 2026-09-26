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

## 环境（Windows + conda）

三个框架的依赖互相冲突（RemDet 需要 torch 2.2 + mmcv 2.2.0），所以建三个 conda 环境。
步骤来自各仓库官方 README，加上实际跑通时发现必须固定的版本（下面注明了原因）。
下面的命令在 Anaconda Prompt 里、本仓库根目录下执行。

**先克隆两个官方仓库**（第一次运行 `train_baselines.py` 也会自动克隆；装环境要先有代码）：

```bash
git clone https://github.com/HZAI-ZJNU/RemDet.git baselines/third_party/remdet
git -C baselines/third_party/remdet checkout 8cf2667e9ff80122e89436a0ebc6558ff1cfdb93
git clone https://github.com/ShihuaHuang95/DEIM.git baselines/third_party/deim
git -C baselines/third_party/deim checkout 09d35d53d39ee3145a1e61e3a989b28b9468d1dd
```

**ultralytics**（可以直接用训练 LiteGTR 的环境，只需要再装 ultralytics）：

```bash
pip install ultralytics==8.4.163
```

**RemDet**：

```bash
conda create -n remdet -y python=3.11
conda activate remdet
pip install torch==2.2.0 torchvision==0.17.0 --index-url https://download.pytorch.org/whl/cu121
cd baselines/third_party/remdet
pip install -r requirements.txt
pip install albumentations==1.4.4 timm
pip install -U openmim
mim install mmengine
mim install mmcv==2.2.0
pip install -v -e . --no-build-isolation
pip install "numpy<2" "opencv-python<4.11"
cd ../../..
```

- `--no-build-isolation`：RemDet 的 setup.py 要 import torch，新版 pip 默认在隔离环境里构建，会找不到 torch。
- 最后一行必须有：torch 2.2 是用 numpy 1.x 编译的，而上面的步骤会顺带装上 numpy 2，
  训练时会报 `A module that was compiled using NumPy 1.x cannot be run in NumPy 2.x`。
- `mim install mmcv==2.2.0` 会先找 OpenMMLab 的预编译包。如果它开始从源码编译
  （输出里出现 `Building wheel for mmcv`，要十几分钟以上，而且需要 Visual Studio Build Tools 的 C++ 工具），
  可以等它编完；装不上的话，RemDet 自带的 mmdet 也接受 mmcv 2.1.0，
  换成 torch 2.1 + mmcv 2.1.0（Windows 预编译包更全）：

  ```bash
  pip install torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cu121
  mim install mmcv==2.1.0
  ```

**DEIM**：

```bash
conda create -n deim -y python=3.11.9
conda activate deim
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu124
pip install -r baselines/third_party/deim/requirements.txt
```

- torchvision 必须是 0.20 或更早：DEIM 的数据增强实现的是 torchvision 旧的 `_transform` 接口，
  0.21 起改名成 `transform`，新版本会在第一个 batch 报 `NotImplementedError`。
  DEIM 的 requirements.txt 只写了 `torchvision>=0.15.2`，所以要先装好固定版本再装 requirements。
- 显卡驱动较旧、不支持 CUDA 12.4 的话，把 `cu124` 换成 `cu121`（torch 2.5.1 两个都有）。

装好后，把三个环境的 python 路径填进 `train_baselines.py` 顶部的 `PYTHON`：

```python
PYTHON = {
    "ultralytics": r"C:\Users\<你>\anaconda3\envs\litegtr\python.exe",
    "remdet":      r"C:\Users\<你>\anaconda3\envs\remdet\python.exe",
    "deim":        r"C:\Users\<你>\anaconda3\envs\deim\python.exe",
}
```

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
- DEIM 官方用 `torchrun` 启动。这里用 `run_deim.py` 在单进程里直接运行官方 `train.py`，不建进程组，
  DEIM 走它自带的单进程模式（不包 DDP、BN 不转 SyncBN）。单卡时这和 `torchrun --nproc_per_node=1`
  在数值上等价，而且不依赖 NCCL（Windows 版 PyTorch 没有 NCCL）。DEIM 仓库里的文件一个都不改。
