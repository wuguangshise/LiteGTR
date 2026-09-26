# 消融实验

只保留论文真正需要的 5 个：四个主张各配一个消融来证明，再加一个设计验证。
每个都继承 `configs/models/model_main.yaml`，**相对主模型只改一个变量**。

| # | 配置 | 改了什么 | 证明什么 |
|---|---|---|---|
| 1 | `no_global_token.yaml` | 关掉整条 token 路径 | **token 路径整体有效** —— 整篇论文的前提 |
| 2 | `no_geometric_writeback.yaml` | 写回去掉高斯几何先验，只留内容注意力 | **几何先验（主创新）有效** |
| 3 | `no_ema_routing.yaml` | 关掉 EMA 光照一致性约束 | **光照一致路由有效** |
| 4 | `token_budget_256.yaml` | token 从 56 增加到 256 | **56 个是否足够** —— 最大的已知风险 |
| 5 | `no_routing_supervision.yaml` | 去掉打分器的 GT 前景监督，打分器输出层恢复 weight decay | **路由监督有效** —— 没有它分数图会塌缩成平的，等于随机路由 |

## 论文里怎么呈现

```
                         mAP    与主模型差值   证明
主模型                    xx.x        —
  − token 路径 (#1)       xx.x      −a      token 路径的贡献
  − 几何先验   (#2)       xx.x      −b      主创新的贡献
  − 光照一致   (#3)       xx.x      −c      路由约束的贡献
  token 256    (#4)       xx.x      ±d      预算是否足够
  − 路由监督   (#5)       xx.x      −e      学习式选择的贡献
```

`#1` 还兼作"TinyNeXt 无 token"基线，和 `configs/baselines/csp_n.yaml` 一起放进主表，
把总增益拆成"backbone 的贡献"和"token 路径的贡献"两段。

## 运行顺序

单卡每个实验约 11–12 小时（200 轮，按每轮约 200 秒估算；以你日志里的 `time` 列为准）。

1. **`no_global_token`**：先跑。token 路径若只贡献零点几，后面的实验方向都要调整
2. **`token_budget_256`**：若明显优于 56，**主模型本身要改**，越早知道越好
3. **`no_geometric_writeback`**
4. **`no_ema_routing`**
5. **`no_routing_supervision`**：需要重跑。第一次 200 轮训练（提交 `3abbba5`）虽然路由配置相同，
   但早于小目标分配修复（STAL，`assigner.stal_size`），和现在的主模型差了不止一个变量，
   不能当作这一行。它的 `token_stats.csv`（`score_entropy` 一路逼近 1.0）仍可作为
   "无监督时分数图塌缩"的证据

## 额外价值：同样的配置，换 DroneVehicle 再跑

`#1` 和 `#3` 在 DroneVehicle 上各跑一次，**不需要新配置**，就能得到按昼夜拆分的结果
（`results_by_condition.csv` 自动生成）：

- `#1` 在夜间掉得比白天多 → 全局上下文在纹理退化时更重要
- `#3` 在夜间掉得比白天多 → 光照一致约束确实在起作用

这组交叉结果可能是全文最有说服力的证据，时间允许的话值得优先安排。

## 怎么跑

**一次跑完（推荐）**：仓库根目录的 `run_experiments.py` 按顺序跑完整模型、全部消融和基线，
可随时 Ctrl+C 中断，再运行会跳过已完成的、续训跑了一半的，结果汇总到
`runs/train/experiments_summary.csv`：

```bash
python run_experiments.py --dry-run   # 先看一眼要跑哪些
python run_experiments.py
```

**单独跑一个**：
独立脚本，同时改这两行，其余不动：

```python
MODEL_CONFIG = r"configs/ablation/no_global_token.yaml"
NAME = "abl_no_global_token"      # 每个实验换一个名字，否则会覆盖上一次的输出
```

**训练配方（学习率、调度、轮数、batch）所有实验必须完全一致**，否则差值里混进了配方的影响。

## P2 候选：全局上下文 × 细节注入（2×2）

不属于上面 5 个消融，而是候选的主模型改进，都针对 P2（2–8 px 目标在这一层检测）：

| 配置 | P2 得到全局上下文 | P2 做细节注入 |
|---|---|---|
| `models/model_main.yaml` | — | — |
| `writeback_p2.yaml` | ✓ | — |
| `detail_inject.yaml` | — | ✓ |
| `writeback_p2_detail_inject.yaml` | ✓ | ✓ |

细节注入（`models/token/detail_inject.py`）：从 stem 第二个卷积**之前**取 stride 2 的特征，
SpaceToDepth 后 1×1 卷积，在 P3 路由打分图标出目标的地方加回 P2：
`P2 = P2 + γ·M·Conv1×1(SpaceToDepth(S))`，γ 每通道一个、初始化为 0。
注入在 P2 的 local path 之前（`inject_at: before_local`），后面还有两层卷积来用它。
约 4.2K 参数、0.1G MACs。

为什么从 stem 取：`tools/analyze_p2_info.py` 用线性探针测 P2 通路上每一步还剩多少目标细节，
stem s2 → C2 → lat(C2) → FPN 后 P2 → 进 head 的 P2 依次是 0.99 → 0.85 → 0.82 → 0.44 → 0.34
（VisDrone val，初始化权重），细节主要丢在 stem 的第二个卷积和 FPN 相加。之前的
`detail_enhance.yaml` 放大的是进 head 前的 P2 自己的高频 —— 取源处正是细节最少的点，
补不回已经丢掉的东西。

为什么要路由掩码：目标框只占图像 5.9%，却有全图 26.7% 的高频能量；GT 中心热图（打分器的
监督目标，M > 0.3）用 5.1% 的面积覆盖 80% 的小目标 P2 格子。对照：

- `detail_inject_global.yaml`：全图注入，不用路由 —— 若和 `detail_inject` 一样好，增益来自早期细节本身，不是路由

四格里 ✓✓ 的增益大于两个单项之和，才说明"一次路由，两处受益"。
`train_candidates.py` 依次训练这四个（`main`、`cand_writeback_p2`、`cand_detail_inject`、
`cand_writeback_p2_detail_inject`）。

## 需要别的消融时

去掉 P2、去掉 FPN、改 Transformer 层数、全局 top-k、随机路由等都仍然支持，
对应的覆盖写法在 `tests/_variants.py` 里。要跑哪个，照着写一个三行的 YAML 即可，例如：

```yaml
_base_: [../models/model_main.yaml]
model:
  token:
    mixer_layers: 0
```
