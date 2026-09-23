# 消融实验索引

所有消融都继承 `configs/models/model_main.yaml`，**相对主模型只改一个变量**，
因此每一行和主模型的差值都能干净地归因到那一处改动。

主模型本身不在这个文件夹里，下表中标 **(主)** 的行指的就是它。

## 怎么跑

独立脚本：只改 `MODEL_CONFIG` 一行，其余不动。

```python
MODEL_CONFIG = r"configs/ablation/mixer_none.yaml"
NAME = "abl_mixer_none"        # 每个实验换一个名字，否则会覆盖输出目录
```

或走仓库入口：

```bash
python tools/train.py --config configs/datasets/visdrone_rgb.yaml configs/ablation/mixer_none.yaml
```

> 这些配置只改**模型结构**。训练配方（学习率、调度、轮数、batch）统一来自
> `_base_/schedule.yaml`，用独立脚本时则由脚本顶部常量决定。**所有消融必须用同一套
> 训练设置**，否则差值里混进了配方的影响。

---

## A. 组件消融 —— 每个部分值多少分

| 配置 | 去掉什么 | 回答的问题 |
|---|---|---|
| `no_global_token` | 整条 token 路径 | **核心创新整体值多少分**。也兼作"TinyNeXt 无 token"基线，一次训练两张表共用 |
| `no_local_cnn` | 局部 CNN 路径 | 局部细节建模的贡献 |
| `no_p2` | 整个 P2 分支 | 高分辨率分支值不值它的算力（P2 约占 CNN 侧 MACs 的 65%） |
| `no_fpn` | FPN 自顶向下融合 | 跨尺度融合对小目标的作用 |

## B. Token 预算扫描 —— 56 个够不够

| 配置 | P3 | P4 | P5 | 合计 |
|---|---|---|---|---|
| `token_budget_56` **(主)** | 32 | 16 | 8 | 56 |
| `token_budget_128` | 64 | 48 | 16 | 128 |
| `token_budget_256` | 128 | 96 | 32 | 256 |
| `token_budget_512` | 256 | 192 | 64 | 512 |

画成"精度 - token 数 - 延迟"曲线。持续上升说明 56 太少；有拐点则拐点即最优预算；
一直平坦说明全局路径本身贡献有限。

## C. Token 来源层 —— 细粒度层要不要贡献 token

总预算固定 56，写回目标固定 P3/P4/P5，**只改 token 从哪些层选出来**。

| 配置 | 来源层 | 预算分配 |
|---|---|---|
| `token_src_p5` | P5 | 56 |
| `token_src_p4p5` | P4 + P5 | 48 + 8 |
| **(主)** | P3 + P4 + P5 | 32 + 16 + 8 |

## D. 路由方式 —— 选 token 的策略

| 配置 | 策略 | 回答的问题 |
|---|---|---|
| `random_routing` | 打分器不训练 ≈ 随机选 | **你的选择比随机选好吗？** 审稿人必问 |
| `global_topk_routing` | 全图 top-k（grid=1） | 局部候选路由是否必要 |
| **(主)** | 网格内 top-k + 打分门控 | — |

## E. Token 间交互 —— Transformer 部分

| 配置 | 自注意力层数 | 回答的问题 |
|---|---|---|
| `mixer_none` | 0 | **token 之间的全局关系建模到底有没有用** |
| **(主)** | 1 | — |
| `mixer_deep` | 2 | Transformer 容量是不是瓶颈（neck 不加宽，只加 token 路径） |

`mixer_none` 和 `no_global_token` 要区分：前者仍然选 token、仍然写回，只是 token
之间不交换信息；后者整条路径都没了。两者的差值就是"选择 + 写回"单独的贡献。

## F. 写回方式 —— 主创新

| 配置 | 写回方式 | 回答的问题 |
|---|---|---|
| `broadcast_writeback` | 均匀注意力 | 最朴素的基线 |
| `no_geometric_writeback` | 仅内容注意力 | **去掉高斯几何先验，精确隔离主创新的贡献** |
| **(主)** | 内容 + 几何先验 | — |
| `writeback_p2` | 额外写回 P2 | 全局上下文注入高分辨率层对小目标有没有帮助 |

`writeback_p2` 显存占用明显更高，若 OOM 可单独调小 batch，但要在实验记录里注明。

## G. EMA 光照一致性

| 配置 | 设置 | 回答的问题 |
|---|---|---|
| `no_ema_routing` | 关闭一致性约束 | **光照不变路由（方案 A）的贡献** |
| `ema_same_view` | teacher 与 student 同输入 | 说明为什么必须用不对称视图（该设置下损失会衰减到 ~0） |
| **(主)** | 光度扰动视图 | — |

---

## 建议的运行顺序

单卡每轮约 16–18 小时，按信息量排：

1. `no_global_token` —— 决定核心创新是否成立，**最先跑**
2. `token_budget_256` —— 挑一个远离 56 的点，快速看出预算曲线趋势
3. `no_geometric_writeback` —— 主创新的直接证据
4. `mixer_none` —— Transformer 部分是否有用
5. `random_routing` —— 应对"是否优于随机"的质疑
6. 其余按需补齐

如果第 1 步显示 token 路径只贡献零点几，先停下来重新审视设计，再决定后面跑什么。
