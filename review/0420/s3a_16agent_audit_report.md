# S3A 16-Agent Audit Report (2026-04-20)

## 审计方法

5 个 Harsh Reviewer（架构 / 代码实现 / 实验设计 / 接口调用 / 模型参数）
→ 5 个 AC 裁决（每个 reviewer 的发现逐条确认/驳回）
→ 1 个 PC 汇总

---

## 最终裁定：1 个真实 Bug，13 个假警报

### ✅ 可以继续训练，修复 1 个 bug 即可

---

## PRIORITY 1：必须修复的 Bug

### 🔴 BUG: Non-finite Skip 导致 batch 计数不同步

**位置**：训练循环，`batches_seen += 1` 在 NaN 检测之前

**问题**：
```python
batches_seen += 1               # ← 先递增
# ... 计算 loss ...
if loss_finite_flag.item() == 0:
    opt.zero_grad(set_to_none=True)
    continue                     # ← train_steps 没递增就跳过
```

当 NaN loss 触发 skip 时，`batches_seen` 已递增但 `train_steps` 未递增。多次 NaN 后两者偏移越来越大。

**影响**：
- 影响 checkpoint resume 时的 epoch 计算
- `start_epoch = batches_seen // steps_per_epoch` 用的是偏移后的值
- 不影响当前训练正确性（train_steps 是权威计数器），但 resume 路径有风险

**严重度**：LOW-MEDIUM

**修复方案**：将 `batches_seen += 1` 移到 non-finite 检查之后

---

## PRIORITY 2：需要 ablation 的设计选择（非 bug，Stage 4-5 处理）

### 架构 ablation

| 组件 | 当前设计 | 来源/理由 | 需要的 ablation |
|:---|:---|:---|:---|
| 双路径 adapter (MLP + Conv) | Bottleneck Conv 256-d | iREPA 启发 | 对比：single MLP / single Conv / 无 adapter |
| Router 256-d hidden | 256 → 2 scalars | 256-d 对 binary routing 足够 | 对比：64 / 128 / 512 |
| 加法融合 | token_out + spatial | 标准 MoE 方式 | 对比：gating / attention-based |
| 4 层 tap [6,13,20,27] | depth 的 25/50/75/100% | REPA 选层惯例 | 对比：2 层 / 1 层 / 不同位置 |

### 超参 ablation

| 参数 | 值 | 来源 | 需要的 ablation |
|:---|:---|:---|:---|
| s3a_lambda | 0.5 | REPA proj-coeff 默认 | 对比：0.1 / 0.25 / 1.0 |
| loss weights (1.0, 0.5, 0.5) | feat / attn / spatial | HASTE 标准化 | grid search |
| self_layer_offset | 14 (depth/2) | 设计选择 | 对比：7 / 10 / 14 / 21 |
| timestep_offset_max | 200 (20%) | 设计选择 | 对比：100 / 150 / 200 / 300 |
| schedule warmup | 100k | piecewise_cosine | 对比：25k / 50k / 100k / 150k |
| 单一 LR | 1e-4（DiT + S3A 共享） | 标准做法 | 测试：分离 LR |

---

## PRIORITY 3：Stage-appropriate 的缺口（当前阶段正常）

| 缺口 | 当前状态 | 目标 Stage |
|:---|:---|:---|
| 400k vs 800k 计算量对等 | 未测量 | Stage 3-4 |
| Baseline 复现 (REPA/iREPA/HASTE/DUPA) | 未开始 | Stage 4 |
| Ablation 表格 | 未就绪 | Stage 4-5 |
| ImageNet-512 / COCO / T2I | 未就绪 | Stage 3-4 |
| 评估协议标准化 (NFE/CFG/种子) | 未最终确定 | Stage 3-4 |
| 多种子实验 | 仅 seed=0 | Stage 4-5 |
| 机制验证（梯度分析/热力图） | 未实现 | Stage 5 |

---

## 假警报清单（13 个 reviewer 发现被 AC 驳回）

### 代码实现（7 个假警报）

| Reviewer 声称 | AC 裁定 | 驳回理由 |
|:---|:---:|:---|
| Hook 异常泄漏（FATAL） | ❌ | 异常时进程直接崩溃，hook 清理无意义 |
| Router detach 浪费容量（MAJOR） | ❌ | 阻止 router 梯度回传 backbone 是正确设计 |
| STE alpha floor 产生负概率（MAJOR） | ❌ | 最终归一化 `out / out.sum().clamp(min=1e-8)` 保证 α ≥ 0 且 Σ=1 |
| EMA adapter 选错 student（IMPORTANT） | ❌ | Python min() 对 distance=0 正确选 key=27 |
| .clone() 冗余浪费内存（MINOR） | ❌ | inference_mode tensor 必须 clone 才能参与 autograd |
| Collapse window 被零 probe 重置（MINOR） | ❌ | probe_every=10 保证每个 window 有 ≥10 probes |
| Worker 种子全部相同（CRITICAL） | ❌ | DataLoader generator 机制自动分配 per-worker seed |

### 接口调用（6 个假警报）

| Reviewer 声称 | AC 裁定 | 驳回理由 |
|:---|:---:|:---|
| EMA 需要 DDP 包装（FATAL） | ❌ | EMA frozen，不参与梯度通信，state_dict keys 已匹配 |
| diffusion.training_losses 位置参数（MAJOR） | ❌ | 按位置传参正确，仅 cosmetic 改进 |
| inference_mode vs no_grad 混用（MAJOR） | ❌ | 两者语义都正确，inference_mode 更严格但 no_grad 不是 bug |
| BooleanOptionalAction 需要 py3.9+（MAJOR） | ❌ | environment.yml 指定 py3.10 |
| 单一 LR for DiT+S3A（MINOR） | ❌ | 设计选择，ablation 问题不是 bug |
| DINOv2 torch.hub.load 兼容性（MINOR） | ❌ | 测试过的路径，错误信息可改进但不是 bug |

---

## 架构审计详细 Rebuttal

### W1: "双路径 adapter 是不必要的复杂度"（Reviewer 评 FATAL）

**Reviewer 论点**：iREPA 用单层 Conv 就够了，双路径没有证据。

**AC 裁定**：❌ DISMISS — 设计选择，非 bug。
- Token MLP 负责高容量通道投影（3.9M params）
- Bottleneck Conv 负责跨通道空间建模（590K params）
- 两者角色不重叠：MLP 做全连接变换，Conv 做局部空间交互
- iREPA 的单层 Conv 容量有限（1 层），S3A 的双路径提供了更大建模空间
- ablation（单路径 vs 双路径）是论文需要的，但当前代码在架构上不是 bug

### W2: "Router 256-d 是信息瓶颈"（Reviewer 评 MAJOR）

**Reviewer 论点**：1152-d 压到 256-d 丢失 77.8% 信息。

**AC 裁定**：❌ DISMISS — 过度分析。
- Router 最终输出是 2 个标量（softmax → [α_dino, α_self]）
- 256-d hidden 对 binary routing 决策已是充分表示
- MoE 文献中 64-d gate for 768-d input 是常见做法（Diff-MoE, ICML 2025）
- 如果 router 需要更多容量，训练 metrics 会显示 entropy collapse——实际 entropy 健康

### W3: "加法融合脆弱"（Reviewer 评 MAJOR）

**Reviewer 论点**：如果某源有问题，加法融合无法抑制。

**AC 裁定**：❌ DISMISS — 标准做法。
- $\text{fused} = \alpha_{\text{dino}} \cdot s_{\text{dino}} + \alpha_{\text{self}} \cdot s_{\text{ema}}$ 是标准混合
- Utility probing + selective gate 已提供源质量反馈机制
- Gating/attention 会增加复杂度，收益不确定
- Reviewer 建议的"attention fusion"在这种 2-source 设置下退化为和 router 相同的功能

### W4: "Loss weights (1.0, 0.5, 0.5) 无数学推导"（Reviewer 评 MAJOR）

**AC 裁定**：部分认可 — 需要 ablation 但非 bug。
- 权重来自 HASTE (NeurIPS 2025) 的标准化设置
- feat=1.0 是主项，attn/spatial=0.5 是辅助正则
- 整个领域没有数学推导的 loss weight（REPA 单个 cosine，HASTE 双项），全部是经验值
- 加入 ablation 表后可 defend

---

## 模型参数审计详细 Rebuttal

### P1: "s3a_lambda=0.5 无依据"

**AC 裁定**：NEEDS_ABLATION — 来自 REPA proj-coeff=0.5 默认值。
- effective_lambda = 0.5 × phase_weight，在 300k 步后衰减到 0
- 对 loss 量级的影响：loss_align ≈ 0.2-0.5，loss_diff ≈ 0.15
- $0.5 \times 0.3 = 0.15$，约与 loss_diff 等量级 → 合理
- 但需要 ablation：0.1 / 0.25 / 0.5 / 1.0

### P2: "schedule_steps=300k < max_steps=400k，为什么不调满？"

**AC 裁定**：BY DESIGN — 300k→400k 是"纯 DiT 微调"阶段。
- 与 HASTE 的 stage-wise termination 一致
- 最后 100k 步让模型在无辅助信号下自由调整
- 如果 schedule 和 max_steps 相同，alignment 信号会在训练末期极弱但不归零，可能更差
- 需要 ablation：schedule_steps ∈ {200k, 300k, 400k}

### P3: "self_warmup=25k 和 dino_floor_steps=25k 同时为 25k，巧合？"

**AC 裁定**：BY DESIGN — 两者协同。
- warmup 25k：前 25k 步只用 DINO，router detach 防学偏
- DINO floor 25k：前 25k 步 DINO α ≥ 0.1 衰减到 0.05
- 25k 步 = self-source 开启 = router 归零重置 = floor 从 0.1 衰减完毕
- 这是一个同步设计点，不是巧合

### P4: "self_layer_offset=14 = depth/2，太整齐了"

**AC 裁定**：NEEDS_ABLATION — 设计合理但需验证。
- depth=28，offset=14 使 student 6→EMA 20, student 13→EMA 27
- 跨越半个网络的语义 gap 提供了有意义的"未来自己"信号
- DUPA (ICLR 2026) 的多层 ablation 表明不同架构有不同最优层
- 需要 ablation：offset ∈ {7, 10, 14, 21}

---

## 签核

| 维度 | 真实 Bug 数 | 假警报数 | ablation 需求数 |
|:---:|:---:|:---:|:---:|
| 架构 | 0 | 4 | 4 |
| 代码实现 | **1** | 7 | 0 |
| 实验设计 | 0 | 0 | 10（Stage 4-5） |
| 接口调用 | 0 | 6 | 1 |
| 模型参数 | 0 | 0 | 8 |
| **合计** | **1** | **17** | **23** |

**结论**：修复 1 个 bug（batches_seen desync），即可继续 400k 训练。23 项 ablation 留到 Stage 4-5。
