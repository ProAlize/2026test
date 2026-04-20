# S3A v4: Bottleneck Conv + Spatial Norm (2026-04-20)

## 问题诊断

### A1: DWConv 无通道混合

原始 SpatiallyFaithfulAdapter 的空间路径使用 DWConv3×3 (groups=2048)：

```
spatial: Linear(1152→2048) → DWConv(2048, 3×3, groups=2048) → Linear(2048→768)
```

问题：
- DWConv groups=hidden_dim → 每个通道独立做 3×3 滤波，**零通道交互**
- iREPA (ICLR 2026) 证明空间结构迁移需要**跨通道空间建模**
- 空间路径 99.5% 参数在两个 2048-d Linear 上，DWConv 本身只有 18K
- 实际效果：空间路径退化为 per-channel 2D blur，token MLP 主导输出

### 缺少 Spatial Normalization

iREPA 的第二个关键发现：对 teacher target 做 spatial norm（减去 patch 均值）可增强空间对比度。
S3A 的 spatial_loss 做了 energy mean-norm，但 cosine_distance 和 affinity_loss 使用原始 DINO target，全局语义分量稀释空间信号。

## 修改方案

### 改动 1: Bottleneck Conv 替换 DWConv

```
旧: Linear(1152→2048) → DWConv(2048, 3×3, groups=2048) → Linear(2048→768)
新: Linear(1152→256)  → Conv2d(256, 256, 3×3)  → GELU → Linear(256→768)
```

核心思路：
- 降维到 bottleneck=256 后做**标准 Conv2d（full channel mix）**
- Conv2d(256,256,3×3) 的 590K 参数全部用于跨通道空间建模
- 加 GELU 激活增加非线性（原 DWConv 后无激活）
- Token MLP 路径 (2048-d) 不变，继续承担大容量通道投影

参数对比（per layer）：

| 组件 | 旧 | 新 |
|:---|---:|---:|
| Token MLP | 3,935K | 3,935K（不变）|
| Spatial Linear | 3,935K | 493K |
| Conv | 20K (DWConv) | 590K (full Conv) |
| **合计** | **7,894K** | **5,021K (-36%)** |

6 层总计：47.4M → 30.1M，**节省 17.2M 参数**。

### 改动 2: DINO Target Spatial Normalization

在 compute_s3a_alignment_loss 中，对 DINO target 做 spatial norm：

```python
dino_mean = dino_layer.mean(dim=1, keepdim=True)  # [B, 1, C]
dino_layer = dino_layer - dino_mean  # 去掉全局语义分量
```

效果：cosine_distance 和 affinity_loss 现在聚焦于 patch 间的**相对空间结构**，而非全局语义相似度。
可通过 --no-s3a-spatial-norm-target 关闭。

## 新增参数

| 参数 | 默认值 | 说明 |
|:---|:---|:---|
| `--s3a-spatial-bottleneck-dim` | 256 | Conv2d bottleneck 宽度。0 禁用空间路径 |
| `--s3a-spatial-norm-target` | True | 对 DINO target 做 spatial norm |

## 与 iREPA 的关系

iREPA 的两个改动：
1. MLP → Conv projector（我们用 bottleneck Conv 实现类似效果）
2. Spatial norm on target（我们直接采用）

区别：
- iREPA 用 Conv **替代** 整个 projector；S3A 用 Conv 作为 MLP 的**空间辅助路径**
- iREPA 单层；S3A 多层 tap + 双源 router
- S3A 的 token MLP 路径保留了比 iREPA 更大的通道投影容量

## Checkpoint 兼容性

新参数加入 backward_compatible_missing_keys：
- `s3a_spatial_bottleneck_dim`: legacy default = None（旧 DWConv 无此概念）
- `s3a_spatial_norm_target`: legacy default = False

旧 checkpoint 无法 strict load 到新 adapter（参数名变化：spatial_dw → spatial_conv + spatial_act）。
需要从头训练或 --allow-legacy-resume-args。

---

## A2: Router GAP — 结论：by design, 不修改

### 诊断

Router 对 256 个 token 做 GAP 后用 256-d MLP 决策，输出 per-sample per-layer 的标量权重 [α_dino, α_self]。
审计时标记为"丢失空间信息"，但经分析认为这是**正确的归纳偏置**。

### 不修改的理由

1. **Router 的职能是宏观调度**：回答"这层/这个噪声水平用多少 DINO"，不需要空间细节
2. **训练 metrics 确认 router 工作正常**：dual_alive=100%，entropy 健康，alpha 非均匀分配
3. **空间细粒度已由 adapter 负责**：bottleneck Conv 做 per-token 空间建模，和 router 的 per-layer 调度分工明确
4. **per-token routing 改动量过大**：alpha floor、policy KL、utility probe、collapse detection 全部需要 per-token 化，训练成本显著增加，收益不确定
5. **GAP 防止 router 过拟合**：只看全局统计量，避免 router 学到与 adapter 重叠的局部 pattern

### 文献支撑

- **Diff-MoE (ICML 2025)**：globally-aware feature recalibration 机制，per-layer 级别的全局信息整合用于动态资源分配。S3A 的 GAP router 是同一思路的简化版。
- **REED (NeurIPS 2025)**：source-level flexible guidance selection，按条件动态选择/融合表征。S3A 的 router 做 source-level mixture（DINO vs self），而非 token-level MoE routing。

### 状态：✅ Closed, by design

---

## B2: DINO Floor 0.05 — 结论：保留，ablation only

### 现状

两层 DINO 下限：
- 衰减 floor: `--s3a-dino-alpha-floor=0.1`，0→25k 步线性衰减到 0
- 永久 floor: `--s3a-protect-source0-min-alpha=0.05`，全程 α_dino ≥ 0.05

### 不修改的理由

1. **Floor 从未被触发**：训练中 a_dino ≈ 0.287，远高于 0.05，floor 是非活跃安全网
2. **双源都有等效保护**：DINO 用 alpha floor，self 用 collapse alarm + auto-mitigation，对称设计
3. **首次训练不动安全网**：在 400k FID 出来之前，不知道 floor 是否在某阶段悄悄救场
4. **代价可量化**：最坏情况 `0.05 × L_dino`，在 effective_lambda cosine decay 后期可忽略
5. **移除应作为 ablation**：论文消融表可跑 "S3A w/o DINO floor" 对照

### 文献支撑

- **DUPA (ICLR 2026)**：timestep ablation 证明教师端用干净 t（接近 ground truth）效果最好，noisy 教师（t=0.8~1）导致 FID 从 25.23 劣化到 30.36。S3A 的 DINO floor = 保证至少有一个可靠锚点参与融合，等价于 DUPA 的"teacher at clean t is ground truth"原则。
- **SRA (ICLR 2026) / SD-DiT (CVPR 2024)**：self-alignment 文献同样认为 t≈σ_min（最干净）的教师信号最可靠。DINO 作为 frozen clean-image encoder，始终提供这种"最干净的锚点"。

### 状态：✅ Closed, keep as safety net

---

## B3: EMA Layer 27 重复 — 结论：保留，shared semantic anchor

### 现状

Cross-layer 映射（offset=14, depth=28）：

| Student Layer | EMA Teacher Layer | Gap | 状态 |
|:---:|:---:|:---:|:---|
| 6 | 20 | +14 | 独占 |
| 13 | **27** | +14 | 共享 |
| 20 | **27** | +7 (capped) | 共享 |
| 27 | 27 (=self) | 0 | disabled |

Student 13 和 Student 20 共用 EMA layer 27 的 adapter。Student 27 无 self-source。

### 不修改的理由

1. **Layer 27 是 DiT 最深层**：包含最丰富的高层语义，作为"shared semantic anchor"被两个 student 对齐是合理的
2. **Router 可差异化调节**：虽然 EMA target 相同，不同层的 alpha_self 由 router 独立决定，实际融合权重不同
3. **Student 27 无 self-source 是正确的**：最深层最接近 DINO 语义空间，单源 DINO 可能是最优选择
4. **改 offset 或 layer 选择影响面大**：涉及 checkpoint 兼容、adapter 数量、EMA hook 层等多处联动
5. **整个领域缺乏数学最优性证明**：DUPA、HASTE 都是经验选层

### 文献支撑

- **DUPA (ICLR 2026)**：reviewer 问"多层同时对齐是否更好"，作者回答"DDT 上只对齐 condition encoder output（最深条件层）效果最优"。DiT layer 27 类似 DDT 的 condition encoder output，被复用有合理性。
- **HASTE (NeurIPS 2025)**：使用固定多层对齐（feature + attention），同样没有层选择的数学推导。
- **整个 REPA 家族**：均使用固定 encoder_depth（默认第 8 层），无自适应选层。

### 可选 ablation

论文消融表可对比：
- S3A offset=14 vs offset=7 vs offset=0（same-layer same-timestep，原始 v2）
- S3A 4-layer vs 2-layer（只用 [13, 27]）vs 1-layer（只用 [7]，对标 REPA）

### 状态：✅ Closed, keep as shared semantic anchor

---

## 总体数学严格性说明

| 设计选择 | 数学推导 | 经验证据 | 同行实践 |
|:---|:---:|:---:|:---|
| A1 Bottleneck Conv | ❌ | ✅ iREPA 27 encoder ablation | iREPA Conv projector |
| A2 Router GAP | ❌ | ✅ 训练 metrics 稳定 | Diff-MoE globally-aware recalib, REED source selection |
| B2 DINO floor | ❌ | ✅ DUPA timestep ablation | DUPA/SRA teacher-at-clean-t |
| B3 EMA layer overlap | ❌ | ✅ DUPA 多层 ablation | DUPA condition encoder output, HASTE multi-layer |

**整个 DiT representation alignment 领域目前没有严格数学理论**指导"几层对齐、什么时候停、哪个 source 更好"。REPA/iREPA/HASTE/DUPA 全部是 ablation-driven design。S3A 的 framing 应该是 **"principled empirical design backed by ablation + prior art"**。
