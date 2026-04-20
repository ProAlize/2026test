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
