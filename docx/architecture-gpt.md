# Stage-wise Spatial Representation Alignment for DiT

## 文档职责（架构真值源）

- 文件角色：`方法结构与实现接口` 的唯一真值源。
- 负责范围：模块划分、张量定义、训练/推理数据流、MVP 与完整版的结构差异。
- 不负责范围：实验排期和算力分配（见 `plan.md`）、主表与指标协议（见 `exp.md`）、文献分级与项目复盘（见 `sum.md`）。
- 冲突仲裁：若与 `plan-gpt.md`、`sum.md` 冲突，以本文件定义的结构规范为准。
- 统一入口：见 `doc-roles.md`。

## 1. 架构目标

这份文档只回答一个问题：

**如果论文主线正式改成 `Stage-wise Spatial Representation Alignment for DiT`，理想的模型架构到底长什么样。**

这里不再写宽泛原则，而是直接给出：

- 模块划分
- 数据流
- 张量形状
- 训练与推理路径
- MVP 版和完整版的区别
- 为什么这样设计

默认任务设定：

- 数据：`ImageNet-1K`
- 分辨率：`256 x 256`
- 任务：`class-conditional generation`
- Backbone：`DiT-B/2` 用于机制验证，`DiT-XL/2` 用于主表
- latent：标准 `VAE latent`

---

## 2. 一句话架构定义

**Frozen DINOv3 SourceBank + EMA self-source option + spatial residual projector on selected DiT mid-blocks + sparse timestep-layer router + fixed stage-wise controller**

这句话对应的完整含义是：

- DiT 主干仍然负责标准扩散去噪
- 训练期额外引入一个 alignment side branch
- side branch 只在少数中间层工作
- side branch 对齐的是**空间特征图**，不是全局 pooled token
- side branch 的对齐强度由训练阶段控制
- side branch 的目标 source layer 由一个轻量 router 选择
- 推理期整个 side branch 被完全移除

---

## 3. 整体模块图

```text
RGB image x -------------------------> VAE Encoder -----------------> latent z
       |                                                          |
       |                                                          v
       |                                               noise schedule -> z_t
       |                                                          |
       |                                                          v
       |                                             +----------------------+
       |                                             |       DiT Backbone   |
       |                                             |   blocks 1 ... L     |
       |                                             +----------------------+
       |                                                       |
       |                                 selected mid-block hidden states h_b
       |                                                       |
       |                                                       v
       |                                          Spatial Projector Bank P_b
       |                                                       |
       |                                                       v
       |                                            projected DiT maps q_b
       |                                                       |
       |                                                       +------+
       |                                                              |
       v                                                              v
SourceBank (external or self)                               Timestep-Layer Router
  - DINOv3 branch                                             input: t, block id, h_b summary
  - DINOv2 branch                                             output: source-layer weights
  - EMA-DiT self branch                                              |
       |                                                              |
       v                                                              v
 multi-layer spatial features s_mk --------------------------> weighted source target
                                                                       |
                                                                       v
                                                         token-level cosine alignment loss
                                                                       |
                                             stage controller lambda(s) *
                                                                       |
                                                                       v
                                                     L = L_diff + lambda * L_align
```

---

## 4. 张量定义与默认尺寸

为了把架构讲清楚，先固定一组默认张量记号。

### 4.1 输入与 latent

- 输入图像：`x ∈ R^{B×3×256×256}`
- VAE latent：`z ∈ R^{B×4×32×32}`
- 加噪 latent：`z_t ∈ R^{B×4×32×32}`

### 4.2 DiT token

对 `32×32` latent 用 patch size `2` patchify：

- token grid：`16 × 16`
- token 数：`N = 256`
- block hidden：`h_b ∈ R^{B×256×D}`

其中：

- `D = 768` 对应 `DiT-B/2`
- 更大 backbone 时 `D` 增大，但 token grid 不变

### 4.3 Source 特征

对 external source 或 self-source，统一抽成空间特征图：

- source branch `m`
- source candidate layer `k`
- feature map：`s_{m,k} ∈ R^{B×C_s×H_s×W_s}`

统一之后再 adapter 到目标对齐网格：

- adapted source map：`ŝ_{m,k} ∈ R^{B×C_a×16×16}`

建议默认：

- `C_a = 1024`

### 4.4 DiT 对齐特征

从 DiT 某个 tap block 抽出的 hidden state：

- `h_b ∈ R^{B×256×D}`

先恢复成空间图：

- `H_b ∈ R^{B×D×16×16}`

经过 spatial projector：

- `q_b ∈ R^{B×C_a×16×16}`

Alignment 实际比较的是：

- `q_b`
- `ŝ_{m,k}`

---

## 5. 五个核心模块

## 5.1 模块 A：DiT Backbone

### 作用

保持原始 DiT 去噪主干不变，作为整个模型的主体。

### 输入

- `z_t`
- timestep embedding `e_t`
- class label embedding `e_y`

### 输出

- 逐 block hidden states
- 最终噪声预测

### 设计要点

- 不在每个 block 都引入 alignment 支路
- 只选择中间 `4` 到 `6` 个 block 作为 `tap blocks`

### 为什么

原因不是省代码，而是结构上更合理：

- 中层往往同时包含语义和布局信息
- 全层对齐很难归因
- 全层对齐会显著增加算力和显存

---

## 5.2 模块 B：SourceBank

### 作用

统一管理所有 source，把 external teacher 和 self-source 都变成同一种接口。

### 推荐包含的 source

主文默认三条：

1. `DINOv3-ViT-L/16`
2. `DINOv2-ViT-L/14`
3. `EMA-DiT self-source`

### 输入

- `x_src`

其中：

- `DINOv3` 分支默认输入 `256×256`
- `DINOv2` 分支输入使用 `14` 的倍数，建议 `252×252`
- `EMA-DiT` 分支输入直接来自当前训练图像经 VAE 后的中间表征对应路径

### 输出

每个 source 分支输出少量候选层：

- `K = 3` 或 `4` 个 tap layers

统一表示为：

- `S_m = { s_{m,1}, s_{m,2}, ..., s_{m,K} }`

### 内部子模块

每个 source 都应带一个小 adapter：

- `1x1 conv` 或 linear projection 对齐 channel
- resize / interpolate 到 `16×16`

最终得到：

- `ŝ_{m,k} ∈ R^{B×C_a×16×16}`

### 为什么这样设计

这是整个架构里最重要的“统一接口”思想：

- external teacher 和 self-source 用相同 API
- source modularity 是实验接口，不是方法 headline
- 后续 projector、router、loss 都不需要为某个 source 写特例

---

## 5.3 模块 C：Spatial-preserving projector

### 作用

把 DiT block hidden state 转成适合和 source 空间特征对齐的表示，同时尽量保留二维拓扑。

### 输入

- `h_b ∈ R^{B×256×D}`

### 处理流程

1. reshape 成 `H_b ∈ R^{B×D×16×16}`
2. 经过 spatial projector

推荐 projector 结构：

```text
H_b
 -> 1x1 Conv (D -> C_a)
 -> DWConv 3x3
 -> Norm
 -> SiLU
 -> 1x1 Conv (C_a -> C_a)
 -> Residual
 = q_b
```

### 输出

- `q_b ∈ R^{B×C_a×16×16}`

### 为什么不用 MLP

因为这里的核心主张不是“做更强映射”，而是“保留空间结构”。

如果直接：

- flatten
- 全局 mixing
- 大 MLP

那么你的架构主张会被自己破坏。

### 为什么推荐 `DWConv 3x3 + 1x1`

因为这是一个很好的中间点：

- 有局部邻域感受野
- 保留 grid topology
- 参数量和 FLOPs 可控
- 比 full attention projector 更容易 defend

---

## 5.4 模块 D：Timestep-layer router

### 作用

在每个 tap block、每个 timestep 上，决定“当前 DiT block 应该看 source 的哪一层”。

### 输入

建议输入三部分：

1. 当前 diffusion timestep embedding：`e_t`
2. 当前 block id embedding：`e_b`
3. 当前 block hidden summary：`g_b = GAP(H_b)`

拼接后：

- `r_b = [e_t ; e_b ; g_b]`

### Router 形态

第一版建议用轻量门控，不要用重 cross-attention。

推荐结构：

```text
r_b
 -> MLP
 -> logits over K candidate source layers
 -> top-k sparse softmax
 -> alpha_b ∈ R^K
```

如果主文只做单 source routing，那么：

- router 只在该 source 的 `K` 个层之间选择

如果扩展到多 source joint routing，可把输出扩展为：

- `alpha_b ∈ R^{M×K}`

但这不建议作为第一版主文结构。

### 输出

- `alpha_b`

然后构造目标 source map：

\[
\tilde{s}_b = \sum_k \alpha_{b,k}\,\hat{s}_{m,k}
\]

### 为什么这样设计

因为你要回答的问题不是“attention 更强吗”，而是：

- 不同 timestep 是否偏好不同 source depth
- 不同 block 是否偏好不同 source depth

轻量 router 刚好能回答这个问题，而且：

- 计算便宜
- 解释性强
- 可画 heatmap

---

## 5.5 模块 E：Stage controller

### 作用

控制 alignment branch 在训练过程中的开启强度。

### 输入

- 当前训练 step 或 epoch 归一化阶段 `s`

### 输出

- `lambda_align(s)`

### 主文默认设计

推荐固定 schedule：

- 前 `10%` training steps：`lambda_align = λ0`
- `10%` 到 `40%`：线性或余弦衰减到 `0`
- 后续：`0`

### 为什么主文默认不用 adaptive stop

因为 adaptive stop 虽然实用，但作为主文默认会引出额外质疑：

- 是不是在用验证 FID 调训练
- 是不是变相 tuning 到 metric

因此：

- 固定 stage-wise controller 做主文默认架构
- adaptive stop 只做附录扩展

### 为什么这个模块是一级公民

因为 `HASTE` 已经告诉你：

- alignment 的帮助不是单调增加
- 训练后期继续强对齐会带来过约束

所以 stage controller 不是 recipe，而是架构定义的一部分。

---

## 5.6 模块 F：Self-alignment branch

### 作用

给整个架构提供一个不依赖 external teacher 的对照分支。

### 设计方式

维护一个：

- `EMA DiT teacher`

特点：

- 与学生 DiT 同 backbone
- 不反传梯度
- 只取中间 tap blocks 的空间特征

### 输入

- 与学生侧同样的训练样本对应路径

### 输出

- 与 external source 相同格式的多层空间特征：
  `ŝ_{self,k} ∈ R^{B×C_a×16×16}`

### 为什么是 EMA DiT，而不是另训一个内部 encoder

因为 EMA DiT 的好处是：

- 变量更少
- 与主干结构完全一致
- 与 external source 的比较更公平
- 符合 `SD-DiT / SRA / DUPA` 所暗示的“模型内生表示也可能足够”的方向

---

## 6. 训练期完整数据流

下面是一次训练 step 的完整数据流。

### 6.1 主干去噪路径

1. 输入图像 `x`
2. VAE encode 得到 `z`
3. 加噪得到 `z_t`
4. 输入 DiT 主干
5. 得到每个 block hidden state 和最终噪声预测
6. 计算 `L_diff`

### 6.2 Source 路径

并行处理 source：

#### external source

1. 输入 `x_src`
2. 进入 frozen `DINOv3` 或 `DINOv2`
3. 抽取 `K` 个候选层 feature maps
4. 通过 adapter/resampler 统一成 `16×16` + `C_a`

#### self source

1. 当前训练图像相关路径进入 `EMA DiT`
2. 抽取对应 tap blocks feature maps
3. 统一成 `16×16` + `C_a`

### 6.3 对齐路径

对每个被选中的 DiT tap block `b`：

1. 取 `h_b`
2. reshape 成 `H_b`
3. 经 spatial projector 得到 `q_b`
4. router 根据 `e_t + e_b + GAP(H_b)` 产生 `alpha_b`
5. 用 `alpha_b` 对 source 候选层加权得到 `\tilde{s}_b`
6. 计算 token-level cosine alignment loss：

\[
L_{align}^{(b)} = 1 - \cos(q_b, \tilde{s}_b)
\]

7. 聚合所有 tap blocks：

\[
L_{align} = \sum_{b \in \mathcal{B}} L_{align}^{(b)}
\]

8. 由 stage controller 给出当前训练阶段权重：

\[
L = L_{diff} + \lambda_{align}(s)\,L_{align}
\]

---

## 7. 推理期完整数据流

推理期必须大幅简化。

### 推理期保留

- VAE decoder
- DiT 主干

### 推理期删除

- SourceBank
- DINOv3 / DINOv2 分支
- EMA self-source 分支
- Spatial projector
- Router
- Stage controller
- Alignment loss

### 为什么

因为你要 defend 的不是“teacher-assisted inference”，而是：

**alignment 作为训练归纳偏置，提高了 DiT 的学习过程。**

这也是整个架构成立的关键：

- 训练复杂度增加
- 推理复杂度保持原 DiT 水平

---

## 8. MVP 版 vs 完整版

## 8.1 MVP 版

MVP 的目标不是投稿，而是先验证主假设。

### 结构

- Backbone：`DiT-B/2`
- SourceBank：只开一个 external source + 一个 self-source
- 默认 external source：`DINOv3`
- Tap blocks：`4`
- Projector：`1x1 -> DWConv3x3 -> Norm -> 1x1`
- Router：单 source 内部的 layer routing
- Stage controller：固定 early-stop

### 不做的东西

- 多 source joint routing
- adaptive stop
- dual-teacher fusion
- heavy cross-attention router
- 多 loss 配方

### MVP 的目的

只回答三个问题：

1. spatial projector 相比原版 REPA 是否有效
2. fixed early-stop 是否比 full-course 更合理
3. DINOv3 与 self-alignment 哪个更适合作为主线

---

## 8.2 完整版

完整版是在 MVP 成立后再做。

### 增强项

- 加入 `DINOv2` 作为历史 control
- 增加 source pilot：`MAE`, `SigLIP`
- 增加 routing 变体：`layer-only`, `timestep-only`, `top-k sparse`
- 增加 adaptive stop 作为附录
- 上 `DiT-XL/2`

### 完整版目标

- 构造公平主表
- 提供机制图
- 明确 external vs self 的边界

---

## 9. 推荐的默认架构实例

如果你现在就要一个“可以开始实现”的默认版本，我推荐如下配置。

### Backbone

- `DiT-B/2`
- tap blocks：第 `4, 7, 10, 13` 层

### SourceBank

- main external: `DINOv3-ViT-L/16`
- control external: `DINOv2-ViT-L/14`
- self source: `EMA DiT-B/2`

### Source adapters

- 每个 source candidate layer：
  - `1x1 conv` 对齐到 `C_a = 1024`
  - bilinear resize 到 `16×16`

### Spatial projector

每个 tap block 一个 projector：

```text
[B,256,D]
 -> reshape [B,D,16,16]
 -> 1x1 conv (D -> 1024)
 -> DWConv 3x3
 -> LayerNorm/GroupNorm
 -> SiLU
 -> 1x1 conv (1024 -> 1024)
 -> residual
```

### Router

```text
input = concat(
  timestep embedding 256d,
  block embedding 64d,
  GAP(current block map) 256d
)
 -> MLP 576 -> 256 -> K logits
 -> top-k sparse softmax
```

默认：

- `K = 4`
- `top-k = 2`

### Stage controller

- `0% ~ 10%`: `lambda = 0.1`
- `10% ~ 40%`: cosine decay `0.1 -> 0`
- `40% ~ 100%`: `lambda = 0`

### Loss

\[
L = L_{diff} + \lambda(s)\frac{1}{|\mathcal{B}|}\sum_{b \in \mathcal{B}} \left(1 - \cos(q_b,\tilde{s}_b)\right)
\]

---

## 10. 为什么这是“理想架构”

这套设计好，不是因为它模块多，而是因为每个模块都对应了一个必须回答的文献问题：

- `Original REPA path` 回答：你到底改进了什么
- `Spatial projector` 回答：空间结构是否关键
- `Stage controller` 回答：对齐是否应在中后期关闭
- `Router` 回答：不同 timestep / block 是否需要不同 source depth
- `Self-alignment branch` 回答：external teacher 是否真的必要

更重要的是，它满足一个非常关键的工程目标：

**训练期把 alignment 作为归纳偏置注入，推理期保持原 DiT 复杂度。**

这会让你的论文比“推理期也依赖 teacher 的方案”更容易 defend。

---

## 11. 当前最推荐的版本

如果你现在就开始实现，我建议：

**先实现这个，不要再加东西：**

- `DiT-B/2`
- `DINOv3 SourceBank`
- `EMA self-source`
- `4` 个 tap blocks
- `spatial residual projector`
- `single-source layer router`
- `fixed stage-wise controller`
- `token-level cosine loss`

同时保留两个关键对照：

- `Original REPA + DINOv2`
- `Your architecture + DINOv2`

这样你就能在不把系统做得过重的情况下，得到最关键的三条证据：

1. 架构增益是否真实存在
2. DINOv3 是否真的值得做主线
3. self-alignment 是否会推翻 external teacher 主线

---

## 12. 关键来源

- REPA: https://openreview.net/forum?id=DJSZGGZYVi
- HASTE: https://openreview.net/forum?id=HK96GI5s7G
- REED: https://openreview.net/forum?id=cIGfKdfy3N
- iREPA: https://openreview.net/forum?id=y0UxFtXqXf
- DUPA: https://openreview.net/forum?id=ALpn1nQj5R
- RAE: https://openreview.net/forum?id=0u1LigJaab
- SRA: https://arxiv.org/abs/2505.02831
- SD-DiT: https://cvpr.thecvf.com/virtual/2024/poster/30284
- DINOv2: https://arxiv.org/abs/2304.07193
- DINOv3: https://arxiv.org/abs/2508.10104
