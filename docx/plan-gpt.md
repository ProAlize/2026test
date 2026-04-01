# plan-gpt: Stage-wise Spatial Representation Alignment for DiT

## 文档状态（历史推导稿）

- 当前状态：`草案/备选池`，不作为执行真值源。
- 保留目的：记录推导过程、候选设定和被比较过的方案。
- 不再承担：当前实验协议定义、当前架构规范定义、当前排期定义。
- 执行时请优先查看：
  - `plan.md`（排期与资源）
  - `exp.md`（实验协议）
  - `architecture-gpt.md`（结构规范）
- 统一入口：见 `doc-roles.md`。

## 1. 核心判断

当前项目如果继续沿用旧版 `Ada-REPA` 的写法，最容易被审成“把 `REPA + iREPA + HASTE + REED` 的正确方向组合到一起”。  
因此，新主线必须收缩成一句更硬的主张：

**有效的 DiT representation alignment 不是 teacher 越强越好，而是表征必须保留空间结构、对齐必须分阶段进行，routing 只是实现这两点的手段。**

这条主线的优势是：

- 能直接回应 `REPA`：为什么 alignment 有效。
- 能直接回应 `HASTE`：为什么全程对齐不是最优。
- 能直接回应 `iREPA`：为什么空间结构比纯全局语义更关键。
- 能直接回应 `SRA / DUPA`：为什么 external teacher 不是默认必需品。
- 能直接回应 `RAE`：本文研究的是标准 `DiT + VAE + class-conditional` 设定下 alignment 的有效条件，而不是宣称 alignment 是唯一未来。

## 2. 论文主问题与三条假设

### 2.1 主问题

在 `ImageNet-256 class-conditional DiT` 设定下，表征对齐真正起作用的关键是什么：

- 空间结构保真
- `timestep × layer` 的合理匹配
- 训练阶段性的注入与终止

### 2.2 三条待验证假设

1. `H1`: 比起更强的全局语义，**保留空间结构的表征**更能稳定提升 DiT 训练。
2. `H2`: alignment 的主要收益集中在训练早期，**全程对齐不是最优策略**。
3. `H3`: 不同 source 的有效性依赖 `timestep × layer`，因此**routing 比固定 matching 更合理**。

## 3. 任务边界与作用域

### 3.1 主战场

- `ImageNet-1K`
- `256x256`
- `class-conditional generation`

### 3.2 可选扩展

- `ImageNet-512`

### 3.3 明确不做的事

- 不把 `text-to-image`
- 控制生成
- 图像编辑
- 不同 latent 范式

直接混入主表。

### 3.4 结论作用域

主文结论只声称在标准 `DiT + VAE + ImageNet-256 class-conditional` 设定下成立。  
除非补第二战场，否则不写成跨任务普适规律。

## 4. 最小可行方法定义

### 4.1 训练目标

默认训练目标定义为：

\[
L = L_{diff} + \lambda(s)\sum_{b \in \mathcal{B}} w_b(t)\, L_{align}^{(b)}
\]

其中：

- `s` 是训练阶段
- `t` 是 diffusion timestep
- `b` 是被选中的 DiT tap blocks
- `w_b(t)` 是 routing 权重
- `L_align` 默认使用 token-level cosine loss

### 4.2 四个保留组件

1. `Spatial-preserving projector`
2. `Timestep-layer routing`
3. `Stage-wise termination`
4. `Teacher/source modularity`

但这四者的地位不相同：

- headline contribution: `spatial preservation + stage-aware alignment`
- supporting mechanism: `routing`
- experiment interface: `source modularity`

## 5. 推荐架构修改

真正要改的不是 DiT 主干，而是训练期的 alignment side branch。  
推理路径应尽量保持原 DiT。

### 5.1 模块一：SourceBank

统一所有 source 的接口，暴露：

- `encode(image) -> {z_1, z_2, ..., z_k}`
- 每个 `z_i` 为多层空间 feature
- 所有 source 都经过统一 adapter 对齐 channel dim

支持的 source：

- `DINOv2`
- `DINOv3`
- `MAE`
- `SigLIP`
- `self-alignment (EMA DiT)`

### 5.2 模块二：Spatial-preserving projector

不要使用 flatten 后的纯 `MLP projector`。

推荐第一版结构：

- token map 恢复为 `H x W x C`
- `1x1 conv`
- `depthwise 3x3 conv`
- `norm`
- `1x1 conv`
- residual

这条 projector 的目标是：

- 保留局部邻域
- 保持 token grid 拓扑
- 减少“空间结构在映射过程中被洗掉”的风险

### 5.3 模块三：Timestep-layer router

第一版不要做“所有 source token 的全连接 cross-attention”。

推荐轻量 gate：

- 输入：
  - timestep embedding
  - 当前 block id
  - 当前 DiT block hidden 的 pooled summary
- 输出：
  - 对候选 `source-layer` 的 soft weights
- 推荐：
  - `top-k = 1` 或 `2`

这样有三个好处：

- FLOPs 可控
- 归因更清楚
- 更容易可视化

### 5.4 模块四：Stage controller

它属于训练策略，不属于推理图。

第一版主文默认使用：

- 前 `10%` 训练 steps 强制开启 alignment
- 之后按固定 schedule 衰减到 `0`

`adaptive stop` 的建议：

- 放附录
- 或作为第二阶段增强版

原因：

- 直接用 `FID` 或验证指标触发停止，容易被质疑是在拿评测调训练

### 5.5 模块五：Tap blocks

不要每一层都接 alignment。

建议：

- 在中间层选择 `4~6` 个 tap blocks
- 每个 tap block 配一个 projector head
- 由 router 选择要对齐的 source-layer

## 6. Source 选择策略

### 6.1 我的建议

如果问“主线外部 source 该不该优先用 `DINOv3`”，我的判断是：

- **工程主线**：可以优先用 `DINOv3`
- **科学对照**：必须保留 `DINOv2`

### 6.2 为什么不能只保留 DINOv3

因为：

- `REPA` 的直接 teacher 是 `DINOv2`
- 如果没有 `DINOv2` 对照，审稿人会问：
  - 你的增益到底来自方法，还是来自更强 teacher

### 6.3 为什么 DINOv3 作为工程主线更合理

原因有两类。

第一类是表示层面：

- `DINOv3` 更偏高质量 dense features
- 对你当前“空间结构保真”主线更契合

第二类是工程层面：

- 官方 `DINOv3` repo 默认图像 transform 使用 `Resize(256, interpolation=BICUBIC)`，并提供 `ViT-S/B/L/16`
- `256x256` 与 patch size `16` 天然对齐

### 6.4 DINOv2 的真实输入约束

你的担心里有一半对，一半不对。

对的部分：

- 常见 `DINOv2-ViT-L/14` 的 patch size 是 `14`
- `256` 不是 `14` 的倍数

不对的部分：

- 这不等于“必须强行改成 `224x224`”

官方 `DINOv2` 模型卡的表述是：

- 模型可接受任意更大的图像，只要其尺寸是 patch size 的倍数
- 如果输入尺寸不是 patch size 的倍数，模型会 crop 到最近的更小倍数

因此，`DINOv2` 的实际工程选项是：

- 显式 resize 到 `252`
- 显式 resize 到 `280`
- pad/crop 到 patch-compatible resolution

而不是只能 `224`

### 6.5 最终建议

建议把 source 策略写成：

- **MVP 默认 external source**：`DINOv3`
- **必做对照 source**：`DINOv2`
- **关键反证 source**：`self-alignment`
- **补充 source**：`MAE`、`SigLIP`

## 7. 推荐实验矩阵

### 7.1 矩阵 A：与原版 REPA 的链式直接对照

必须存在，且是主文最重要的一组。

设置：

- `No alignment`
- `Original REPA = static MLP + fixed layer + full-course`
- `Original REPA + spatial projector`
- `+ routing`
- `+ fixed early stop`
- 可选附录：`+ adaptive stop`

目的：

- 回答你的增益到底来自什么
- 避免被审成“只是 alignment 更重了”

### 7.2 矩阵 B：空间结构假设验证

用同一个 source，只改表征形式：

- global pooled
- intact spatial map
- patch-shuffled spatial map
- token-permuted spatial map
- spatial map + spatial projector

目的：

- 直接验证“空间结构被破坏后性能是否下降”
- 这是对 `iREPA` 最关键的回应

### 7.3 矩阵 C：stage-wise schedule

设置：

- full-course
- early `10%`
- early `25%`
- early `50%`
- cosine-to-zero
- 附录：adaptive stop

关键指标：

- `FID`
- `sFID`
- `Recall`
- `wall-clock to target FID`
- `throughput`

### 7.4 矩阵 D：source pilot

先只在 `DiT-B/2` 上跑：

- `DINOv2`
- `DINOv3`
- `MAE`
- `SigLIP`
- `self-alignment`

目的：

- 判断 external teacher 是否必要
- 判断“强语义”是否真的更好
- 给主文筛 shortlist

### 7.5 矩阵 E：routing 解释实验

设置：

- fixed shallow
- fixed middle
- fixed deep
- layer-only routing
- timestep-only routing
- timestep + layer routing
- 可选：top-k sparse routing

输出：

- routing heatmap
- layer/source 选择统计
- 与 `FID / Recall` 的关系

### 7.6 矩阵 F：公平主表

主表只放同协议、尽量完整复现的方法：

- `DiT`
- `REPA`
- `HASTE` 或 `REED` 至少一个
- Ours
- 可选：`SD-DiT`

不建议直接混入主表：

- `SANA-1.5`
- `RAE`
- 其他不同任务或不同 latent 范式工作

## 8. 指标与可视化

### 8.1 主指标

- `FID`
- `sFID`
- `Precision`
- `Recall`
- `training FLOPs`
- `throughput`
- `wall-clock to target FID`
- `GPU hours`
- `3` 个 seed 的均值与标准差

### 8.2 必须有的图

- routing heatmap
- gradient/loss interaction 图
- external vs self 的正反例图
- spatial intact vs shuffled 的对照图

## 9. 实验顺序建议

不要一开始就做“大而全”版本。

建议顺序：

1. `DiT-B/2 + No alignment`
2. `DiT-B/2 + Original REPA`
3. 加 `spatial projector`
4. 加 `fixed early stop`
5. 加 `routing`
6. 跑 source pilot
7. 选 shortlist
8. 上 `DiT-XL/2` 主表

这个顺序不能反。

## 10. 风险与转向规则

### 10.1 如果 self-alignment 最优

论文主线切换为：

**external teacher 是否必要**

### 10.2 如果 spatial projector 收益不稳定

论文主线切换为：

**stage-wise termination 是主要贡献，spatial module 退到辅助消融**

### 10.3 如果 routing 收益有限

论文主线切换为：

**空间结构保真 + 分阶段终止是必要条件，routing 只是弱增强**

### 10.4 如果 DINOv3 只是在工程上更方便

论文写法必须避免：

- “DINOv3 更强所以更好”

应该写成：

- `DINOv3` 作为主线 source 的原因是
  - 与 `256x256` patch compatibility 更自然
  - dense feature 更适合 spatial alignment
- 同时保留 `DINOv2` 作为 `REPA` continuity baseline

## 11. 最小可行版本

建议的 MVP：

- Backbone：`DiT-B/2`
- Sources：
  - `DINOv3`
  - `DINOv2`
  - `self-alignment`
- Main recipe：
  - spatial projector
  - fixed early stop
  - lightweight routing
- Main controls：
  - no alignment
  - original REPA
  - spatial intact vs shuffled

如果这版都站不住，就不要直接上 `DiT-XL/2`。

## 12. 关键来源

- REPA: https://openreview.net/forum?id=DJSZGGZYVi
- HASTE: https://openreview.net/forum?id=HK96GI5s7G
- REED: https://openreview.net/forum?id=cIGfKdfy3N
- REG: https://openreview.net/forum?id=koEALFNBj1
- iREPA: https://openreview.net/forum?id=y0UxFtXqXf
- DUPA: https://openreview.net/forum?id=ALpn1nQj5R
- RAE: https://openreview.net/forum?id=0u1LigJaab
- SRA: https://arxiv.org/abs/2505.02831
- SD-DiT: https://cvpr.thecvf.com/virtual/2024/poster/30284
- DINOv2 model card: https://github.com/facebookresearch/dinov2/blob/main/MODEL_CARD.md
- DINOv3 repo: https://github.com/facebookresearch/dinov3
