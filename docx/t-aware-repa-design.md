# T-Aware REPA 设计方案

## 1. 目标定义

本文档将当前项目的主线收敛为：

**按扩散时间步 `t` 动态调节表征对齐强度的 REPA 变体，即 `t-aware REPA`。**

这里的“时间步”明确指 **diffusion timestep**，不是训练 step。

我们要验证的核心命题是：

- 在逆扩散的高噪声阶段，大 `t` 的 DiT 更需要外部语义监督来稳定全局语义和布局学习。
- 在低噪声阶段，小 `t` 的 DiT 更应专注于细节恢复，外部语义对齐若持续过强，反而可能形成约束。

因此，alignment 不应只依赖训练进度，而应显式依赖样本自己的噪声阶段。


## 2. 现有代码与目标设计的差异

当前实现仍是：

- `train_2.py` 中通过 `get_alignment_weight(step, args)` 按训练 step 生成全局权重。
- 然后用
  `loss = loss_diff + align_weight * loss_align`
  作为总损失。
- `t` 只参与加噪和 DiT 前向，不参与 alignment 强度调度。

这意味着当前实现本质上是：

**REPA-style alignment + training-step linear decay**

而不是：

**t-aware semantic alignment**

对应代码位置：

- 当前训练步调度：[train_2.py](/home/liuchunfa/2026qjx/2026test/train_2.py#L266)
- 当前 loss 汇总：[train_2.py](/home/liuchunfa/2026qjx/2026test/train_2.py#L643)
- DiT 的 `t` 条件入口：[models_2.py](/home/liuchunfa/2026qjx/2026test/models_2.py#L242)
- DiT 的 `c = t + y` 条件化：[models_2.py](/home/liuchunfa/2026qjx/2026test/models_2.py#L267)


## 3. 顶会文献给出的直接启发

### 3.1 REPA, ICLR 2025 Oral

链接：

- https://openreview.net/forum?id=DJSZGGZYVi

直接启发：

- 外部表征对齐确实能显著加速 DiT/SiT 训练。
- 对齐对象是 noisy hidden state 与 clean teacher representation。
- 但原版 REPA 使用的是静态 projector 和固定对齐系数，没有显式建模不同 diffusion timestep 的差异。

对我们意味着：

- `REPA` 证明“对齐有用”。
- 但没有回答“应该在什么噪声阶段对齐得更强”。


### 3.2 HASTE, NeurIPS 2025 Poster

链接：

- https://openreview.net/forum?id=HK96GI5s7G

直接启发：

- alignment 在训练早期帮助大，后期可能变成束缚。
- 全程静态对齐不是最优。
- 两阶段 termination 是有效的。

对我们意味着：

- reviewer 对“全程统一 alignment”提出质疑是有文献依据的。
- 但 HASTE 主要讨论的是 **training phase** 维度，不是 **diffusion timestep** 维度。

因此我们不应简单重复 HASTE，而应把创新点明确落在：

**同一训练阶段内，不同 `t` 是否应当有不同语义对齐强度。**


### 3.3 iREPA, ICLR 2026 Poster

链接：

- https://openreview.net/forum?id=y0UxFtXqXf

直接启发：

- 决定 alignment 有效性的关键不是更强的全局语义，而是更好的空间结构保真。
- 简单卷积 projector 和 spatial normalization 就能显著改善 REPA。

对我们意味着：

- “换更强 teacher”不是核心故事。
- 如果要做 `t-aware REPA`，projector 最多只能作为支撑项，不能再当 headline contribution。


### 3.4 DUPA, ICLR 2026 Poster

链接：

- https://openreview.net/forum?id=ALpn1nQj5R

直接启发：

- REPA 主要帮助早层获取稳健语义。
- 不依赖外部 teacher 的 self-alignment 也可以工作。

对我们意味着：

- “高噪声阶段需要更强语义条件”这个方向是有支撑的。
- 同时也提醒我们后续必须做 external teacher 与 self-alignment 的比较，避免被 reviewer 质疑“为什么一定要 DINO”。


### 3.5 SRA, ICLR 2026 Poster

链接：

- https://openreview.net/forum?id=ds5w2xth93

直接启发：

- SRA 直接对齐“高噪声条件下的早层表示”和“低噪声条件下的晚层表示”。
- 它说明 DiT 内部的表征质量确实具有明确的噪声阶段差异。

对我们意味着：

- `t` 不是一个可有可无的超参数，而是表征学习机制本身的一部分。
- 用 per-sample `g(t)` 控制 alignment 是合理且有顶会邻近工作的。


### 3.6 DiffCR, CVPR 2025

链接：

- https://openaccess.thecvf.com/content/CVPR2025/html/You_Layer-_and_Timestep-Adaptive_Differentiable_Token_Compression_Ratios_for_Efficient_Diffusion_CVPR_2025_paper.html

直接启发：

- 不同 denoising timestep 的最优计算模式不同。
- 文中观察到更 noisy 的 timestep 和更 clear 的 timestep 需要不同的处理强度。

对我们意味着：

- “不同 `t` 不该被一视同仁”不仅体现在 alignment，也体现在 DiT 的计算资源分配中。
- 这为我们引入 `t` 维度的动态 alignment 提供了来自 CVPR 的旁证。


### 3.7 AsynDM, ICLR 2026 Poster

链接：

- https://openreview.net/forum?id=ZHb4bduWkM

直接启发：

- 生成对齐能力可以通过动态 timestep schedule 改善。
- 不同区域或不同内容在 denoising 中并不需要统一节奏。

对我们意味着：

- “固定同步的 timestep 使用方式并不一定最优”已经被顶会工作挑战。
- 虽然它是 inference-time 设计，不是我们的直接方法，但能强化“时间步动态化是合理研究方向”。


## 4. 文献收敛后的方法定位

基于以上文献，最稳妥、最可 defend 的项目主线应当是：

### 主创新

**在 REPA 框架中引入 per-sample diffusion-timestep-aware alignment gating。**

### 关键支撑

- spatially faithful projector
- 与 diffusion loss 共享同一个 `t` 和同一个 `x_t`

### 可选增强

- training-phase stop 或早停
- self-alignment source
- lightweight routing

不应再把下面这些写成主创新：

- “更强的 DINOv3 teacher”
- “复杂 routing 本身”
- “训练 step 线性衰减”


## 5. 方法定义

### 5.1 基本符号

- `x_0`: VAE latent
- `t`: diffusion timestep
- `x_t`: 对 `x_0` 按 `t` 加噪后的 latent
- `h_l(x_t, t, y)`: DiT 在第 `l` 个 tap layer 的 token 表示
- `s(x_img)`: DINO teacher 提取的 clean image patch tokens
- `P(.)`: projector
- `g(t)`: 根据 diffusion timestep 生成的 alignment gate


### 5.2 推荐的目标函数

对 batch 内第 `i` 个样本：

```math
L_i = L_{diff,i} + \lambda \cdot g(t_i) \cdot L_{align,i}
```

其中：

```math
L_{align,i} =
1 - \frac{1}{N}\sum_{j=1}^{N}
\cos(P(h_{l,i,j}), s_{i,j})
```

总损失为：

```math
L = \frac{1}{B} \sum_{i=1}^{B} L_i
```

这里最关键的变化是：

- `g(t_i)` 是 **每个样本自己的权重**
- 不是整个 batch 共用一个训练 step 权重


### 5.3 `g(t)` 的推荐形式

第一阶段建议只做 3 个简单版本：

1. `constant`
   - `g(t) = 1`
   - 作为 REPA-style baseline

2. `linear-high-noise`
   - 大 `t` 权重大，小 `t` 权重小
   - 用于直接验证“高噪声更需要语义”

3. `threshold-high-noise`
   - 当 `t >= t0` 时 `g(t)=1`，否则 `0`
   - 最容易解释，也最容易和 reviewer 对齐

若后续效果明确，再加：

4. `logSNR/cosine`
   - 按噪声水平而不是原始整数 `t` 建模
   - 物理意义更强，但第一阶段不是必须


## 6. 这个方案如何把 DINO 语义“注入”到 DiT

这里的“注入”不是推理时把 DINO feature 作为外部输入喂给 DiT。

它的真实机制是：

1. teacher 支路从 clean RGB 图像中提取 DINO patch tokens。
2. student 支路从 `x_t` 中提取 DiT 中间 token。
3. 通过 `P(h_l)` 与 DINO tokens 的对齐损失，把语义梯度反传回 DiT。
4. 因为每个样本的对齐强度由 `g(t_i)` 控制，所以大 `t` 样本获得更强的语义约束，小 `t` 样本获得更弱的语义约束。

因此模型学到的是：

- 高噪声阶段内部表示更偏语义和布局
- 低噪声阶段内部表示更偏细节恢复

这和“在推理时输入 DINO”是两回事。

推理阶段依然保持原始 DiT 复杂度，不需要真实图像，也不需要额外 DINO 前向。


## 7. 为什么这个设计比 training-step 衰减更合理

training-step 衰减回答的是：

- 在训练前期和后期，alignment 要不要变弱

但它没有回答：

- 在同一个训练时刻里，batch 内不同噪声水平的样本，是否应接受不同强度的语义约束

而 DiT 的真实 denoising 任务恰恰是按 `t` 区分的。

因此：

- `step-aware` 是优化过程层面的调度
- `t-aware` 是建模机制层面的调度

对于你们当前的主问题，“早期 denoising 更需要语义，后期 denoising 更需要细节”，后者才是直接对应的变量。


## 8. 对当前代码的最小改动路径

这里只定义设计，不直接改代码。

### 8.1 第一阶段：最小可验证版本

只做以下 4 个变化：

1. 删除基于训练 step 的 `align_weight`
2. 增加 `get_diffusion_weight(t, num_timesteps, schedule)`
3. 将 token cosine loss 改成 per-sample 版本
4. 让 diffusion loss 和 alignment loss 共用同一个 `t` 与同一个 `noise`

这版不改 DiT 主干，不加 routing，不加多 source。


### 8.2 第二阶段：支撑项增强

在第一阶段验证有效后再加：

1. spatial projector
2. self-alignment 对照
3. HASTE-style training-phase termination


### 8.3 第三阶段：可选扩展

若前两阶段结果稳定，再考虑：

1. timestep-only routing
2. block-wise multi-tap alignment
3. external-to-self source transition


## 9. 推荐实验矩阵

### 9.1 核心主表

1. `No alignment`
2. `Original REPA-style constant alignment`
3. `Current code: training-step linear decay`
4. `T-aware REPA: linear-high-noise`
5. `T-aware REPA: threshold-high-noise`

这里最关键的是让 baseline 链条完整，否则 reviewer 会认为你只是换了个 schedule 名字。


### 9.2 关键消融

1. `shared noise` vs `independent noise`
2. `MLP projector` vs `spatial projector`
3. `DINO teacher` vs `self-alignment`
4. `mid-layer tap` vs `late-layer tap`


### 9.3 机制分析

建议至少画 3 类图：

1. `g(t)` 曲线图
2. 不同 `t` bin 下的 alignment loss 曲线
3. 不同 checkpoint 下的 FID 趋势图

如果资源允许，可以额外做：

4. 不同 `t` 下的 token cosine similarity 分布
5. 不同 `t` 下的 sample visualization


## 10. reviewer 可能的质疑与应对

### 10.1 “这不就是 HASTE 吗”

应对：

- 不是。
- HASTE 主要建模的是 training phase 的早停。
- 我们的主变量是 diffusion timestep `t`，是 per-sample noise-level control。


### 10.2 “DINO 在推理时不存在，那你怎么叫 injection”

应对：

- DINO 是训练时 teacher，不是推理时输入。
- injection 发生在训练期的 representation shaping，而不是 inference-time conditioning。


### 10.3 “这是不是只是换了个 loss weight”

应对：

- 不是简单换权重。
- 权重不再依赖训练进度，而是依赖样本的噪声阶段。
- 这直接改变了 DiT 在不同 denoising 阶段所接收的语义约束模式。


### 10.4 “为什么一定要 DINO”

应对：

- 不应预设 DINO 必然最佳。
- 必须加入 self-alignment 或其他 source 对照，尤其参考 DUPA 与 SRA 的发现。


## 11. 最终建议

结合当前代码成熟度与 reviewer 风险，建议你们按以下顺序推进：

1. 先把当前方法正式降级表述为
   `REPA-style baseline with training-step decay`
2. 再实现最小版 `t-aware REPA`
3. 只在 `t-aware REPA` 有稳定收益后，再引入 spatial projector
4. 最后才考虑 routing 或 source transition

这样做的好处是：

- 论文主线更集中
- 和 REPA / HASTE / iREPA / SRA / DUPA 的关系更清楚
- 更容易向 reviewer 证明你们的核心贡献不是“缝组件”，而是：

**alignment 强度应显式依赖 diffusion timestep，而不是只依赖训练阶段。**


## 12. 参考文献链接

- REPA, ICLR 2025 Oral: https://openreview.net/forum?id=DJSZGGZYVi
- HASTE, NeurIPS 2025 Poster: https://openreview.net/forum?id=HK96GI5s7G
- iREPA, ICLR 2026 Poster: https://openreview.net/forum?id=y0UxFtXqXf
- DUPA, ICLR 2026 Poster: https://openreview.net/forum?id=ALpn1nQj5R
- SRA, ICLR 2026 Poster: https://openreview.net/forum?id=ds5w2xth93
- DiffCR, CVPR 2025: https://openaccess.thecvf.com/content/CVPR2025/html/You_Layer-_and_Timestep-Adaptive_Differentiable_Token_Compression_Ratios_for_Efficient_Diffusion_CVPR_2025_paper.html
- AsynDM, ICLR 2026 Poster: https://openreview.net/forum?id=ZHb4bduWkM
