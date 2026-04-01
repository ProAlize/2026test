# 论文结构规划：从 Ada-REPA 到可验证的 DiT 表征对齐研究

## 1. 题目方向

### 主标题候选

- **Stage-Wise Spatial Representation Alignment for Diffusion Transformers**
- **When and What to Align: Spatial, Layer-wise, and Stage-wise Representation Guidance for DiT**

### 说明

`Ada-REPA` 可以继续作为内部项目代号，但不建议直接作为论文主标题。标题应突出：

- 研究问题
- 机制
- 适用边界

而不是内部命名。

## 2. 论文核心问题

本文不再把主问题定义为“如何用更强 teacher 提升 DiT”，而是定义为：

**外部表征对 DiT 训练真正有用的是什么，在哪些 `timestep / layer / training phase` 有用，以及何时应该停止对齐。**

这一定义有三个好处：

- 问题更清晰
- 更容易形成机制性实验
- 更符合 2025-2026 文献对 `REPA` 路线的继续追问

### 2.1 主创新层级

为避免论文被审成“多个合理组件的组合”，本文的创新层级需要明确写成：

- **主创新主张**
  表征对齐的关键不在于引入更强 teacher，而在于让对齐满足两个条件：`spatially preserved` 与 `stage-aware`。
- **方法实例化**
  `Spatial-preserving projector` 与 `Stage-wise termination` 是对上述主张的核心实例化。
- **辅助机制**
  `Timestep-layer routing` 用于增强上述主张的可实现性与解释性。
- **实验框架，不作为主创新**
  `Source modularity` 主要用于公平比较 external teacher 与 self-alignment，不应在摘要中写成独立方法贡献。

## 3. 摘要改写草案

### 背景

近期工作表明，representation alignment 可以显著改善 Diffusion Transformer 的训练效率与生成质量。然而，现有方法通常依赖静态、全局语义主导的外部 teacher，并在整个训练过程中持续施加 alignment 约束。

### 研究缺口

这一设定留下了三个未被充分回答的问题：

1. 对 DiT 训练真正有效的，究竟是 teacher 的全局语义，还是其保留的空间结构。
2. 表征注入应发生在哪些 `timestep` 与 `layer`。
3. alignment 是否只在训练早期有效，而在中后期反而成为过约束。

### 方法概述

为回答这些问题，我们提出一个围绕“空间结构保真 + 阶段性感知”构建的 DiT 表征对齐框架。该框架包含：

- `Spatial-preserving projector`
- `Timestep-layer routing`
- `Stage-wise termination`
- `source modularity`

其中前两项分别对应本文的主创新原则与关键训练机制，`source modularity` 仅作为统一实验接口。我们在统一的 `ImageNet-256 class-conditional` 协议下，对 external teacher 与 self-alignment 进行公平比较，并系统分析不同 source、不同 layer、不同 timestep 与不同训练阶段的作用差异。

### 预期贡献

- 给出比“更强 teacher 更好”更细粒度的机制性结论
- 说明空间结构与阶段性注入在 DiT 表征对齐中的作用
- 为 `REPA` 及其后续路线提供统一、可复现的比较框架

## 4. 引言结构

### 4.1 第一段：问题背景

- DiT 与 flow-transformer 在生成建模中的快速发展
- 训练效率与生成质量之间的长期张力
- representation guidance 成为加速与稳训的重要路线

### 4.2 第二段：现有工作的不足

- 现有方法多采用静态、全程的 feature alignment
- 对“何时对齐、对齐什么、何时停止”缺乏统一结论
- 不同工作对 external teacher 的必要性给出了相互张力很强的答案

### 4.3 第三段：本文要解决的问题

- 空间结构是否比全局语义更关键
- 对齐收益是否集中在训练早期
- 不同 source 是否对应不同的 `timestep / layer` 使用模式

### 4.4 第四段：本文贡献

建议将贡献压缩为以下三条：

1. 提出一个以空间结构保真、动态层时刻路由和阶段性终止为核心的 DiT 表征对齐框架。
2. 在统一协议下系统比较多种 external teacher 与 self-alignment，澄清何种表征最有效。
3. 提供一组机制性实验，解释表征对齐何时有用、何时失效，以及为什么失效。

## 5. 相关工作

### 5.1 DiT 与生成 backbone 的效率演进

建议覆盖：

- `DiT`
- `SiT`
- `Scaling Rectified Flow Transformers`
- `TinyFusion`
- `FlexiDiT`
- `EDiT`
- `SANA-1.5`

写作原则：

- 区分训练效率、推理效率、架构压缩、线性注意力等不同路线
- 不将不同任务协议的结果混写成同一张比较表

### 5.2 Representation alignment 与 guidance

建议覆盖：

- `SD-DiT`
- `REPA`
- `REG`
- `HASTE`
- `REED`

写作重点：

- 从“是否有效”转向“为什么有效、何时失效”

### 5.3 Self-alignment 与 teacher-free 路线

建议覆盖：

- `SRA`
- `DUPA`

写作重点：

- external teacher 并不是唯一选择
- 需要将其放进统一协议下公平比较

### 5.4 表征属性与 latent 替代路线

建议覆盖：

- `iREPA`
- `RAE`
- 可选补充：`DINOv3`

写作重点：

- 空间结构与 latent 质量正在成为替代“更强 teacher”的关键因素

## 6. 方法章节

### 6.1 Problem Setup

- 定义 `ImageNet-256 class-conditional` 主设定
- 定义 DiT 主干
- 定义 source feature、projector、alignment loss、schedule
- 明确所有 external encoder 默认冻结
- 明确 `self-alignment` 采用 DiT 的 `EMA` 副本作为 source network
- 明确主训练目标：
  `L = L_diff + lambda(t_train) * L_align`
- 明确默认对齐损失：
  `L_align = E_(l,k ~ r)[1 - cos(P_l(h_l), z_k)]`
  其中 `r` 为 layer-source routing 分配，`P_l` 为对应层的 projector

### 6.2 Spatial-Preserving Projector

核心表述：

- 现有纯 `MLP projector` 会在映射过程中弱化空间结构
- 我们使用保留局部结构的 projector，使 representation transfer 更适配生成任务

### 6.3 Timestep-Layer Routing

核心表述：

- 不同 `timestep` 对 source feature 的需求不同
- 不同 DiT 层位对 feature granularity 的需求不同
- 因此采用显式 routing，而不是固定层对层匹配
- routing 是核心思想的实现机制，而不是独立于“空间保真 + 阶段性感知”的新问题

### 6.4 Stage-Wise Termination

核心表述：

- alignment 的作用不是全程单调增加
- 在训练中后期，过强对齐可能限制生成模型自身分布建模
- 因此引入固定或自适应的 early-stop 机制
- 默认自适应规则应写清楚：
  前 `10%` 训练阶段强制开启 alignment，之后每 `10K` steps 用 `FID-10k` 代理指标做 patience 检测，连续 `3` 次无改善则永久关闭 alignment

### 6.5 Source Modularity

核心表述：

- 不预设某个 teacher 最优
- 将 `DINOv2 / DINOv3 / SigLIP / MAE / self-alignment` 放入统一接口下比较
- `Source modularity` 的作用是构造公平实验，而不是把“更多 source”写成方法贡献

### 6.6 Main Hypotheses

方法章节末尾应明确写出待检验假设，而不是直接写结论：

1. 空间结构保真优于单纯的全局语义强化。
2. 对齐收益集中在训练早期。
3. source 的有效性依赖于 `timestep / layer`。

## 7. 实验章节

实验章节不再围绕“百团大战式对打”组织，而围绕三类问题组织：

在写法上还应增加一条原则：

- 所有 headline claim 都必须由 `reproduced` baseline 支撑

### 7.1 组件是否真的有效

- projector
- routing
- stage-wise termination
- 原版 `REPA` 直接对照

### 7.2 什么 source 最有效

- external teachers
- self-alignment

### 7.3 为什么有效

- 梯度交互分析
- 路由热力图
- failure cases
- `FID / Recall / throughput / wall-clock` 联合分析

### 7.4 与强 baseline 的公平比较

只保留统一协议下的 baseline 进入主表。

主表必须额外说明：

- 哪些 baseline 为本工作完整复现
- 哪些结果仅作补充引用
- 若某方法未复现，则不能用于摘要 headline claim

## 8. 结论章节

结论应避免写成“我们全面超过所有新架构”，而应写成：

- 哪种表征属性对 DiT 最重要
- 对齐在哪些阶段帮助最大
- external teacher 与 self-alignment 的边界
- 本方法在哪些条件下有效，在哪些条件下无明显收益

还应明确加上一句作用域限制：

- 本文主结论首先成立于 `ImageNet-256 class-conditional DiT` 设定，除非扩展实验支持，否则不将其表述为跨任务普适规律

## 9. 论文写作原则

### 9.1 明确禁止的写法

- “屠榜”
- “镇压”
- “降维打击”
- “霸主”
- 任何未经验证的绝对化判断

### 9.2 建议采用的写法

- “we hypothesize”
- “we observe”
- “under a unified protocol”
- “we find that”
- “the gain concentrates in the early stage”

### 9.3 贡献书写原则

- 每条贡献都必须能在实验中被明确指向
- 每条贡献都必须可证伪
- 贡献数量控制在 `3` 条左右，不堆砌名词

## 10. 参考文献更新清单

### 严格 CCF A 主线

1. `SD-DiT`, CVPR 2024
2. `Scaling Rectified Flow Transformers for High-Resolution Image Synthesis`, ICML 2024
3. `TinyFusion`, CVPR 2025
4. `FlexiDiT`, CVPR 2025
5. `SANA-1.5`, ICML 2025
6. `EDiT`, ICCV 2025
7. `REG`, NeurIPS 2025
8. `HASTE`, NeurIPS 2025
9. `REED`, NeurIPS 2025

### 顶级但非严格 CCF A 的关键补充

1. `SiT`, ECCV 2024
2. `REPA`, ICLR 2025
3. `SRA`, arXiv 2025
4. `DINOv3`, arXiv 2025
5. `iREPA`, ICLR 2026
6. `DUPA`, ICLR 2026
7. `RAE`, ICLR 2026

## 11. 当前版本的论文定位

修改后的论文不再定位为“用双 teacher 打败一切”的方案论文，而定位为：

**一篇研究 DiT 表征对齐机制与边界条件的论文。**

只有在这个定位下，方法、实验和叙事才会统一，也才更接近真正的顶会强稿标准。

### 11.1 负结果转向预案

如果实验结果不支持当前优先假设，论文定位应按以下规则转向：

- 若 `self-alignment` 优于全部 external teacher，则论文主问题改写为“external teacher 是否必要”。
- 若 `Stage-wise termination` 带来主要收益，而 projector 与 routing 收益有限，则论文主问题改写为“何时停止 alignment 比对齐什么更重要”。
- 若 `Spatial-preserving projector` 带来主要收益，而 routing 收益有限，则论文主问题改写为“空间结构保真是表征对齐生效的关键条件”。

这部分需要在投稿前同步反映到标题、摘要、贡献和实验顺序中。
