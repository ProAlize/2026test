# Ada-REPA 项目总审、文献调研与重构建议

## 文档职责（审计快照，非执行真值源）

- 文件角色：`阶段性审计/复盘` 与 `文献趋势快照`。
- 负责范围：问题诊断、风险评估、方向性建议、文献分层参考。
- 不负责范围：当前实验协议、当前架构参数、当前排期执行。
- 使用方式：本文件用于“解释为什么要改”，不用于“直接作为配置执行”。
- 冲突仲裁：与 `plan.md`、`exp.md`、`architecture-gpt.md` 冲突时，以后三者为准。
- 统一入口：见 `doc-roles.md`。

截至 `2026-03-31`，我已完整阅读当前目录下的全部文件：`plan.md`、`exp.md`、`paper.md`，并结合子代理审计与联网检索，对项目现状、相关工作、主要风险和可行升级路线做出如下总结。

## 1. 一句话结论

当前仓库不是一个接近投稿完成态的项目，而是一个研究提案包。它的问题意识是对的，但事实核验、对标协议、方法新意和可复现性都还没有达到 CVPR 主会强稿，更不可能仅凭现状达到 CVPR Best Paper 标准。

更直接地说：

- 这个项目目前的最大优点是抓住了 `DiT 训练效率 + 表征引导` 这一真实热点。
- 这个项目目前的最大缺点是把很多尚未被证明的判断写成了既定事实，并且混用了不同任务、不同协议、不同发表状态的 baseline。

## 2. 我审计到的当前项目优点

- 选题方向正确。`REPA`、`DiT/SiT`、外部表征引导、训练加速、多样性与真实度权衡，都是 2024-2026 的真实前沿问题。
- 你已经意识到单纯追 `FID` 不够，尝试引入梯度余弦、路由热力图、`Recall/Diversity` 之类的解释性证据，这一点优于很多只会堆分数的 proposal。
- 三份文档之间形成了初步的“时间线-实验矩阵-论文骨架”联动关系，说明你不是零散 brainstorm，而是在构造一篇论文。
- 你提出的 `time-conditioned projector`、`cross-layer soft matching`、多 teacher 对比，本身都对应真实痛点，而不是完全脱离文献的空想。

## 3. 当前项目的关键缺点

### 3.1 现在只有 proposal，没有项目实体

当前目录里没有以下任何一项：

- 训练代码
- 模型定义
- 配置文件
- 数据处理脚本
- 实验日志
- 表格与图
- 复现实验命令
- checkpoint 或结果摘要

所以当前状态离“可投稿论文”至少还差一个完整的实现与验证层。

### 3.2 文献归属和会议级别存在错误

经核验，当前文档中至少有这些问题：

- `REPA` 不是 `NeurIPS 2024`，而是 `ICLR 2025 Oral`。
- `SiT` 不是 `ICLR 2024`，而是 `ECCV 2024`。
- 严格按 CCF 官方目录，`ICLR` 和 `ECCV` 都不属于 `CCF A`，因此如果你要写“2024 到现在所有 CCF A 相关论文”，必须把它们放在“顶级但非严格 CCF A 补充”层，而不能混写成同一层。

### 3.3 baseline 不公平

你当前实验主设定是 `ImageNet-256 class-conditional`，但矩阵三把以下工作混进了同一张“对打表”：

- `SANA-1.5`
- `EasyControl`
- 若干未充分核验的 `arXiv` 预印本

问题在于：

- 一部分是 `text-to-image`
- 一部分是控制/编辑
- 一部分不是同一训练协议
- 一部分不是同一评测基准
- 一部分不是稳定的同行评审结果

这种表在正式投稿里会被认为比较对象不兼容。

### 3.4 当前主张还不够“可证伪”

文档里多次把以下命题写成几乎既定结论：

- `DINOv3` 会导致 mode collapse
- `JEPA` 是扩散模型的天然盟友
- 双引擎一定“全面屠榜”

这些都属于高风险断言，因为当前仓库里没有先导实验、没有统计设计、没有反例分析、没有失败案例，也没有对照排除混杂因素。

### 3.5 方法新意正在被更强更新工作逼近

截至 `2026-03-31`，相关工作已经把这条线往前推进到以下方向：

- 不只是“是否要对齐表征”，而是“什么时候对齐、对齐什么、何时停止”
- 不只是“语义强不强”，而是“空间结构是否更关键”
- 不只是“外部 teacher”，而是“自对齐是否足够”
- 甚至进一步演化为“直接在高质量语义 latent 中训练，不再需要额外 alignment loss”

在这个背景下，单纯的 `AdaLN projector + soft matching + 双 teacher`，不足以自动构成强 novelty。

### 3.6 写作风格不符合顶会主文风格

文档中大量存在：

- “屠榜”
- “镇压”
- “降维打击”
- “霸主”

这类措辞在内部动员可以理解，但不适合论文写作。顶会强稿的语言应该是：

- 提出一个明确问题
- 提出可验证假设
- 给出方法
- 证明边界条件
- 报告失败模式

## 4. 2024 到现在的相关论文调研

### 4.1 检索范围说明

为了避免“所有相关论文”这个目标无限扩张，我将“相关方向”限定为以下四类：

- `Diffusion Transformer / flow-transformer` 的主干演进与效率改进
- `representation alignment / guidance` 用于扩散训练加速与生成质量提升
- `self-alignment / external-teacher-free` 的替代路线
- `representation-rich latent / representation autoencoder` 对外部对齐路线的替代

### 4.2 严格 CCF A 主线文献

严格 CCF A 会议名单采用 CCF 官方人工智能目录，A 类包括：`AAAI / NeurIPS / ACL / CVPR / ICCV / ICML / IJCAI`。这里与本项目直接相关的主线工作如下。

| 年份 | 会议 | 论文 | 方向 | 对本项目的启示 |
| --- | --- | --- | --- | --- |
| 2024 | CVPR | [SD-DiT: Unleashing the Power of Self-supervised Discrimination in Diffusion Transformer](https://cvpr.thecvf.com/virtual/2024/poster/30284) | 用自监督判别信号改善 DiT 训练 | 说明“生成模型内部判别表征”本身就是一个真实方向，不必只盯外部 teacher。 |
| 2024 | ICML | [Scaling Rectified Flow Transformers for High-Resolution Image Synthesis](https://proceedings.mlr.press/v235/esser24a.html) | Rectified Flow Transformer 与大规模生成 | 说明 2024 起顶会主线已经强烈关注 transformer-based generative backbone 本身的扩展与 scaling。 |
| 2025 | CVPR | [TinyFusion: Diffusion Transformers Learned Shallow](https://openaccess.thecvf.com/content/CVPR2025/html/Fang_TinyFusion_Diffusion_Transformers_Learned_Shallow_CVPR_2025_paper.html) | DiT 深度裁剪与恢复性剪枝 | 你的“效率”叙事若不报告 wall-clock、FLOPs、收敛速度，就会被这类工作压制。 |
| 2025 | CVPR | [FlexiDiT: Your Diffusion Transformer Can Easily Generate High-Quality Samples with Less Compute](https://cvpr.thecvf.com/virtual/2025/poster/34601) | 变计算预算推理 | 说明 DiT 效率研究已经从训练扩展到推理阶段的弹性计算。 |
| 2025 | ICML | [SANA 1.5: Efficient Scaling of Training-Time and Inference-Time Compute in Linear Diffusion Transformer](https://icml.cc/virtual/2025/poster/46604) | 线性化 DiT、训练与推理联合 scaling | 是真实强对标，但它属于 `text-to-image` 路线，不能直接与 ImageNet-256 class-conditional 主表横比。 |
| 2025 | ICCV | [EDiT: Efficient Diffusion Transformers with Linear Compressed Attention](https://openaccess.thecvf.com/content/ICCV2025/html/Becker_EDiT_Efficient_Diffusion_Transformers_with_Linear_Compressed_Attention_ICCV_2025_paper.html) | 线性压缩注意力 | 说明“注意力高效化”是另一条成熟效率路线，你的方法必须说明比它新在哪里。 |
| 2025 | NeurIPS | [Representation Entanglement for Generation: Training Diffusion Transformers Is Much Easier Than You Think](https://openreview.net/forum?id=koEALFNBj1) | 从“对齐”走向“纠缠/内生语义 token” | 表明只做外部 feature alignment 已不是唯一主线。 |
| 2025 | NeurIPS | [REPA Works Until It Doesn't: Early-Stopped, Holistic Alignment Supercharges Diffusion Training](https://openreview.net/forum?id=HK96GI5s7G) | HASTE：早停 + holistic alignment | 直接挑战你现在“全程对齐”的设定，说明对齐帮助只在训练早期最强。 |
| 2025 | NeurIPS | [Learning Diffusion Models with Flexible Representation Guidance](https://openreview.net/forum?id=cIGfKdfy3N) | REED：何时、如何引导的系统化框架 | 说明现在不只是问“有没有 guidance”，而是问“引导的形式、时机和 curriculum”。 |

补充说明：

- 在我本次检索范围内，没有发现 `AAAI` 和 `IJCAI` 主会中与本项目同等贴近、且足以作为主表核心 baseline 的代表性工作。
- 这并不意味着这些会议完全没有 diffusion 论文，而是说在“`DiT/ImageNet-256/representation-guided training`”这个窄定义下，没有检索到足够强的主线代表作。

### 4.3 顶级但非严格 CCF A 的必要补充

这些工作不应混写进“严格 CCF A 主线”，但它们对你的项目判断至关重要。

| 年份 | 会议/来源 | 论文 | 作用 |
| --- | --- | --- | --- |
| 2024 | ECCV | [SiT: Exploring Flow and Diffusion-based Generative Models with Scalable Interpolant Transformers](https://eccv.ecva.net/virtual/2024/poster/690) | 你文档里反复对标的骨干工作；它是真实重要 baseline，但不是严格 CCF A。 |
| 2025 | ICLR Oral | [Representation Alignment for Generation: Training Diffusion Transformers Is Easier Than You Think](https://openreview.net/forum?id=DJSZGGZYVi) | 这是 REPA 的真实出处，也是你项目的直接起点。 |
| 2025 | arXiv | [No Other Representation Component Is Needed: Diffusion Transformers Can Provide Representation Guidance by Themselves](https://arxiv.org/abs/2505.02831) | SRA 路线，指出 DiT 可能自己就能提供 representation guidance。 |
| 2025 | arXiv | [DINOv3](https://arxiv.org/abs/2508.10104) | 这是 teacher 候选，但它不是 CCF A 会议论文。不能把“使用 DINOv3”写成“顺应 CCF A 最新共识”。 |
| 2026 | ICLR Poster | [What matters for Representation Alignment: Global Information or Spatial Structure?](https://openreview.net/forum?id=y0UxFtXqXf) | iREPA：指出空间结构比全局语义更关键，直接冲击你文档中的“DINOv3 语义更强所以更好”叙事。 |
| 2026 | ICLR Poster | [Dual-Path Condition Alignment for Diffusion Transformers](https://openreview.net/forum?id=ALpn1nQj5R) | DUPA：证明不用外部 teacher，也可以做高效自对齐。 |
| 2026 | ICLR Poster | [Diffusion Transformers with Representation Autoencoders](https://openreview.net/forum?id=0u1LigJaab) | RAE：直接在语义更强的 latent 中训练，甚至绕过额外 alignment loss。 |

### 4.4 这条文献线现在告诉了我们什么

从 2024 到 2026，这个方向的大趋势很清楚：

1. 2024：大家在问，DiT 还能不能更快、更稳、更可扩展。
2. 2025：大家发现外部表征引导确实有效，但开始追问它为什么有效、什么时候失效。
3. 2025 下半年到 2026：大家进一步发现，真正重要的可能不是“teacher 更强的全局语义”，而是“空间结构、阶段性注入、自对齐能力，以及更好的 latent 表示”。

这意味着你当前方案最需要避免的误区是：

- 不要把“更多 teacher、语义更强”误写成这条线的必然未来。

## 5. 当前项目有哪些判断是成立的，哪些已经过时

### 5.1 仍然成立的部分

- `REPA` 这一方向值得做。
- `time-aware` 或 `timestep-aware` 的对齐方式值得研究。
- 不同 teacher 的异构影响值得比较。
- 只报一个 `FID` 不够，必须解释机制。

### 5.2 已经明显站不住的部分

- “DINOv3 一定带来 mode collapse”目前没有足够证据。
- “双教师一定更强”没有文献共识，反而会引入更复杂的归因问题。
- “静态 projector + 全程对齐”在 `HASTE` 之后已经不是安全设定。
- “更强 global semantics = 更好 alignment teacher”在 `iREPA` 之后已经被强烈质疑。
- “继续追加 teacher 就能打赢最新架构”在 `SRA / DUPA / RAE` 出现后不再成立，因为替代路线已经开始削弱外部 teacher 的必要性。

## 6. 如果目标真是冲击 CVPR 级强稿，应该如何重构项目

我建议不要继续把项目写成“DINOv3 + JEPA 双引擎屠榜”。更有机会的版本是把它改造成一个更尖锐的问题驱动型项目。

### 6.1 推荐的新核心问题

建议把论文主问题收缩为：

**外部表征对 DiT 训练真正有用的是什么，在哪些 timestep / layer / training phase 有用，以及何时应该停止对齐？**

这个问题比“我用了两个 teacher 所以更强”更像顶会强稿问题，因为它：

- 可以被实验证伪
- 能解释已有文献分歧
- 容易构造强消融
- 能兼容你已经想做的 `AdaLN`、routing、teacher 对比

### 6.2 推荐的方法重构

建议把当前 Ada-REPA 重构为一个更清晰的版本，例如：

**Stage-wise Spatial Representation Alignment for DiT**

核心组件只保留四个：

1. `Spatial-preserving projector`
   用卷积式或局部结构保留的 projector 取代纯 MLP。
   这是因为 `iREPA` 已经强烈提示“空间结构比全局语义更重要”。

2. `Timestep-layer routing`
   不是简单 soft matching，而是显式学习“什么 timestep、什么层、从哪个 teacher 或哪个 source 取什么表征”。

3. `Stage-wise termination`
   对齐损失必须允许在中后期关闭，不能默认全程开启。
   这是 `HASTE` 给出的强信号。

4. `Teacher/source modularity`
   不要默认双 teacher 最优。
   应把 `DINOv2 / DINOv3 / SigLIP / MAE / self-alignment` 统一成一个 source pool，在同一协议下比较。

### 6.3 不建议继续作为主线的部分

- 不建议把 `DINOv3 + JEPA` 双教师组合作为论文主 novelty。
- 不建议把 `SANA-1.5` 这种不同任务的模型放进主表作为“必须正面对打”的对象。
- 不建议用口号式写法驱动选题。

## 7. 面向 CVPR Best Paper 水位的最低实验要求

### 7.1 主战场必须统一

推荐主战场固定为：

- `ImageNet-256 class-conditional`

可选扩展战场：

- `ImageNet-512`
- 一个 text-to-image 数据集，但只能作为跨任务验证，不能与主表混写

### 7.2 主表 baseline 建议

主表应该只放可公平比较的同协议方法：

- `DiT`
- `SiT`
- `SD-DiT`
- `REPA`
- `REG`
- `HASTE`
- `REED`
- 你的方法

补充层可以再加：

- `SRA`
- `DUPA`
- `iREPA`
- `RAE`

注意：

- `RAE` 更像替代范式，不一定适合放同一主表，但必须在 related work 和补充实验里正面讨论。

### 7.3 指标必须升级

不能只报：

- `FID`

至少还要报：

- `sFID`
- `Precision`
- `Recall`
- `training FLOPs`
- `throughput`
- `wall-clock to target FID`
- `GPU hours`
- 多随机种子均值与方差

### 7.4 必须有的机制性实验

- teacher/source 对比：`DINOv2 / DINOv3 / SigLIP / MAE / self-alignment`
- projector 结构对比：`MLP / Conv / Conv + spatial norm`
- 对齐时机对比：`全程 / 前 10% / 前 25% / 触发式 early stop`
- 对齐层位对比：`浅层 / 中层 / 深层 / 多层`
- timestep 路由可视化
- 梯度冲突或 loss interaction 分析
- failure cases

### 7.5 可复现性要求

至少要有：

- 训练配置文件
- 一键启动命令
- 评测脚本
- 日志导出脚本
- 主结果表复现实验说明
- 随机种子
- 硬件与耗时说明

没有这一层，连强稿都谈不上，更别说 best paper。

## 8. 建议你现在就删除或重写的内容

建议从现有三份文档中立即删除或重写以下内容：

- 所有“屠榜”“镇压”“霸主”式表述
- 所有未经核验的会议归属
- 所有不同任务协议混排的主表
- 所有把假设写成结论的句子

建议替换成：

- 问题定义
- 假设
- 风险
- 验证设计
- 失败判据

## 9. 最终结论

如果保持当前写法，这个项目更像一份“战报式 proposal”，而不是一篇能过审稿的顶会论文。

如果按本文件建议重构，那么它最有机会成为一篇围绕以下命题展开的强稿：

**表征引导真正起作用的关键不是 teacher 语义越强越好，而是空间结构保真、分阶段注入，以及对齐在正确时机终止。**

这个命题是：

- 清晰的
- 可证伪的
- 有文献张力的
- 能产生机制性实验的

也只有沿这个方向重构，Ada-REPA 才有机会从“想法很多的 proposal”变成“有论文主线的项目”。

## 10. 主要来源

- CCF 官方人工智能会议分级目录：https://www.ccf.org.cn/Academic_Evaluation/AI/
- SD-DiT, CVPR 2024：https://cvpr.thecvf.com/virtual/2024/poster/30284
- Scaling Rectified Flow Transformers for High-Resolution Image Synthesis, ICML 2024：https://proceedings.mlr.press/v235/esser24a.html
- SiT, ECCV 2024：https://eccv.ecva.net/virtual/2024/poster/690
- REPA, ICLR 2025 Oral：https://openreview.net/forum?id=DJSZGGZYVi
- TinyFusion, CVPR 2025：https://openaccess.thecvf.com/content/CVPR2025/html/Fang_TinyFusion_Diffusion_Transformers_Learned_Shallow_CVPR_2025_paper.html
- FlexiDiT, CVPR 2025：https://cvpr.thecvf.com/virtual/2025/poster/34601
- SANA-1.5, ICML 2025：https://icml.cc/virtual/2025/poster/46604
- EDiT, ICCV 2025：https://openaccess.thecvf.com/content/ICCV2025/html/Becker_EDiT_Efficient_Diffusion_Transformers_with_Linear_Compressed_Attention_ICCV_2025_paper.html
- REG, NeurIPS 2025：https://openreview.net/forum?id=koEALFNBj1
- HASTE, NeurIPS 2025：https://openreview.net/forum?id=HK96GI5s7G
- REED, NeurIPS 2025：https://openreview.net/forum?id=cIGfKdfy3N
- SRA, arXiv 2025：https://arxiv.org/abs/2505.02831
- DINOv3, arXiv 2025：https://arxiv.org/abs/2508.10104
- iREPA, ICLR 2026：https://openreview.net/forum?id=y0UxFtXqXf
- DUPA, ICLR 2026：https://openreview.net/forum?id=ALpn1nQj5R
- RAE, ICLR 2026：https://openreview.net/forum?id=0u1LigJaab
