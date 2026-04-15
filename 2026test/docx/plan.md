# 科学化重构时间线与交付计划

## 文档职责（执行真值源）

- 文件角色：`执行计划` 与 `资源/里程碑管理` 的唯一真值源。
- 负责范围：阶段时间线、退出条件、交付物、算力与优先级。
- 不负责范围：模型结构细节（见 `architecture-gpt.md`）、实验协议细节（见 `exp.md`）、文献审计结论（见 `sum.md`）。
- 冲突仲裁：若与 `exp.md` 冲突，实验协议以 `exp.md` 为准；若与 `architecture-gpt.md` 冲突，结构细节以 `architecture-gpt.md` 为准。
- 统一入口：见 `doc-roles.md`。

## 总体目标

当前目标不再定义为“短期打榜”，而是定义为：

**在统一协议下完成一个可复现、可证伪、可解释的 DiT 表征对齐项目，并将其推进到 CCF A 级别强稿的完成态。**

内部项目代号仍可保留为 `Ada-REPA`，但论文主线将重构为：

**外部表征对 DiT 训练真正有用的是什么，在哪些 `timestep / layer / training phase` 有用，以及何时应该停止对齐。**

## 核心研究假设

1. 真正有效的不是“更强的全局语义”，而是**保留空间结构的表征**。
2. 表征对齐的收益主要集中在训练早期，**全程对齐不是默认最优策略**。
3. 不同 source 的价值应在统一协议下比较，外部 teacher 不应被预设为必然优于 self-alignment。

## 统一研究边界

- 主战场：`ImageNet-256 class-conditional`
- 可选扩展：`ImageNet-512 class-conditional`
- 不再将 `text-to-image`、控制生成、编辑任务与主表混排
- 所有主结果必须报告质量、效率和稳定性，而不只报告 `FID`

## 资源与执行约束

- 默认算力假设：`1` 个 `8x A100/H100 80GB` 节点或等效资源
- 若实际资源低于该假设，则优先保证 `DiT-B/2` 全部机制实验和 `DiT-XL/2` 的单一最佳配置
- `ImageNet-512` 只在 `ImageNet-256` 主结论稳定后才启动
- pilot 阶段允许宽 source sweep，主结果阶段只保留 shortlist，不做全量 source 重跑

## 主表复现策略

- `DiT`、`REPA` 和 Ours 必须完整复现
- `HASTE` 或 `REED` 至少完整复现一个
- 其余方法若无法在统一协议下完整复现，只能作为补充讨论对象
- 摘要 headline claim 只能建立在完整复现结果之上

## 时间线

### 阶段 0：文献与协议校正 (`2026-03-31` 至 `2026-04-03`)

- 重写 `paper.md`、`exp.md`、`plan.md`
- 校正所有会议归属、任务设定和 baseline 分层
- 将论文主问题从“多 teacher 提升”收缩为“表征属性与阶段性对齐机制”
- 明确主表、补充表和附录表的边界

交付物：

- 新版 `paper.md`
- 新版 `exp.md`
- 新版 `plan.md`

### 阶段 1：代码与复现基线搭建 (`2026-04-04` 至 `2026-04-10`)

- 搭建统一训练框架：`DiT-B/2` 与 `DiT-XL/2`
- 整理数据、评测和日志导出脚本
- 首先复现不带 alignment 的 `DiT` 基线
- 接着复现 `REPA` 风格基线，确保能在本地协议下稳定运行
- 明确定义 `self-alignment`、`alignment loss` 与 `adaptive stop` 的默认实现

交付物：

- 可运行训练脚本
- 可运行评测脚本
- 基线配置文件
- 第一版复现日志
- 方法定义说明页

### 阶段 2：先导实验与风险筛查 (`2026-04-11` 至 `2026-04-18`)

- 用 `DiT-B/2` 进行 source pilot：
  - `DINOv2`
  - `DINOv3`
  - `SigLIP`
  - `MAE`
  - self-alignment
- 验证 `Spatial-preserving projector` 是否优于纯 `MLP projector`
- 验证全程对齐与早停对齐的差异
- 确定是否保留某些 source 到主实验

阶段退出条件：

- 如果 external teacher 全部不优于 self-alignment，则项目主线切换为“自对齐优先”
- 如果空间保真 projector 没有带来稳定收益，则不再将其写成核心贡献
- 如果早停收益不稳定，则将其降级为附录结论
- 如果 alignment loss 形式比 projector 或 routing 更影响结果，则优先重审 loss 设计而非继续堆组件

交付物：

- pilot 表格
- 风险清单
- source shortlist

### 阶段 3：机制实验与主方法定型 (`2026-04-19` 至 `2026-05-02`)

- 完成以下核心消融：
  - `projector` 结构
  - `timestep-layer routing`
  - `alignment schedule`
  - `source pool`
- 生成以下分析结果：
  - 梯度交互分析
  - timestep 路由热力图
  - layer 选择统计
  - failure cases

阶段退出条件：

- 必须拿到一个清晰主结论，例如“空间结构比全局语义更关键”或“早停对齐显著优于全程对齐”
- 如果没有形成单一清晰结论，则重新收缩论文问题，避免方法堆叠

交付物：

- 主方法定版
- 分析图初稿
- 机制性结论摘要

### 阶段 4：公平主表与规模化实验 (`2026-05-03` 至 `2026-05-20`)

- 进入该阶段前，必须先确认算力预算、source shortlist 和默认 stop 规则
- 在统一协议下完成 `DiT-XL/2` 或同等级 backbone 主实验
- 与以下方法做公平比较：
  - `DiT`
  - `SiT`
  - `SD-DiT`
  - `REPA`
  - `REG`
  - `HASTE`
  - `REED`
- 报告以下指标：
  - `FID`
  - `sFID`
  - `Precision`
  - `Recall`
  - `throughput`
  - `training FLOPs`
  - `wall-clock to target FID`
  - `GPU hours`

交付物：

- 主结果表
- 效率表
- 多随机种子结果

### 阶段 5：稳健性、失败分析与论文写作 (`2026-05-21` 至 `2026-06-05`)

- 跑多随机种子并计算方差
- 补充失败案例和边界条件
- 写清楚何时 external teacher 有用，何时无用
- 在 related work 中单独讨论：
  - `REPA`
  - `SRA`
  - `iREPA`
  - `DUPA`
  - `RAE`

交付物：

- 完整论文初稿
- 全部主图和附图
- 失败模式章节

### 阶段 6：可复现封包与投稿准备 (`2026-06-06` 至 `2026-06-15`)

- 清理配置、命令与日志
- 补全附录和实现细节
- 统一术语与图表格式
- 按目标 venue 要求准备匿名化材料

交付物：

- 提交版论文
- 附录
- 复现实验说明

## 质量闸门

项目只有在同时满足以下条件时，才算进入“可投稿”状态：

1. 有可运行代码、配置和评测脚本。
2. 有至少一张统一协议下的公平主表。
3. 有机制性证据，而不只是单点分数。
4. 有失败分析和边界条件说明。
5. 有多随机种子和效率统计。
6. 有原版 `REPA` 的直接对照结果。
7. 有 `self-alignment`、`adaptive stop` 和 `alignment loss` 的明确实现定义。

## 负结果转向规则

- 若 `self-alignment` 最优，则将论文重心转向“external teacher 是否必要”。
- 若 `Stage-wise termination` 带来主要收益，则将论文重心转向“训练阶段比 teacher 选择更重要”。
- 若 `Spatial-preserving projector` 带来主要收益，则将论文重心转向“空间结构保真是关键条件”。
- 若以上任何一种转向发生，必须同步重写标题、摘要、贡献和主实验顺序。

## 明确不再采用的写法

- 不再使用“屠榜”“镇压”“降维打击”等措辞
- 不再混用不同任务协议的 baseline
- 不再把未经验证的判断写成结论
- 不再把 `arXiv` 工作与严格 CCF A 主会结果写成同一层级

## 联动规则

`plan.md`、`exp.md`、`paper.md` 仍保持联动，但联动原则改为：

- `paper.md` 定义主问题、方法与写作边界
- `exp.md` 定义验证协议、主表和风险控制
- `plan.md` 定义执行顺序、退出条件和交付物

任何一处修改，都必须同步检查另外两处是否仍然逻辑一致。
