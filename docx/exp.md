# 实验计划详解与验证矩阵

## 文档职责（实验真值源）

- 文件角色：`实验协议与评测规范` 的唯一真值源。
- 负责范围：任务边界、训练/评测协议、实验矩阵、主表与消融、指标和验收标准。
- 不负责范围：项目排期与算力排班（见 `plan.md`）、模型结构实现细节（见 `architecture-gpt.md`）、文献快照与复盘（见 `sum.md`）。
- 冲突仲裁：若与 `plan-gpt.md` 或 `sum.md` 冲突，以本文件为准。
- 统一入口：见 `doc-roles.md`。

## 1. 统一实验设定

### 1.1 主任务

- 主战场：`ImageNet-1K, 256x256, class-conditional generation`
- 可选扩展：`ImageNet-1K, 512x512`
- 主表不再混入 `text-to-image`、控制生成或图像编辑任务

### 1.2 主干模型

- 小规模机制验证：`DiT-B/2`
- 主结果模型：`DiT-XL/2` 或与之同量级 backbone

### 1.3 训练与评测协议

- 优化器：`AdamW`
- 学习率初始值：`1e-4`
- 目标全局 batch size：`256`
- `EMA decay = 0.9999`
- pilot 阶段统一训练 `100K` steps，主结果阶段统一训练 `400K` steps
- 统一采样 `50,000` 张图像计算主指标
- 使用相同的 pretrained VAE、类别标签、采样器和随机种子集合
- `CFG` 采用统一扫描网格：`{1.0, 1.5, 2.0, 2.5, 3.0, 4.0}`，主表明确标注最佳值与默认报告值
- `FID / sFID` 统一使用相同实现与预处理流程，推荐固定为 `clean-fid`
- 所有 external source encoder 默认冻结，不参与梯度更新
- source feature 默认从固定候选层集合中选择：`{1/4, 1/2, 3/4, final}`
- 所有方法采用统一数据预处理、训练步数和评测脚本
- 至少运行 `3` 个随机种子用于主结论确认

主表复现策略：

- `DiT`、`REPA` 风格基线和 Ours 必须完整复现
- 至少一个强现代 baseline 必须完整复现，优先 `HASTE` 或 `REED`
- 无法完整复现的方法只能出现在补充表或 related work 中，并明确标记为 `reported`

### 1.4 必报指标

主结果不能只报 `FID`，至少必须包含：

- `FID`
- `sFID`
- `Precision`
- `Recall`
- `throughput (img/s)`
- `training FLOPs`
- `wall-clock to target FID`
- `GPU hours`

### 1.5 source pool

主文只使用统一协议下可直接比较的 source：

- `DINOv2`
- `DINOv3`
- `SigLIP`
- `MAE`
- self-alignment

说明：

- `I-JEPA` 可作为附录中的历史性补充 baseline，但不再作为主叙事核心。
- 如果 pilot 结果显示 external teacher 全部不优于 self-alignment，则主线转为“source-free / self-guided alignment”。

## 2. 要验证的核心假设

### 假设 A

对 DiT 训练真正有帮助的，不是更强的全局语义，而是**可保留空间结构的表征**。

### 假设 B

alignment 的收益集中在训练早期，**全程对齐不是默认最优策略**。

### 假设 C

不同 source 的作用与 `timestep / layer` 强相关，因此需要**timestep-layer routing**，而不是静态层对层匹配。

## 3. 方法组件定义

为了避免“工程拼接式创新”，主方法只保留四个组件：

1. `Spatial-preserving projector`
   保留局部空间结构的 projector，用于替代纯 `MLP projector`。

2. `Timestep-layer routing`
   依据 timestep 和 DiT 层位动态选择 source feature。

3. `Stage-wise termination`
   alignment loss 允许在训练中后期关闭，避免全程强约束。

4. `Source modularity`
   所有 source 在同一接口下比较，不预设双 teacher 最优。

### 3.1 操作性定义

- `Original REPA baseline`
  定义为：静态 `MLP projector` + 固定 layer matching + 全程 alignment。
- `self-alignment`
  定义为：使用当前 DiT 的 `EMA` 副本作为 source network，并从对应空间分辨率的中间层提取 feature，不引入外部 encoder。
- `alignment loss`
  默认定义为 routed feature 对上的归一化余弦距离：
  `L_align = mean(1 - cos(P(h_dit), z_src))`。
  主文默认使用该形式，`MSE` 与其他变体只在附录补充。
- `adaptive stop`
  默认规则为：训练前 `10%` steps 强制开启 alignment；此后每 `10K` steps 评估一次 `FID-10k` 代理指标，若连续 `3` 次评估没有改善，则永久关闭 alignment。
  梯度统计触发的 stop 规则只作为附录补充。

## 4. 核心实验矩阵

### 矩阵一：组件消融矩阵

**目的**：验证主方法中的每个组件是否带来独立、稳定的收益。  
**模型**：`DiT-B/2`  
**source**：先固定 `DINOv2`，控制变量后再扩展

| 实验编号 | Alignment Recipe | 目标 |
| --- | --- | --- |
| Exp 1.1 | No alignment | 作为无 alignment baseline |
| Exp 1.2 | Original REPA: static `MLP` + fixed matching + full-course | 提供对原版路线的直接对照 |
| Exp 1.3 | Spatial projector + fixed matching + full-course | 验证空间结构保真是否优于原版静态投影 |
| Exp 1.4 | Spatial projector + routing + full-course | 验证动态路由是否优于固定匹配 |
| Exp 1.5 | Spatial projector + routing + stage-wise termination | 验证完整方法是否优于全程对齐 |

观测重点：

- `FID / sFID / Recall`
- 收敛速度
- `throughput`
- 梯度交互是否更稳定
- 相对 `Original REPA baseline` 的直接增益来源

### 矩阵二：source 对比矩阵

**目的**：在统一协议下比较 external teacher 与 self-alignment。  
**模型**：`DiT-B/2`  
**方法**：固定为矩阵一中表现最好的版本

| 实验编号 | Source | 类型 | 主要验证点 |
| --- | --- | --- | --- |
| Exp 2.1 | `DINOv2` | 对比式表征 | 作为稳定视觉 teacher baseline |
| Exp 2.2 | `DINOv3` | 更强语义 teacher | 验证强语义是否必然有利 |
| Exp 2.3 | `SigLIP` | 视觉-语言表征 | 验证跨模态语义是否适合 class-conditional 任务 |
| Exp 2.4 | `MAE` | 重建式表征 | 验证弱不变性、强结构信息是否更有效 |
| Exp 2.5 | self-alignment | 无外部 teacher | 验证 external teacher 是否必要 |

附录可选：

- `I-JEPA`

观测重点：

- 哪类 source 对 `Recall` 更友好
- 哪类 source 在训练早期收敛更快
- 哪类 source 更依赖空间保真 projector

### 矩阵三：对齐时机矩阵

**目的**：验证 alignment 在什么训练阶段最有效。  
**模型**：`DiT-B/2` 或最佳 source 的中等规模模型

| 实验编号 | 对齐策略 | 描述 |
| --- | --- | --- |
| Exp 3.1 | Full-course | 全程保持 alignment |
| Exp 3.2 | Early 10% | 只在前 `10%` 训练阶段开启 |
| Exp 3.3 | Early 25% | 只在前 `25%` 训练阶段开启 |
| Exp 3.4 | Early 50% | 只在前 `50%` 训练阶段开启 |
| Exp 3.5 | Adaptive stop | 按 `FID-10k` patience 规则自适应停止 |

观测重点：

- `wall-clock to target FID`
- 最终 `FID / Recall`
- 中后期是否出现过约束现象

### 矩阵四：层位与 timestep 分析矩阵

**目的**：验证“什么层、什么 timestep 使用什么 source”最有效。  
**模型**：矩阵一和矩阵二筛出的最佳配置

实验维度：

- 只对齐浅层
- 只对齐中层
- 只对齐深层
- 多层共享路由
- 固定路由
- 学习式 routing

观测重点：

- 路由热力图
- timestep-source 对应关系
- layer-source 对应关系

### 矩阵五：公平主表矩阵

**目的**：在统一协议下与强 baseline 公平比较。  
**模型**：`DiT-XL/2`

主表 baseline：

- `DiT`
- `SiT`
- `SD-DiT`
- `REPA`
- `REG`
- `HASTE`
- `REED`
- Ours

说明：

- `SANA-1.5`、`TinyFusion`、`FlexiDiT`、`EDiT` 等工作可在 related work 或补充实验中讨论其效率思路，但不应在不同协议下直接混入同一主表。
- 主表中的每个方法必须明确标注 `reproduced` 或 `reported`。
- 摘要和主结论只允许依赖 `reproduced` 的结果，不允许依赖纯引用结果。

主表必须报告：

- 主质量指标
- 训练效率指标
- 多随机种子均值与标准差

## 5. 分析与可视化计划

### 5.1 梯度交互分析

- 对比无 alignment、全程 alignment、早停 alignment 的梯度相似度
- 验证是否存在明显的中后期冲突

### 5.2 routing 热力图

- 展示不同 timestep 从哪些层和哪些 source 取特征
- 检查是否存在稳定的阶段性模式

### 5.3 source 行为画像

- 对比不同 source 在 `FID / Recall / 收敛速度` 上的表现
- 对比不同 source 是否更依赖空间结构保留

### 5.4 failure cases

- 展示多样性不足的样例
- 展示细节纹理丢失的样例
- 展示语义正确但空间结构差的样例
- 展示 external teacher 优于 self-alignment 的样例，以及相反的反例样例

## 6. 风险控制与转向条件

### 风险 1：external teacher 无明显优势

应对：

- 立即将主问题改写为 self-alignment 与 external guidance 的公平比较
- 将 external teacher 作为分析对象，而不是核心贡献前提
- 标题与摘要同步改写为“external teacher 是否必要”的研究问题

### 风险 2：空间保真 projector 没有稳定收益

应对：

- 降级为附录消融
- 将主方法重心转向 `schedule + routing`

### 风险 3：早停策略无法稳定提升最终结果

应对：

- 保留其作为效率结论
- 不再把其写为“必要条件”

### 风险 4：主表无法公平复现全部 baseline

应对：

- 区分“主表 baseline”和“补充讨论 baseline”
- 只保留协议一致、可复现的对象进入主表

### 风险 5：alignment loss 本身成为主要变量

应对：

- 增补 `cosine vs MSE` 的小型消融
- 若 loss 形式对结论影响超过 projector 或 routing，则重新定义主创新层级，避免误报贡献

## 7. 验收标准

实验部分只有在满足以下条件时才算完成：

1. 有统一协议下的主表。
2. 有至少一个清晰、可证伪且经实验支持的主结论。
3. 有效率结果，而不只是质量结果。
4. 有多随机种子统计。
5. 有失败案例和边界条件说明。
