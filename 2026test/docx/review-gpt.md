# 项目总览与方法重构

## 1. 文档定位

本文件用于替代旧版 `project-overview.md`。

这不是一份“把已有想法重新罗列一遍”的摘要，而是一份新的项目设计稿。目标是把当前项目从“几个模块并列堆叠的 proposal”，重构成一条更集中、更像方法论文的主线。

本文件的基本立场是：

- 项目仍然应该是一篇方法论文
- 但方法主轴必须收敛，不能再用三个松散创新点并列支撑
- 新主线要能同时回应 `REPA`、`iREPA` 和 `HASTE`

---

## 2. 新的项目定义

本项目不再定义为：

**我们提出三个互相并列的新模块来改进 REPA。**

而重新定义为：

**我们提出一种面向 DiT 的 stage-aware representation alignment 方法，在保留空间结构的前提下，使外部表征只在合适的训练阶段和噪声阶段发挥作用。**

更简洁地说，新的方法主线是：

**Stage-Aware Spatial Alignment for DiT**

这个定义仍然是方法导向的，不是纯机制分析，也不是单纯可解释性工作。

---

## 3. 核心问题

当前 `REPA` 路线的真正矛盾，不是“teacher 不够强”，而是下面两个问题同时没有被处理好：

1. 表征迁移过程中，空间结构容易被普通 `MLP projector` 弱化
2. alignment 的有效性不是静态的，而是会随训练阶段和噪声阶段变化

因此，本项目的方法核心不是再叠加更多模块，而是围绕一个中心命题展开：

**alignment 应该是 stage-aware 的，同时其特征传递方式必须 spatially faithful。**

---

## 4. 方法主轴

### 4.1 方法名称

建议暂定方法名：

**SASA-DiT**

即：

**Stage-Aware Spatial Alignment for Diffusion Transformers**

这个名字的好处是：

- 保留你们原始设计中“spatial”与“stage-aware”两个最关键方向
- 不把 `routing` 或 `early stop` 提前绑定成 headline
- 给后续实现留出版本空间

### 4.2 一句话方法定义

SASA-DiT 在 DiT 训练期间引入一条 alignment side branch，通过空间结构保真的 projector 将 DiT 中间 token 映射到 teacher feature space，并使用一个 stage-aware controller 按照训练阶段与 diffusion timestep 动态调节 alignment 强度。

### 4.3 总体训练目标

方法的主训练目标写成：

`L = L_diff + lambda_stage(s, t) * L_align`

其中：

- `s` 表示 training step
- `t` 表示 diffusion timestep
- `lambda_stage(s, t)` 是本方法的核心控制量

这里最关键的点是：

**本方法的中心创新不再是“有没有 alignment”，而是“alignment 何时、以多大强度发生”。**

---

## 5. 三层方法结构

为了避免方法重新发散，新的设计只保留三层结构。

### 5.1 第一层：主创新层

#### Stage-Aware Alignment Controller

这是整篇方法的真正主创新。

它不再只是简单的 early stop，也不是抽象的“时间步注入”，而是一个明确的二维控制器：

`lambda_stage(s, t) = lambda_train(s) * lambda_diff(t)`

其中：

- `lambda_train(s)` 控制 alignment 在整个训练过程中的阶段性强弱
- `lambda_diff(t)` 控制不同噪声水平下 alignment 的作用强弱

这条设计直接对应当前项目最值得做实的核心点：

- 不是只有训练前期和后期不同
- 不同 diffusion timestep 对外部语义的需求也不同

因此，方法的第一性创新点应表述为：

**提出一个同时建模 training phase 与 noise level 的 stage-aware alignment 机制。**

### 5.2 第二层：关键支撑层

#### Spatially Faithful Projector

projector 的角色不再被包装成一个独立 headline innovation，而被定义为：

**为了让 stage-aware alignment 真正服务于生成任务所必需的结构支撑。**

它的职责是：

- 保留 token grid 的局部拓扑
- 避免普通 `MLP projector` 把对齐过程进一步推向全局语义
- 让 alignment 对布局、边界和局部结构更友好

推荐第一版结构仍然是：

- token reshape 为 `H x W`
- `1x1 Conv`
- `DWConv 3x3`
- `Norm`
- `SiLU`
- `1x1 Conv`
- residual

这部分在论文中的定位应当是：

- 关键设计
- 必要支撑
- 不是单独 claim 的主创新

### 5.3 第三层：增强模块层

#### Source Transition / Routing-Lite

第三层不再硬写成“timestep-layer routing”。

原因很直接：

- 如果一上来就写成 full routing，容易显得像把已有思路缝在一起
- 当前工程实现也还没有足够支撑复杂 routing

因此更合理的第三层是二选一：

1. `Source Transition`
   前期更多依赖 external teacher，后期弱化 external teacher，必要时切换到 self-alignment

2. `Routing-Lite`
   不做复杂 layer-source router，只做轻量的 block-wise 或 timestep-wise gating

这层模块的作用是增强方法，而不是定义方法。

---

## 6. 与现有工作的区别

### 6.1 相对 REPA

相对 `REPA`，本项目不再把 alignment 视为静态、全程、统一强度的辅助项。

核心区别是：

- `REPA` 更接近固定对齐
- SASA-DiT 强调 stage-aware 对齐

因此，和 `REPA` 的核心差异不应再写成“我们用了更复杂的 projector”，而应写成：

**我们改变了 alignment 的作用机制，而不仅是改变映射结构。**

### 6.2 相对 iREPA

相对 `iREPA`，本项目不应重复 claim “空间结构比全局语义更重要” 这件事本身。

更合理的差异定位是：

- `iREPA` 主要回答“align 什么信息更重要”
- SASA-DiT 进一步回答“这些信息应在什么阶段被使用”

因此，你们和 `iREPA` 的关系应该是：

- 接受它关于 spatial structure 的结论
- 在其基础上往 `stage-aware control` 推进一步

### 6.3 相对 HASTE

相对 `HASTE`，本项目也不能只把自己写成“更平滑的 early stop”。

真正应强调的差异是：

- `HASTE` 主要强调 training phase 上的两阶段终止
- SASA-DiT 同时显式建模 training phase 和 diffusion timestep

换句话说：

`HASTE` 更像一维 schedule，
SASA-DiT 目标是二维 stage control。

这才是你们最有机会形成差异性的地方。

---

## 7. 当前代码的重新定位

当前代码不应再被称为最终方法实现。

更准确的定位是：

**当前代码是 SASA-DiT 之前的过渡 baseline。**

它当前真正实现的是：

- 单一 `DINOv3` teacher
- 单层 DiT token tap
- token-wise `MLP projector`
- token cosine alignment
- 仅按 training step 线性衰减

因此，它更接近：

**REPA-style baseline with linear training-step decay**

这个定位非常重要，因为它决定了后续论文写作不能再把当前结果直接写成“Ours final”。

---

## 8. 新的创新点组织方式

旧版写法的问题是把三个创新点并列展开，导致每一点都很容易被 reviewer 说成：

- 不是新的
- 不够完整
- 或者和已有工作高度重叠

新的写法必须改成“一个主创新 + 两个服务它的设计”。

### 8.1 主创新

**Stage-aware alignment controller**

这是论文最核心的创新点。

它负责回答：

- alignment 在训练的什么时候该强、什么时候该弱
- alignment 在不同噪声水平下是否应当区别对待

### 8.2 服务性设计一

**Spatially faithful feature transfer**

它不是独立创新主角，而是保证主创新成立的必要支撑。

### 8.3 服务性设计二

**Source transition or routing-lite**

它的作用是增强 stage-aware alignment，而不是单独 claim 成一个庞大新模块。

---

## 9. 方法版本规划

为了避免第一版就做得过重，建议按三个版本推进。

### 9.1 Version A：最小方法版

只做：

- `lambda_train(s)`
- `lambda_diff(t)`
- 原有 MLP projector

这个版本的目标是先验证：

**二维 stage control 是否比单纯 training-step schedule 更有效。**

这是最关键的第一步。

### 9.2 Version B：完整主文版

在 Version A 有正结果之后，再加入：

- spatially faithful projector

这样主文就形成了：

- 主创新：stage-aware controller
- 关键支撑：spatial projector

### 9.3 Version C：增强版

如果还有资源，再在后续版本中加入：

- external-to-self transition
或
- lightweight routing

这部分更适合作为增强实验或附加模块，不应抢主线。

---

## 10. 实验主线重排

### 10.1 主实验问题

主实验不再围绕“三个模块各自有没有用”展开，而围绕下面这个链条展开：

1. 静态 alignment 是否不够
2. 单轴 stage schedule 是否不够
3. 二维 stage-aware alignment 是否更合理
4. spatial projector 是否进一步增强这种方法

### 10.2 建议的核心对照

主表建议至少包含：

1. No alignment
2. REPA-style full-course alignment
3. HASTE-style early stop
4. current linear decay baseline
5. SASA-DiT-A: `lambda_train(s) * lambda_diff(t)`
6. SASA-DiT-B: `lambda_train(s) * lambda_diff(t) + spatial projector`

如果资源允许，再补：

7. SASA-DiT-C: + source transition / routing-lite

### 10.3 主实验要回答的不是

不要再把主表问题写成：

- 哪个 projector 最复杂
- 哪个 routing 最花哨
- 哪个 teacher 组合最多

主实验真正要回答的是：

**在 DiT alignment 中，二维 stage-aware control 是否能带来比静态或单轴 schedule 更稳定的收益。**

---

## 11. 应用场景

### 11.1 主应用场景

本项目的唯一主场景仍然是：

- `ImageNet-1K`
- `256x256`
- `class-conditional generation`
- `DiT + VAE`

### 11.2 当前不进入主文主线的场景

以下内容暂不进入第一版主文：

- text-to-image
- 多 teacher 联合主线
- 推理时动态多阶段注入
- 高分辨率扩展作为核心卖点

这些都可以留作后续扩展，但不应干扰主方法叙事。

---

## 12. 论文贡献建议写法

如果后续实验支持当前设计，贡献建议组织为：

1. 我们提出一种面向 DiT 的 stage-aware representation alignment 方法，通过联合建模 training phase 与 diffusion timestep，使 alignment 强度随训练阶段和噪声水平动态变化。
2. 我们引入 spatially faithful 的 feature transfer branch，使 alignment 更适配生成任务中的局部结构建模，而不是仅强化全局语义。
3. 我们在统一的 `ImageNet-256 class-conditional` 协议下系统比较静态 alignment、单轴 stage schedule 与二维 stage-aware alignment，验证后者具有更强的稳定性和更清晰的收益来源。

这套写法比旧版的“三个创新点并列”更收敛，也更像一篇方法论文。

---

## 13. 失败时的转向规则

如果后续实验不完全支持当前主线，也不要硬保三点创新。

### 13.1 如果二维 stage control 有效

则保持当前方法主线不变。

### 13.2 如果 spatial projector 有效，但 `lambda_diff(t)` 收益有限

则转向：

**空间结构保真是 DiT alignment 生效的关键条件**

### 13.3 如果 early stop 或 source transition 收益远大于其他模块

则转向：

**alignment 何时停止或切换，比对齐什么更重要**

### 13.4 如果 self-alignment 明显优于 external teacher

则转向：

**external teacher 是否真的必要**

---

## 14. 一句话总结

新的项目设计不再把论文写成“三个零散模块的拼接改进”，而是收缩为一篇更明确的方法论文：

**以 stage-aware alignment controller 为主创新，以 spatially faithful projector 为关键支撑，以 source transition / routing-lite 为增强模块，构建一条更集中、更容易 defend 的 DiT representation alignment 主线。**
