# Stage-wise Spatial Representation Alignment for DiT

## 逐阶段实现与实验执行清单

本文件配合 [module-checklist-gpt.md](/home/liuchunfa/2026qjx/2026test/docx/module-checklist-gpt.md) 使用。

前者回答“代码里要有什么”，本文件回答“先做什么、怎么验、做到哪一步才继续”。

---

## 1. 总体执行原则

- 先拿到 **单 source + fixed stage-wise + no router** 的稳定训练
- 再加 `self-alignment`
- 再加 `routing`
- 最后才做 `DINOv2` control 和大模型主表

不要一开始就并行做：

- 多 source
- 多分辨率
- adaptive stop
- 多 source joint routing

---

## 2. Phase 0：无行为变化的代码重构

### 目标

把现有 `train_2.py` 中与 alignment 相关的逻辑拆清楚，但先不改变训练行为。

### 必做项

- 把当前 `REPAProjector`、teacher、preprocess、extract token 的逻辑分块
- 给 checkpoint 增加结构化字段，不改变已有保存内容
- 在 [models_2.py](/home/liuchunfa/2026qjx/2026test/models_2.py) 增加多层 feature 导出接口，但默认仍只取一层

### 验收

- 现有 `run_sit_repa_dinov3_b24_160k.sh` 能继续跑
- loss 曲线与旧版本基本一致

---

## 3. Phase 1：Original REPA 基线稳定化

### 目标

把 `Original REPA` 作为之后所有增量实验的对照锚点。

### 结构

- source: `DINOv2` 或当前现成 `DINOv3` 临时代替
- projector: `MLP`
- matching: fixed layer
- schedule: full-course

### 必做项

- 跑通单一 `repa_token_layer`
- 记录：
  - `FID`
  - `sFID`
  - `Recall`
  - `throughput`
  - `GPU memory`

### 验收

- 有稳定下降的训练 loss
- 有一组 baseline checkpoint
- 有一条可作为后续比较锚点的日志

---

## 4. Phase 2：Spatial projector only

### 目标

验证“把 MLP 换成 spatial projector”本身有没有独立价值。

### 结构

- source: `DINOv3`
- projector: `SpatialProjector2D`
- matching: fixed layer
- schedule: full-course
- router: none

### 必做项

- 实现 `SpatialProjector2D`
- 替换当前 `REPAProjector`
- 保持其他变量不变

### 必须输出

- `MLP vs Conv1x1 vs DWConv3x3+1x1`

### 验收

- 至少一个 spatial projector 版本稳定优于 MLP
- 若不能优于 MLP，则暂停后续 routing，先复盘 projector

---

## 5. Phase 3：Fixed stage-wise controller

### 目标

验证“对齐在训练早期更有用”。

### 结构

- source: `DINOv3`
- projector: 最佳 spatial projector
- matching: fixed layer
- schedule: fixed early-stop
- router: none

### 必做项

- 实现 `StageController`
- 默认 schedule：
  - `0%~10%`: `lambda = 0.1`
  - `10%~40%`: decay to `0`
  - `40%~100%`: `0`

### 必做对照

- full-course
- early 10%
- early 25%
- early 50%
- cosine decay

### 验收

- 至少一种 early-stop 明显优于 full-course 或在相近效果下更省 wall-clock

---

## 6. Phase 4：Self-alignment branch

### 目标

回答 external teacher 是否真的必要。

### 结构

- 增加 `EMA DiT self-source`
- 其他模块保持不变

### 必做项

- `EMA DiT` 中间特征导出
- 走同一个 `SourceBank -> adapter -> alignment loss` 路径

### 必做对照

- `DINOv3`
- `self-alignment`

### 验收

- 至少能稳定训练
- 能给出 `external vs self` 的明确趋势

### 决策规则

- 如果 `self` 明显更优，后续主线要转向“external teacher 是否必要”

---

## 7. Phase 5：Single-source layer routing

### 目标

验证 `H3`：不同 `timestep × layer` 的有效性不同。

### 结构

- source: 单 source
- projector: 最佳 spatial projector
- schedule: 最佳 fixed stage-wise
- router: 单 source 内部层路由

### 必做项

- 实现 `SparseLayerRouter`
- 先只做 source 内部 `K` 层 routing
- `top-k = 2`

### 必做对照

- fixed shallow
- fixed middle
- fixed deep
- layer-only routing
- timestep+layer routing

### 必须输出

- routing heatmap
- 不同 timestep 对不同 layer 的偏好统计

### 验收

- learned routing 至少在一个维度上明显优于固定层，或产生清晰的可解释模式

---

## 8. Phase 6：DINOv2 历史 control

### 目标

把“架构增益”和“更换 teacher 的增益”分开。

### 结构

- source: `DINOv2`
- projector / schedule / router：使用当前最佳配置

### 必做项

- 加入 `/14` 输入适配
- 推荐用 `252x252` source 输入

### 必做对照

- `Original REPA + DINOv2`
- `Your method + DINOv2`

### 验收

- 如果在 `DINOv2` 上仍能带来稳定增益，你的方法才算真正对 `REPA` 有结构改进价值

---

## 9. Phase 7：主文 source 筛选

### 目标

决定主文保留哪些 source。

### 建议顺序

- 必选：
  - `DINOv3`
  - `DINOv2`
  - `self`

- 可选附录：
  - `MAE`
  - `SigLIP`

### 决策规则

- 主文只保留 `3` 条 source 线
- 其他 source 不进主表

---

## 10. Phase 8：大模型主表

### 目标

在较大 backbone 上得到公平主表。

### 建议

对你当前机器，优先上：

- `DiT-B/2` 完整消融
- `DiT-XL/2` 或接近尺度只跑最优配置

如果显存或速度不够，不要强行把所有消融搬到 `XL`

### 主表 baseline

- `DiT`
- `Original REPA`
- `HASTE` 或 `REED` 至少一个
- Ours

### 主表指标

- `FID`
- `sFID`
- `Precision`
- `Recall`
- `training FLOPs`
- `throughput`
- `wall-clock to target FID`
- `GPU hours`

---

## 11. 机制实验清单

主文必须有：

- `MLP vs spatial projector`
- `full-course vs fixed early-stop`
- `fixed layer vs routing`
- `DINOv3 vs self`
- `Original REPA + DINOv2 vs Ours + DINOv2`

附录可有：

- `adaptive stop`
- `MAE`
- `SigLIP`
- `top-k` 不同设置
- `DINOv2` 输入尺寸策略

---

## 12. 日志与可视化清单

### 训练时必须记录

- `loss_diff`
- `loss_align`
- `lambda_align`
- per-block alignment loss
- router layer weights
- GPU memory
- imgs/s

### 训练后必须产出

- routing heatmap
- stage-wise loss curve
- external vs self qualitative cases
- projector 对比表

---

## 13. 失败判据与转向规则

### 如果 spatial projector 不稳定

- 先停 routing
- 回到 projector 结构本身

### 如果 early-stop 没优势

- 主文降级 schedule 结论
- 把重点放回 spatial / self

### 如果 self-source 最优

- 主问题转向 external teacher necessity

### 如果 routing 没有清晰模式

- 不要强行把 routing 放 headline
- 作为附录增强项

---

## 14. 最终推荐执行顺序

严格按下面顺序推进：

1. 无行为变化重构
2. `Original REPA` 基线
3. `SpatialProjector2D`
4. `StageController`
5. `EMADiTSelfSource`
6. `SparseLayerRouter`
7. `DINOv2` control
8. 大模型主表

如果顺序乱了，最容易出现的问题是：

- 你不知道提升来自哪里
- debug 成本暴涨
- 论文叙事也会跟着散掉

---

## 15. 完成标志

只有满足以下条件，才说明这套清单已经被真正执行完：

1. 有 `Original REPA` 对照日志
2. 有 `SpatialProjector2D` 优于或至少不弱于 MLP 的证据
3. 有固定 stage-wise schedule 的有效证据
4. 有 `DINOv3 / DINOv2 / self` 三条主线结果
5. 有 routing heatmap
6. 有至少一个大模型主表结果
7. 有能直接写进论文的机制图与表
