# Stage-wise Spatial Representation Alignment for DiT

## 实验矩阵设计与结果记录台账

## 文档职责

- 文件角色：`实验矩阵设计 + 过程记录 + 结果归档` 的统一台账。
- 适用范围：`ImageNet-1K, 256x256, class-conditional, standard VAE latent, DiT backbone`。
- 记录对象：
  - 方法设计版本
  - 实验矩阵
  - 对比模型
  - 训练/评估配置
  - 关键结果
  - 结论与后续动作

## 使用方式

- 每个实验一行，不重复记。
- 每次改动方法结构时，先更新“设计版本表”，再立实验。
- 每次实验结束后，必须填写：
  - `状态`
  - `checkpoint`
  - `核心指标`
  - `一句话结论`
  - `后续动作`

建议状态枚举：

- `planned`
- `running`
- `done`
- `failed`
- `deprecated`
- `legacy`

---

## 1. 全局固定设定

| 项目 | 当前默认值 | 备注 |
| --- | --- | --- |
| 数据集 | ImageNet-1K | class-conditional |
| 主分辨率 | 256x256 | 主文默认 |
| latent | standard VAE latent | 不改 latent 范式 |
| backbone-mvp | DiT-B/2 | 机制验证 |
| backbone-main | DiT-XL/2 | 主表 |
| default external source | DINOv3-ViT-L/16 | 工程主线 |
| control external source | DINOv2-ViT-L/14 | REPA 历史锚点 |
| self source | EMA DiT | 主文级对照 |
| default align loss | token-level cosine | 主文默认 |
| default stage schedule | fixed early-stop | 主文默认 |
| default router | sparse timestep-layer router | 主文默认 |
| 主文 scope | ImageNet-256 + standard DiT/VAE | 不外推跨任务 |

---

## 2. 核心假设跟踪

| 假设 ID | 假设内容 | 关键验证实验 | 当前状态 | 当前证据 | 是否成立 | 备注 |
| --- | --- | --- | --- | --- | --- | --- |
| H1 | 保留空间结构的表征比更强的全局语义更稳定提升 DiT 训练 | 矩阵 B | planned | 当前仅有 legacy `SiT-XL/2 + DINOv3` strict ADM 结果，不能直接判断 H1 | 待定 | 重点看 spatial intact vs global/shuffled |
| H2 | alignment 收益集中在训练早期，全程对齐不是最优 | 矩阵 C | planned | 当前仅有 full-course/旧式 schedule 证据，不足以判断 | 待定 | 重点看 wall-clock/FID/Recall |
| H3 | 不同 source 的有效性依赖 timestep × layer，routing 比固定 matching 更合理 | 矩阵 E | planned | 当前没有 routing heatmap 或层路由实证 | 待定 | 必须有 heatmap |

---

## 3. 设计版本表

### 3.1 架构版本追踪

| 设计版本 | 日期 | SourceBank | Projector | Router | Stage Controller | Self Source | 目标 | 状态 | 备注 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| V0 | 2026-04-13 | 单一 DINOv3 teacher | MLP projector | none | 旧式静态/非 stage-aware 方案 | none | 复现当前 REPA 风格基线 | legacy | 当前已有实测主要来自该路线，且是 SiT 分支 |
| V1 | 2026-04-13 | DINOv3 | SpatialProjector2D | none | fixed early-stop | none | 验证 spatial + stage | planned | MVP |
| V2 | 2026-04-13 | DINOv3 + EMA self | SpatialProjector2D | none | fixed early-stop | EMA DiT | external vs self | planned | 主问题扩展 |
| V3 | 2026-04-13 | DINOv3 + DINOv2 + EMA self | SpatialProjector2D | sparse layer router | fixed early-stop | EMA DiT | 完整主文版本 | planned | 主线完整形态 |
| V4 | 2026-04-13 | 同 V3 | SpatialProjector2D | sparse layer router | adaptive stop | EMA DiT | 附录扩展 | planned | 不作为主文默认 |

### 3.2 模块状态追踪

| 模块 | 文件落点 | 目标类/接口 | 当前状态 | 验收标准 | 备注 |
| --- | --- | --- | --- | --- | --- |
| 多 tap block 导出 | models_2.py | `DiT.forward_features_multi` | planned | 一次前向返回多个 block tokens/maps | 不能重复跑 DiT |
| DINOv3 source | train_2.py / 新模块文件 | `DINOv3Source` | planned | 返回多层 spatial features | 不只最后一层 token |
| DINOv2 source | train_2.py / 新模块文件 | `DINOv2Source` | planned | 返回多层 spatial features | 输入建议 252 |
| self source | train_2.py / 新模块文件 | `EMADiTSelfSource` | planned | 能走同一 source 接口 | stop-gradient |
| source 统一接口 | 新模块文件 | `SourceBank` | planned | 统一输出 `{source: {layer: map}}` | 主文只留 3 条 source |
| spatial projector | 新模块文件 | `SpatialProjector2D` | planned | 输出 `[B, C_a, 16, 16]` | 替代 MLP |
| sparse router | 新模块文件 | `SparseLayerRouter` | planned | 输出层权重和 top-k 选择 | 要可视化 |
| stage controller | 新模块文件 | `StageController` | planned | 能按训练阶段衰减 lambda | 主文默认 fixed |
| loss 聚合 | train_2.py | `compute_alignment_loss` | planned | 支持多 tap block 汇总 | 记录 per-block |
| ckpt 扩展 | train_2.py | checkpoint fields | planned | 能 resume 新模块状态 | 需兼容旧 ckpt |

---

## 4. 训练与评估配置台账

| 配置 ID | backbone | image size | source | batch | steps | cfg | sampling steps | fid samples | 备注 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| CFG-LEGACY-001 | SiT-XL/2 | 256 | DINOv3 | train batch=36 / eval per-proc=1 | train ckpt@160000 | 1.8 | 250 | 50000 | strict ADM 50k；已有实测 |
| CFG-MVP-001 | DiT-B/2 | 256 | DINOv3 | 待填 | 待填 | 待填 | 待填 | 5000 | 开发期默认 |
| CFG-MVP-002 | DiT-B/2 | 256 | self | 待填 | 待填 | 待填 | 待填 | 5000 | external vs self |
| CFG-CONTROL-001 | DiT-B/2 | 256 | DINOv2 | 待填 | 待填 | 待填 | 待填 | 5000 | REPA 历史 control |
| CFG-MAIN-001 | DiT-XL/2 | 256 | DINOv3 | 待填 | 待填 | 待填 | 250 | 50000 | strict 主表 |

---

### 4.1 已有实测实验（legacy，不直接作为新主文证据）

#### 4.1.1 记录原则

以下实验是当前仓库与 `/data` 路径中已经存在的、可确认跑过的结果。

它们的作用是：

- 提供现有系统的真实起点
- 提供已有 checkpoint 与 strict ADM 指标
- 帮助评估工程速度和现有实现强弱

它们**不应直接被当作**新主线 `Stage-wise Spatial Representation Alignment for DiT` 的核心证据，因为：

- backbone 是 `SiT-XL/2`，不是当前主线默认 `DiT`
- source 只有 `DINOv3`
- 结构仍是旧式 REPA 风格，不含新主线的 spatial projector / stage-aware controller / routing / self-alignment

#### 4.1.2 legacy 实验表

| Legacy ID | 实验名称 | backbone | source | checkpoint | 训练配置摘要 | 评估配置摘要 | FID | sFID | Precision | Recall | 状态 | 备注 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| L1 | SiT REPA DINOv3 160k strict ADM | SiT-XL/2 | DINOv3 | `/data/liuchunfa/2026qjx/repa_sit_dinov3_b36_160k/sit_xl2_dinov3_b_enc8_b36_160k/checkpoints/0160000.pt` | batch=36, encoder_depth=8, proj_coeff=0.5, proj_diff_schedule=linear_high_noise, max_train_steps=160000 | strict ADM 50k, 1 GPU, per-proc-batch-size=1, 250 steps, cfg=1.8, mode=sde | 24.0940 | 6.3070 | 0.6443 | 0.5369 | done | 当前唯一确认完成的 strict ADM 实测结果 |

#### 4.1.3 legacy 结果一句话结论

| Legacy ID | 一句话结论 | 对新主线的意义 | 不足 |
| --- | --- | --- | --- |
| L1 | 旧式 `SiT-XL/2 + DINOv3` 路线在 strict ADM 50k 上可得到 `FID 24.09 / Recall 0.5369` | 提供现有工程起点与评估脚手架 | 不能回答 H1/H2/H3，也不能证明新主线有效 |

---

## 5. 核心实验矩阵总表

| 实验 ID | 实验组 | 目的 | backbone | source | 关键变量 | 对照对象 | 当前状态 | 优先级 | 备注 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| A1 | REPA 直接对照链 | 无 alignment 基线 | DiT-B/2 | none | no alignment | A2-A5 | planned | P0 | 起点 |
| A2 | REPA 直接对照链 | 原版 REPA 基线 | DiT-B/2 | DINOv2 或 DINOv3 | MLP + fixed + full-course | A1/A3/A4/A5 | planned | P0 | 必须先跑 |
| A3 | REPA 直接对照链 | 验证 spatial projector | DiT-B/2 | 同 A2 | spatial projector | A2 | planned | P0 | 验证 H1 起点 |
| A4 | REPA 直接对照链 | 验证 routing 增益 | DiT-B/2 | 同 A3 | router on | A3 | planned | P1 | 先不急 |
| A5 | REPA 直接对照链 | 验证 stage-wise 增益 | DiT-B/2 | 同 A4 | fixed early-stop | A4 | planned | P0 | 验证 H2 |
| B1 | 空间结构假设 | global pooled control | DiT-B/2 | DINOv3 | pooled feature | B2-B5 | planned | P0 | H1 关键 |
| B2 | 空间结构假设 | intact spatial map | DiT-B/2 | DINOv3 | spatial intact | B1/B3/B4/B5 | planned | P0 | H1 关键 |
| B3 | 空间结构假设 | patch shuffled | DiT-B/2 | DINOv3 | spatial broken | B2 | planned | P1 | 破坏控制 |
| B4 | 空间结构假设 | token permuted | DiT-B/2 | DINOv3 | spatial broken | B2 | planned | P1 | 破坏控制 |
| B5 | 空间结构假设 | spatial + projector | DiT-B/2 | DINOv3 | spatial projector | B2 | planned | P0 | 主文候选 |
| C1 | stage-wise schedule | full-course | DiT-B/2 | DINOv3 | no early stop | C2-C6 | planned | P0 | H2 |
| C2 | stage-wise schedule | early 10% | DiT-B/2 | DINOv3 | stop@10% | C1 | planned | P0 | H2 |
| C3 | stage-wise schedule | early 25% | DiT-B/2 | DINOv3 | stop@25% | C1 | planned | P0 | H2 |
| C4 | stage-wise schedule | early 50% | DiT-B/2 | DINOv3 | stop@50% | C1 | planned | P1 | H2 |
| C5 | stage-wise schedule | cosine decay | DiT-B/2 | DINOv3 | decay schedule | C1 | planned | P1 | H2 |
| C6 | stage-wise schedule | adaptive stop | DiT-B/2 | DINOv3 | adaptive | C1-C5 | planned | P2 | 附录候选 |
| D1 | source pilot | DINOv2 | DiT-B/2 | DINOv2 | source swap | D2-D5 | planned | P1 | 历史 control |
| D2 | source pilot | DINOv3 | DiT-B/2 | DINOv3 | source swap | D1/D3/D4/D5 | planned | P0 | 默认 external |
| D3 | source pilot | MAE | DiT-B/2 | MAE | source swap | D1/D2/D4/D5 | planned | P2 | 附录候选 |
| D4 | source pilot | self-alignment | DiT-B/2 | self | source swap | D1/D2/D3/D5 | planned | P0 | 主问题关键 |
| D5 | source pilot | SigLIP | DiT-B/2 | SigLIP | source swap | D1-D4 | planned | P2 | 附录候选 |
| E1 | routing 分析 | fixed shallow | DiT-B/2 | DINOv3 | fixed shallow | E4-E7 | planned | P1 | H3 |
| E2 | routing 分析 | fixed middle | DiT-B/2 | DINOv3 | fixed middle | E4-E7 | planned | P1 | H3 |
| E3 | routing 分析 | fixed deep | DiT-B/2 | DINOv3 | fixed deep | E4-E7 | planned | P1 | H3 |
| E4 | routing 分析 | layer-only routing | DiT-B/2 | DINOv3 | layer router | E1-E3 | planned | P1 | H3 |
| E5 | routing 分析 | timestep-only routing | DiT-B/2 | DINOv3 | timestep router | E4/E6 | planned | P2 | H3 补充 |
| E6 | routing 分析 | timestep+layer routing | DiT-B/2 | DINOv3 | joint router | E4/E5 | planned | P1 | H3 主候选 |
| E7 | routing 分析 | top-k sparse routing | DiT-B/2 | DINOv3 | sparse router | E6 | planned | P2 | 附录候选 |
| F1 | 公平主表 | DiT baseline | DiT-XL/2 | none | no align | F2-F4 | planned | P0 | 主表 |
| F2 | 公平主表 | REPA baseline | DiT-XL/2 | DINOv2 | old REPA | F1/F3/F4 | planned | P0 | 主表 |
| F3 | 公平主表 | 强现代 baseline | DiT-XL/2 | 待定 | HASTE 或 REED | F1/F2/F4 | planned | P1 | 至少复现一个 |
| F4 | 公平主表 | Ours | DiT-XL/2 | DINOv3 / self | 最优配置 | F1-F3 | planned | P0 | 主表核心 |

---

## 6. 每组实验详细记录模板

## 6.1 矩阵 A：REPA 直接对照链

### 目标叙述

这组实验只回答一个问题：

**我们的提升是否来自 alignment 结构本身，而不只是“开了 alignment”。**

| 实验 ID | checkpoint | 训练配置 | FID | sFID | Precision | Recall | throughput | wall-clock | 结果摘要 | 结论 | 后续动作 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| A1 | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 | 待定 | 待填 |
| A2 | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 | 待定 | 待填 |
| A3 | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 | 待定 | 待填 |
| A4 | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 | 待定 | 待填 |
| A5 | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 | 待定 | 待填 |

## 6.2 矩阵 B：空间结构假设

### 目标叙述

这组实验用于支撑 `H1`：

**空间结构被破坏后，alignment 效果是否显著下降。**

| 实验 ID | spatial 形式 | checkpoint | FID | sFID | Recall | 结果摘要 | 是否支持 H1 | 备注 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| B1 | global pooled | 待填 | 待填 | 待填 | 待填 | 待填 | 待定 | 待填 |
| B2 | intact spatial | 待填 | 待填 | 待填 | 待填 | 待填 | 待定 | 待填 |
| B3 | patch shuffled | 待填 | 待填 | 待填 | 待填 | 待填 | 待定 | 待填 |
| B4 | token permuted | 待填 | 待填 | 待填 | 待填 | 待填 | 待定 | 待填 |
| B5 | intact + projector | 待填 | 待填 | 待填 | 待填 | 待填 | 待定 | 待填 |

## 6.3 矩阵 C：stage-wise schedule

### 目标叙述

这组实验用于支撑 `H2`：

**alignment 是否主要在训练早期有效。**

| 实验 ID | schedule | checkpoint | FID | Recall | wall-clock to target FID | throughput | 结果摘要 | 是否支持 H2 | 备注 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| C1 | full-course | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 | 待定 | 待填 |
| C2 | early 10% | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 | 待定 | 待填 |
| C3 | early 25% | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 | 待定 | 待填 |
| C4 | early 50% | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 | 待定 | 待填 |
| C5 | cosine decay | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 | 待定 | 待填 |
| C6 | adaptive stop | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 | 待定 | 附录候选 |

## 6.4 矩阵 D：source pilot

### 目标叙述

这组实验回答：

- `DINOv3` 是否值得做主线 default source
- `self-alignment` 是否能替代 external teacher

| 实验 ID | source | checkpoint | FID | sFID | Precision | Recall | 现象 | 主文保留? | 备注 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| D1 | DINOv2 | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 | 待定 | 历史 control |
| D2 | DINOv3 | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 | 待定 | 默认 external |
| D3 | MAE | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 | 待定 | 附录候选 |
| D4 | self | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 | 待定 | 关键反证线 |
| D5 | SigLIP | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 | 待定 | 附录候选 |

## 6.5 矩阵 E：routing 分析

### 目标叙述

这组实验用于支撑 `H3`：

**不同 timestep/block 是否真的偏好不同 source layer。**

| 实验 ID | routing 形式 | checkpoint | FID | Recall | heatmap 路径 | 结果摘要 | 是否支持 H3 | 备注 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| E1 | fixed shallow | 待填 | 待填 | 待填 | 待填 | 待填 | 待定 | 待填 |
| E2 | fixed middle | 待填 | 待填 | 待填 | 待填 | 待填 | 待定 | 待填 |
| E3 | fixed deep | 待填 | 待填 | 待填 | 待填 | 待填 | 待定 | 待填 |
| E4 | layer-only | 待填 | 待填 | 待填 | 待填 | 待填 | 待定 | 待填 |
| E5 | timestep-only | 待填 | 待填 | 待填 | 待填 | 待填 | 待定 | 待填 |
| E6 | timestep+layer | 待填 | 待填 | 待填 | 待填 | 待填 | 待定 | 主候选 |
| E7 | top-k sparse | 待填 | 待填 | 待填 | 待填 | 待填 | 待定 | 附录候选 |

## 6.6 矩阵 F：最终公平主表

### 目标叙述

主表只保留：

- 同协议
- 尽量完整复现
- 能支撑主文 headline claim 的模型

| 实验 ID | 模型 | source | checkpoint | FID | sFID | Precision | Recall | FLOPs | wall-clock | reproduced? | 备注 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| F1 | DiT | none | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 | yes/no | 待填 |
| F2 | REPA | DINOv2 | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 | yes/no | 待填 |
| F3 | HASTE/REED | 待定 | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 | yes/no | 至少一个 |
| F4 | Ours | DINOv3 / self | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 | yes | 主文核心 |

---

## 7. 对比模型选择记录

| 模型 | 年份 | 会议/来源 | 是否进主表 | 作用 | 不进主表的原因/备注 |
| --- | --- | --- | --- | --- | --- |
| DiT | 2023 | ICCV | 是 | 基础 backbone baseline | 必选 |
| SD-DiT | 2024 | CVPR | 可选 | 内生判别表征对照 | 若无法完整复现可放补充 |
| REPA | 2025 | ICLR | 是 | 历史主线起点 | 必选 |
| HASTE | 2025 | NeurIPS | 是/可选 | stage-wise 对照 | 至少复现一个现代强 baseline |
| REED | 2025 | NeurIPS | 是/可选 | flexible guidance 对照 | 与 HASTE 至少保一个 |
| iREPA | 2026 | ICLR | 否 | 文献论证，不一定同协议主表 | 更适合 related work |
| DUPA | 2026 | ICLR | 否 | teacher-free 趋势 | 更适合分析与补充 |
| RAE | 2026 | ICLR | 否 | latent 替代范式 | 不同范式，不宜混主表 |
| SANA-1.5 | 2025 | ICML | 否 | 效率背景工作 | 非同协议，不混主表 |

---

## 8. 日志归档记录

| 运行日期 | 实验 ID | 日志路径 | checkpoint 路径 | 样本路径 | 指标文件路径 | 状态 | 备注 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 2026-04-10 ~ 2026-04-13 | L1 | `/data/liuchunfa/2026qjx/repa_sit_dinov3_b36_160k/sit_xl2_dinov3_b_enc8_b36_160k/checkpoints/offline_eval_adm/0160000_adm50k/launch_adm_20260410_162101.log` | `/data/liuchunfa/2026qjx/repa_sit_dinov3_b36_160k/sit_xl2_dinov3_b_enc8_b36_160k/checkpoints/0160000.pt` | `/data/liuchunfa/2026qjx/repa_sit_dinov3_b36_160k/sit_xl2_dinov3_b_enc8_b36_160k/checkpoints/offline_eval_adm/0160000_adm50k/generated/SiT-XL-2-0160000-size-256-vae-ema-cfg-1.8-seed-0-sde.npz` | `/data/liuchunfa/2026qjx/repa_sit_dinov3_b36_160k/sit_xl2_dinov3_b_enc8_b36_160k/checkpoints/offline_eval_adm/0160000_adm50k/adm_metrics.json` | done | strict ADM 50k 已完成 |
| 待填 | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 |

---

## 9. 失败案例与异常记录

| 日期 | 实验 ID | 异常类型 | 现象 | 初步原因 | 是否复现 | 处理动作 | 结果 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 待填 | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 |

---

## 10. 决策日志

### 10.1 关键决策表

| 日期 | 决策 | 依据实验 | 决策内容 | 影响范围 | 后续动作 |
| --- | --- | --- | --- | --- | --- |
| 2026-04-13 | default external source | D1 vs D2 | 暂定 DINOv3 为主线 external source | 主文 source 选择 | 待实验验证 |
| 2026-04-13 | control source | A2/F2 | 保留 DINOv2 作为 REPA 历史 control | 主表与 related work | 必做 |
| 2026-04-13 | stage controller | C 系列 | 主文默认 fixed early-stop | 主方法定义 | adaptive 只放附录 |
| 2026-04-13 | self source | D4 | self-alignment 为主问题一等公民 | 主文问题定义 | 必做 |
| 2026-04-13 | legacy strict baseline | L1 | 现有唯一已完成 strict ADM 实测来自 `SiT-XL/2 + DINOv3`，只作为工程起点，不直接作为新主文证据 | 台账解释与主文证据边界 | 新主线实验需单独补跑 |

### 10.2 后续待决问题

| 问题 | 由哪个实验决定 | 当前状态 | 决策门槛 |
| --- | --- | --- | --- |
| DINOv3 是否优于 DINOv2 | D1 vs D2 | 待定 | 至少在一个主要质量指标上稳定占优 |
| self-alignment 是否能替代 external source | D2 vs D4 | 待定 | 若 self 更优，主线改写 |
| routing 是否进主文 headline | E 系列 | 待定 | 必须有清晰 heatmap + 数值增益 |
| adaptive stop 是否进附录 | C6 | 待定 | 必须有附加收益且无明显泄漏风险 |

---

## 11. 主文写作映射

| 论文主张 | 对应实验 | 最低要求 | 当前状态 |
| --- | --- | --- | --- |
| spatial preservation 是关键 | B 系列 + A3 | spatial intact 显著优于 global/shuffled | 当前无直接实证，legacy L1 不能支撑 |
| stage-wise alignment 优于 full-course | C 系列 + A5 | 至少一种 early-stop 优于 full-course | 当前无直接实证，legacy L1 不能支撑 |
| routing 比固定 matching 更合理 | E 系列 + A4 | heatmap + 数值增益 | 当前无直接实证 |
| external teacher 不是默认必须 | D 系列 | self 进入主文对照 | 当前无直接实证 |
| 我们比 REPA 有结构改进 | A2 vs A5, F2 vs F4 | DINOv2 control 上仍有增益 | 当前无直接实证，legacy L1 只给出现有起点 |

---

## 12. 当前推荐填表顺序

为了减少管理混乱，建议每次只按下面顺序更新本文件：

1. 先更新“设计版本表”
2. 再更新“训练与评估配置台账”
3. 再填写对应实验行
4. 结果出来后更新“关键假设跟踪”
5. 最后写“决策日志”

如果跳过这一步，最容易出现的问题是：

- 跑完实验不知道对应哪个设计版本
- 指标记录缺上下文
- 论文写作时无法回溯为什么做过某个决定
