# NeurIPS 2026 完整实验矩阵（ImageNet-256）

更新时间：2026-04-13（CST）  
目标投稿：NeurIPS 2026 Main Track（Abstract: 2026-05-04 AOE；Full: 2026-05-06 AOE）

## 0. 现实约束与策略

- 你当前可用算力：
  - `A6000 4卡 * 2 台`（共 8 GPU）
  - `RTX 5880 Ada 4卡 * 1 台`（共 4 GPU）
  - `H800 8卡 * 2 台`（共 16 GPU）
- 截至 2026-04-13 距 NeurIPS full paper 截止（2026-05-06）约 23 天。
- 因此矩阵分为两层：
  1. `Core Matrix (NeurIPS 必做)`：保证主结论+主表可提交。
  2. `Extended Matrix (Journal/Camera-ready 扩展)`：用于 PR/TPAMI/IJCV 扩展版。

---

## 1. 统一协议（全实验强制）

- 数据：ImageNet-1K，`256x256`，class-conditional。
- 主干：`DiT-B/2`（机制）+ `DiT-XL/2`（主表）。
- 训练：`AdamW, lr=1e-4, global batch=256, EMA=0.9999`。
- 评测：
  - 快速代理：`FID-5k`（开发阶段）
  - 报告指标：`FID-50k, sFID, Precision, Recall, throughput, wall-clock, GPU hours`
- 采样默认：`steps=250, cfg=1.5`，并做 guidance interval 补充。
- 种子：机制实验 `1~2 seed`；主表实验 `>=3 seed`。

---

## 2. 资源池分工

- `Pool-H`（H800*16）：所有 `DiT-XL/2` 训练与最终 50k 评测。
- `Pool-A`（A6000*8）：`DiT-B/2` 消融、source 对比、controller 实验。
- `Pool-5880`（5880*4）：评测队列（FID/sFID/PR）、日志整理、失败重跑。

并行策略：
- `Pool-H` 同时跑 2 个 8-GPU 任务。
- `Pool-A` 同时跑 2 个 4-GPU 任务。
- `Pool-5880` 常驻评测服务，避免训练 GPU 被评测抢占。

---

## 3. Core Matrix（NeurIPS 必做，建议 2026-04-14 ~ 2026-05-02 完成）

说明：下表是“必须跑完”的最小完备集合，共 20 组（含多 seed 展开后约 32~40 runs）。

| ID | 组别 | 配置变化 | 模型 | Seed | 目标 | 资源 | 估算GPUh | 主文用途 |
|---|---|---|---|---|---|---|---:|---|
| C01 | 校准 | 无对齐 DiT baseline（现有训练脚本） | B/2 | 1 | 校准速度与损失曲线 | Pool-A | 80 | 训练基线 |
| C02 | 校准 | REPA-faithful static projector（3层SiLU, dim=2048） | B/2 | 1 | 校准对齐分支开销 | Pool-A | 100 | 对照基线 |
| C03 | 评测校准 | C01 评测，FID-5k/FID-50k差异 | B/2 | - | 建立 proxy->final 映射 | Pool-5880 | 12 | 方法学附录 |
| C04 | 评测校准 | evaluator 一致性（单口径固定） | B/2 | - | 锁定唯一评测口径 | Pool-5880 | 8 | 可复现声明 |
| A01 | 组件 | No alignment | B/2 | 2 | 绝对 baseline | Pool-A | 160 | 主表/消融 |
| A02 | 组件 | Original REPA（3层MLP+fixed+full） | B/2 | 2 | 复现 REPA 参照 | Pool-A | 200 | 主表/消融 |
| A03 | 组件 | iREPA-style spatial projector only | B/2 | 2 | 验证空间投影增益 | Pool-A | 200 | 消融核心 |
| A04 | 组件 | + routing（无 early stop） | B/2 | 2 | 验证路由独立收益 | Pool-A | 220 | 消融核心 |
| A05 | 组件 | + stage termination（完整 Ours-A） | B/2 | 2 | 验证完整方案 | Pool-A | 220 | 主方法 |
| A06 | 组件 | Ours-A 去掉 attention alignment | B/2 | 2 | 验证 holistic 必要性 | Pool-A | 220 | 机制证据 |
| S01 | Source | Ours-A + DINOv2 | B/2 | 2 | source 基线 | Pool-A | 220 | 主文 source 表 |
| S02 | Source | Ours-A + DINOv3 | B/2 | 2 | 强语义 teacher 对比 | Pool-A | 220 | 主文 source 表 |
| S03 | Source | Ours-A + MAE | B/2 | 2 | 结构型 teacher 对比 | Pool-A | 220 | 主文 source 表 |
| S04 | Source | Ours-A + SigLIP | B/2 | 2 | 跨模态表征对比 | Pool-A | 220 | 主文 source 表 |
| S05 | Source | Ours-A + self-alignment (EMA) | B/2 | 2 | external 必要性检验 | Pool-A | 220 | 主文 source 表 |
| T01 | 时机 | Ours-A full-course | B/2 | 2 | 对齐时机基线 | Pool-A | 220 | 时机图 |
| T02 | 时机 | Ours-A early-25% | B/2 | 2 | 早停收益 | Pool-A | 200 | 时机图 |
| T03 | 时机 | Ours-A adaptive stop | B/2 | 2 | 自适应停用效果 | Pool-A | 210 | 时机图 |
| T04 | 时机 | Ours-A + diff-timestep weighting on/off | B/2 | 2 | 二维控制有效性 | Pool-A | 220 | 方法核心证据 |
| I01 | 采样联动 | Ours-A: guidance interval vs full CFG | B/2 | 2 | 训练-采样联动证据 | Pool-5880 | 20 | 主文机制图 |

### 3.1 XL 主表最小集合（NeurIPS 必做）

| ID | 组别 | 配置变化 | 模型 | Seed | 目标 | 资源 | 估算GPUh | 主文用途 |
|---|---|---|---|---|---|---|---:|---|
| X01 | 主表 | DiT 无对齐 | XL/2 | 3 | 主表 baseline | Pool-H | 2,400 | 主表 |
| X02 | 主表 | REPA-style baseline | XL/2 | 3 | 与 REPA 直接对照 | Pool-H | 2,700 | 主表 |
| X03 | 主表 | Ours-A 最佳 source（来自 S01-S05） | XL/2 | 3 | 主方法结果 | Pool-H | 2,700 | 主表 |
| X04 | 主表补充 | Ours-A + second-best source | XL/2 | 2 | source 稳健性 | Pool-H | 1,800 | 主表/附录 |
| X05 | 采样 | X03 的 CFG grid + guidance interval | XL/2 | - | 最优采样协议 | Pool-5880 | 40 | 报告协议 |
| X06 | 复核 | X01/X02/X03 统一 50k 复评测 | XL/2 | - | 口径一致性 | Pool-5880 | 30 | 可复现声明 |

> Core Matrix 合计预算（含多 seed）：约 `12k~14k GPUh`。  
> 你现有 28 GPU 并行下，理论上可在 2~3 周内跑完“投稿必须部分”，但前提是评测队列稳定且失败重跑率低于 15%。

---

## 4. Extended Matrix（期刊扩展，NeurIPS 后继续）

说明：用于 Pattern Recognition / TPAMI / IJCV 扩展版，优先补“机制深度 + 鲁棒性 + 泛化”。

| ID | 组别 | 配置变化 | 模型 | Seed | 资源 | 估算GPUh | 用途 |
|---|---|---|---|---|---|---:|---|
| E01 | 机制 | Ours-A + dual-path self/external（Ours-B） | B/2 | 3 | Pool-A | 360 | 新颖性强化 |
| E02 | 机制 | Ours-B vs Ours-A | XL/2 | 3 | Pool-H | 2,700 | 期刊主结论 |
| E03 | Projector | Conv projector vs masked adapter | B/2 | 3 | Pool-A | 360 | projector 深挖 |
| E04 | Projector | E03 最优配置上 XL/2 | XL/2 | 2 | Pool-H | 1,800 | 期刊补强 |
| E05 | 控制器 | 2D vs 3D controller（加 layer-group） | B/2 | 3 | Pool-A | 420 | 机制创新 |
| E06 | 控制器 | E05 最优配置上 XL/2 | XL/2 | 2 | Pool-H | 1,800 | 期刊主图 |
| E07 | Loss | cosine / mse / hybrid 对齐 | B/2 | 3 | Pool-A | 360 | 排除伪贡献 |
| E08 | Source鲁棒 | source dropout / source corruption | B/2 | 3 | Pool-A | 420 | 鲁棒性 |
| E09 | 训练规模 | 100k/200k/400k scaling law | B/2 | 2 | Pool-A | 320 | 规模规律 |
| E10 | 主战场外 | ImageNet-512（仅最佳3配置） | XL/2 | 2 | Pool-H | 2,400 | 泛化扩展 |

---

## 5. 执行排程（绝对日期）

- `2026-04-14 ~ 2026-04-16`：C01-C04（校准+评测口径锁定）
- `2026-04-16 ~ 2026-04-23`：A01-A06、S01-S05（B/2 主体）
- `2026-04-23 ~ 2026-04-27`：T01-T04、I01（控制器与联动）
- `2026-04-24 ~ 2026-05-02`：X01-X06（XL 主表最小集合）
- `2026-05-03 ~ 2026-05-05`：主文数字冻结、图表复核、文本定稿

NeurIPS 关键时间：
- Abstract 截止：`2026-05-04 (AOE)`
- Full Paper 截止：`2026-05-06 (AOE)`

---

## 6. 主文与附录映射（防止实验做了但写不进论文）

- 主文必须：`A01 A02 A05 S01~S05 T01~T03 X01~X03 X05`
- 主文可选：`A03 A04 A06 T04 X04 X06`
- 附录/期刊：`E01~E10`

---

## 7. 失败转向阈值（强制执行）

1. 若 `S05(self)` 在 B/2 与 XL/2 均不劣于最优 external source（差值 < 0.2 FID），主叙事转向“external 是否必要”。
2. 若 `A03/A04` 对 `A02` 提升不稳定（3 seeds 中有 2 seed 无提升），projector 不作为 headline novelty。
3. 若 `T03(adaptive stop)` 不能同时改善 `wall-clock` 与最终 FID，降级为效率补充结论。

---

## 8. 台账绑定规则（必须）

- 每个 ID 进入 `ledger/exp_registry_neurips2026.csv`
- 每次训练 run 进入 `ledger/run_log.csv`
- 每次评测进入 `ledger/result_board.csv`
- 每次方案取舍进入 `ledger/decision_log.md`

没有台账记录的结果，不进入主文图表。
