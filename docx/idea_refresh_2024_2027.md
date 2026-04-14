# Ada-REPA Idea Refresh (2024-2027)  
As of **2026-04-13**

## 0) Scope & Reality Check

- 本文聚焦你当前项目主线：**DiT 表征对齐 / 阶段性对齐 / source 选择 / spatial faithful transfer**。
- 时间范围按你要求覆盖 2024-2027；但截至 **2026-04-13**，2027 主会结果大多尚未公开。本文把 2027 作为“待跟踪窗口”，不做伪确定结论。

---

## 1) Top-Venue Evidence Map (2024-2026)

## A. 已确认顶会（强证据）

1. **REPA** — ICLR 2025 Oral  
   - 核心：外部视觉表征对齐可显著加速 DiT/SiT 训练。
   - 链接：https://openreview.net/forum?id=DJSZGGZYVi

2. **Guidance Interval** — NeurIPS 2024 Main  
   - 核心：CFG 不应全程施加，中段区间 guidance 可同时提质+提速。
   - 链接：https://proceedings.neurips.cc/paper_files/paper/2024/hash/dd540e1c8d26687d56d296e64d35949f-Abstract-Conference.html

3. **DiffiT** — ECCV 2024  
   - 核心：时序感知注意力 + ViT denoiser 设计，强调 backbone/效率共同优化。
   - 链接：https://www.ecva.net/papers/eccv_2024/papers_ECCV/html/1231_ECCV_2024_paper.php

4. **Diff-MoE** — ICML 2025  
   - 核心：time-aware + space-adaptive experts，动态分配计算到时间阶段与空间 token。
   - 链接：https://proceedings.mlr.press/v267/cheng25d.html

5. **HASTE** — NeurIPS 2025 Poster  
   - 核心：holistic alignment（feature+attention）+ stage-wise termination。
   - 链接：https://openreview.net/forum?id=HK96GI5s7G

6. **REG** — NeurIPS 2025 Oral  
   - 核心：把语义 token 与生成过程“纠缠”，减轻“仅训练时 external target”脱节。
   - 链接：https://openreview.net/forum?id=koEALFNBj1

7. **DUPA** — ICLR 2026 Poster  
   - 核心：无外部 teacher 的双路径自对齐，降低数据分布不匹配风险。
   - 链接：https://openreview.net/forum?id=ALpn1nQj5R

8. **iREPA** — ICLR 2026 Poster  
   - 核心：证明 spatial structure 比 global semantic accuracy 更关键；conv projector + spatial norm。
   - 链接：https://openreview.net/forum?id=y0UxFtXqXf

9. **SRA**（No Other Representation Component Is Needed）— ICLR 2026 Poster  
   - 核心：主张 DiT 本身可提供有效表征指导，强调“内生表示”路径。
   - 链接：https://openreview.net/forum?id=ds5w2xth93

10. **U-REPA** — NeurIPS 2025 Poster  
    - 核心：把 REPA 从 DiT/SiT 扩到 U-Net 并提出鲁棒对齐范式。
    - 链接：https://openreview.net/forum?id=im3FJ6quii

11. **Learning Diffusion Models with Flexible Representation Guidance**（你文档中的 REED 对应来源）— NeurIPS 2025 Poster  
    - 核心：按条件动态选择/融合表征指导，减少“单一 source 固定绑定”。
    - 链接：https://openreview.net/forum?id=cIGfKdfy3N

## B. 强相关 preprint（值得纳入，但需标注“未顶会确认”）

1. **REPA-E**（arXiv 2025）  
   - 核心：端到端联合调 tokenizer(VAE)+diffusion，通过 alignment loss 才可稳定。
   - 链接：https://arxiv.org/abs/2504.10483

2. **RAE-DiT**（arXiv 2025）  
   - 核心：用表示型编码器+轻解码器替代 VAE，减少对额外 alignment loss 的依赖。
   - 链接：https://arxiv.org/abs/2510.11690

3. **PixelREPA (JiT)**（arXiv 2026）  
   - 核心：指出 REPA 在 pixel-space JiT 可能失效，提出 target transform + masked adapter。
   - 链接：https://arxiv.org/abs/2603.14366

4. **SRA 2**（arXiv 2026）  
   - 核心：在 SRA 路线下继续强化“内生对齐”与训练稳定性细节。
   - 链接：https://arxiv.org/abs/2601.17830

## C. 关于“顶级期刊（2024-2027）”

- 截至 2026-04，**DiT 表征对齐这个子方向的高影响结果主要仍在 ICLR/NeurIPS/ICML/CVPR/ECCV 与 arXiv**。
- 顶级期刊（如 TPAMI/IJCV/NMI）在该细分议题尚未形成与上述会议同等密度的“可直接复用方法族”。
- 建议：主方法论以顶会为依据，期刊只作为“补充背景与跨域证据”。

---

## 2) 对你项目的“更 sophisticated”升级路线

你当前主线（spatial projector + routing + stage schedule）是对的，但可升级为下面四层结构：

## Layer-1: Alignment Target 选择机制（先决定“对齐什么”）

1. **空间结构优先评分器**（iREPA启发）
   - 不再按 encoder ImageNet acc 排序 source。
   - 先计算 spatial self-similarity preservation 指标，按空间保持度挑 source shortlist。

2. **Dual-source 组合（external + self）**
   - 前期 external 提供强先验；中后期增加 self/dual-path 约束（DUPA思路）缓解 teacher straitjacket。

## Layer-2: Alignment Form（再决定“怎么对齐”）

1. **Holistic 对齐默认化**（HASTE）
   - 不是只做 token cosine；改为 feature + attention 的双通道对齐。

2. **Projector 升级为 Spatial Adapter**
   - baseline: conv projector + spatial norm（iREPA）
   - advanced: masked transformer adapter（PixelREPA）用于防 shortcut regression。

## Layer-3: Controller（再决定“何时何地对齐”）

1. **二维控制器升级为三维控制器**
   - 你已有：train-phase × diffusion-timestep
   - 新增：layer-group 维度（来自 Diff-MoE 的 time/space heterogeneity 思路）

2. **One-shot termination + soft fadeout 双策略**
   - 对标 HASTE：保留 one-shot
   - 额外加 soft-fade（5k~20k steps）防止 abrupt switch 导致震荡。

## Layer-4: Inference Coupling（训练和采样联动）

1. **固定报告 guidance interval**（NeurIPS 2024）
   - 不再只给固定 CFG scale；必须报告 guidance 起止区间。

2. **对齐退出点与 guidance 区间联调**
   - 假设：更早退出 alignment 可能需要更窄 guidance 区间；这是可发表的机制实验点。

---

## 3) 推荐的“主线版本”定义（便于投稿叙事）

## Version A（稳健主线，最容易出结果）

- iREPA式空间投影（conv + spatial norm）
- HASTE式阶段终止（feature+attention）
- 你的二维控制器（train-phase × timestep）
- 统一 guidance interval 评估

一句话贡献：**“Spatially-faithful holistic alignment with stage-aware control.”**

## Version B（进阶版，冲更强 novelty）

- 在 Version A 基础上加入 dual-path self-alignment（DUPA风格）
- 外部 teacher 与 self 信号做动态权重融合

一句话贡献：**“From external alignment to hybrid external-self curricula.”**

---

## 4) 台账（Ledger）重构模板

你现在缺的不是单个实验，而是“可追责的研究账本系统”。建议拆成 6 本：

## 4.1 `ledger/lit_ledger.csv`（文献台账）

字段：
- `paper_id`
- `venue`
- `year`
- `status`（top_conf / preprint / workshop）
- `core_claim`
- `relevance_to_project`
- `adopted_component`
- `risk_if_adopt`

## 4.2 `ledger/exp_registry.csv`（实验注册台账）

字段：
- `exp_id`
- `hypothesis`
- `factor_changed`
- `fixed_factors`
- `dataset`
- `model_size`
- `seed_list`
- `budget_gpu_hours`
- `success_metric`
- `kill_criteria`

## 4.3 `ledger/run_log.csv`（运行台账）

字段：
- `run_id`
- `exp_id`
- `git_commit`
- `config_path`
- `start_time`
- `end_time`
- `machine`
- `gpu_type`
- `failure_type`
- `artifact_path`

## 4.4 `ledger/result_board.csv`（结果总账）

字段：
- `run_id`
- `fid`
- `sfid`
- `precision`
- `recall`
- `throughput`
- `wallclock_to_target_fid`
- `variance_over_seeds`
- `is_main_table_eligible`

## 4.5 `ledger/risk_register.md`（风险台账）

每条风险固定格式：
- `Risk`
- `Trigger`
- `Impact`
- `Owner`
- `Mitigation`
- `Decision Deadline`

## 4.6 `ledger/decision_log.md`（决策台账）

每次关键决策记录：
- `Decision`
- `Evidence`
- `Alternatives Rejected`
- `Who Approved`
- `Date`
- `Rollback Plan`

---

## 5) 你下一步最值得做的 14 天计划

1. 建立 6 本台账（先空表+字段）
2. 完成 Version A 最小闭环：
   - baseline REPA
   - + iREPA
   - + HASTE terminate
   - + guidance interval
3. 每个设置先跑 B/2 小规模多 seed，过门槛再上 XL/2
4. 写一页“负结果规则”：任何新模块若两轮未带来稳定收益，自动降级到附录

---

## 6) 目前不确定点（需要你确认）

1. 你目标投稿窗口是 **NeurIPS 2026** 还是 **ICLR 2027**？
2. 你可稳定使用的预算是“单个 8xA100 节点 * 几周”？
3. 你更希望先冲 **最稳的 Version A**，还是直接做 **Version B（更激进）**？
4. 你主战场是否固定为 ImageNet-256，还是要并行 512 作为主表？

> 这 4 个答案会直接决定我下一步给你的“可执行实验矩阵”和优先级剪枝策略。
