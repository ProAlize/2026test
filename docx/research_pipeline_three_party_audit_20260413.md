# Research-Pipeline 三方审计结果与 S3A 设计说明

日期：2026-04-13  
模式：`research-pipeline`（nightmare reviewer标准）

---

> 更新注记（2026-04-14）  
> 为避免实现口径混淆：`t-aware` 与 `SASA` 已冻结为 REPA-faithful projector（3层SiLU, dim=2048），`S3A` 继续使用外置可替换 adapter/projector 分支。

---

## 1) 三方审计评分汇总

## 审计A（Novelty / Positioning）

- 新颖性：`4.6/10`
- 问题重要性：`8.1/10`
- 差异清晰度：`6.0/10`
- 可投稿说服力：`2.0/10`
- 总分：`4.2/10`

判定：问题选得对，但当前更像“合理重组”，不是“不可替代新机制证据”闭环。  
关键证据见：[idea_s3a_with_audit_feedback.md:53](/home/liuchunfa/2026qjx/2026test/docx/idea_s3a_with_audit_feedback.md:53)、[idea_s3a_with_audit_feedback.md:107](/home/liuchunfa/2026qjx/2026test/docx/idea_s3a_with_audit_feedback.md:107)、[paper.md:34](/home/liuchunfa/2026qjx/2026test/docx/paper.md:34)

## 审计B（Empirical / Experimental）

- 实验充分性：`3.0/10`
- 对照公平性：`3.0/10`
- 评测可信度：`1.0/10`
- 统计稳健性：`2.0/10`
- 总分：`2.3/10`

判定：最大的硬伤是评测可信度与统计功效，当前状态离可投稿证据链很远。  
关键证据见：[fid_root_cause_report_20260413.md:49](/home/liuchunfa/2026qjx/2026test/docx/fid_root_cause_report_20260413.md:49)、[exp.md:38](/home/liuchunfa/2026qjx/2026test/docx/exp.md:38)、[exp_registry_neurips2026.csv:2](/home/liuchunfa/2026qjx/2026test/ledger/exp_registry_neurips2026.csv:2)

## 审计C（Engineering / Reproducibility）

- 实现完备性：`4/10`
- 可复现实操性：`3/10`
- 风险控制：`2/10`
- 工程质量：`6/10`
- 总分：`4/10`

判定：工程质量中等，但复现闭环与风险控制不足，会直接拖累论文可信度。  
关键证据见：[train_2.py:641](/home/liuchunfa/2026qjx/2026test/train_2.py:641)、[eval_fid_ddp.py:121](/home/liuchunfa/2026qjx/2026test/eval_fid_ddp.py:121)、[train_sasa.py:144](/home/liuchunfa/2026qjx/2026test/train_sasa.py:144)

## 综合评分（简单平均）

- `(4.2 + 2.3 + 4.0) / 3 = 3.5/10`

结论：**方向有潜力，但目前是“提案态”，不是“可过审稿态”。**

---

## 2) 综合诊断（为什么会被审成 naive）

1. 评测链路未冻结，导致“结果可信度”先天不足。  
2. 方法叙事虽完整，但缺“不可替代的新机制证据”。  
3. 主文要求与现有证据不匹配（多seed、公平对照、强baseline复现未闭环）。  
4. 工程上 resume / deterministic / 供应链安全 /采样精确计数缺口会放大审稿质疑。

---

## 3) S3A 详细设计（你要的“先 idea 后实验”版本）

## 3.1 研究问题（不是“换 projector”）

S3A 针对的问题是：

- **What**：对齐哪类表征最有效（external vs self vs 混合）
- **When**：在哪些训练阶段与扩散阶段对齐最有效
- **How long**：何时应该停止/局部停止对齐

这比“改个投影器结构”高一层级。

## 3.2 模块化定义

### M1. Source Reliability Router

目标：在每个 `(s, t, l)` 动态分配 source 权重。  
输入：`h_l`, `{z_l^k}`, `z_l^self`, `s`, `t`。  
输出：`alpha_k(s,t,l)`, `alpha_self(s,t,l)`，并满足归一化。

融合目标：

`z_l^* = sum_k alpha_k z_l^k + alpha_self z_l^self`

意义：把 external/self 的“谁更好”变成可学习、可解释曲线，而不是先验判断。

### M2. Spatially Faithful Adapter

主干：conv projector + spatial norm（继承 iREPA 经验）。  
旁路：轻量 token MLP（增强 source shift 下鲁棒性）。

输出：`p_l(h_l)`。

### M3. Holistic Alignment Loss

`L_align = lambda_f L_feat + lambda_a L_attn + lambda_s L_spatial`

- `L_feat`：token feature 对齐
- `L_attn`：attention map 对齐
- `L_spatial`：局部结构保持约束

意义：避免“语义对齐了，但几何/纹理被破坏”。

### M4. 3D Curriculum + Stop Gate

`w(s,t,l) = g_phase(s) * g_time(t) * g_layer(l)`

- `g_phase`：前强后弱
- `g_time`：扩散时刻加权
- `g_layer`：层组重要性

Stop Gate：
- 监控窗口化收益 `Delta_metric`（FID proxy + Recall proxy + gradient conflict）
- 连续 K 次无收益则关闭对应 `(source, layer-group)` 对齐
- 支持“局部停”，而不是全局一刀切

## 3.3 与现有工作的可辩护差异

- 相对 REPA：从固定 teacher/固定对齐升级到动态 source + 3D 控制 + selective stop。
- 相对 iREPA：不把 spatial projector 当主创新，而是当必要条件；主创新在控制与路由。
- 相对 HASTE：在 holistic + stop 基础上加入 external/self 动态融合与 layer-group 维度。
- 相对 DUPA：不是替换 external，而是统一 external/self 并学习何时用谁。

---

## 4) 最小充分证据包（投稿前）

必须先完成下面 8 组，才能支撑 S3A 不是 naive：

1. No alignment
2. REPA static
3. + Spatial adapter only
4. + Holistic only
5. + 3D curriculum only
6. + Stop gate only
7. Full S3A（external-only）
8. Full S3A（external+self dynamic）

同时必须给 4 张机制图：
- `(s,t,l)` 路由热力图
- stop gate 触发轨迹
- source 权重演化
- failure cases

---

## 5) 从 3.5/10 提到可投稿区间的“第一周动作”

1. 冻结评测口径（单一实现、精确样本计数、禁止目录污染）。
2. 给 3 个训练脚本加 `--resume`，可恢复 `model/ema/opt/repa_projector/train_steps/RNG`。
3. 主文关键实验升到 `>=3 seeds`，并输出均值+方差/CI。
4. XL 主表至少补 1 个强现代 baseline（HASTE 或 REED）同协议复现。

---

## 6) 最终一句话反馈

你现在的 idea 方向是对的；真正风险不在“想法太弱”，而在“证据链太松”。  
S3A 可以成为强稿主线，但前提是先把评测与复现闭环补齐，再用最小充分证据证明它不是模块拼装。
