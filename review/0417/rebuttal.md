# Rebuttal: Response to S3A v3 Multi-Agent Audit (2026-04-17)

**审计文件**: `RESEARCH_REVIEW_S3A_V3_MULTIAGENT_20260417.md`
**回应方**: S3A 设计团队
**日期**: 2026-04-17

---

## 0. 核心分歧：审计使用了错误版本的实验数据

**事实**：审计报告的所有运行证据来自 commit `0b466ad`（v2），而非 commit `9dcf858`（v3）。

v2 和 v3 的架构差异是根本性的：

| 维度 | v2 (0b466ad) | v3 (9dcf858) |
|------|-------------|-------------|
| Self-source 层 | 同层（student layer l → EMA layer l） | 跨层（student 6 → EMA 20, offset=14） |
| Self-source 时间步 | 同时间步（t = t） | 跨时间步（t_ema = max(0, t-k), k∈[0,200]） |
| EMA adapter 权重来源 | 与 student adapter 共享 key | 独立 key，从 closest student adapter 初始化 |
| 恒等退化 | $\text{pred} \approx \text{ema\_proj}$（已证实） | 被三重对称性破坏消除（待验证） |

审计结论中"dual_alive=4.55%"、"synergy_margin=-0.4"等数据全部来自 v2 run，**不能作为评判 v3 的证据**。

**我们接受**：v3 尚无实验数据，双源是否成功需要实验验证。
**我们不接受**：将 v2 的失败数据作为 v3 已失败的证据。

---

## 1. 逐条回应审计发现

### 对 P0-1 "控制目标错位"

**审计观点**：控制器优化"选更好单源"，不是"维持协同双源"。synergy_margin 没进入控制闭环。

**我们的回应**：
- **部分同意**。当前控制器的 reopen 条件确实只看 utility EMA，不看 synergy。这是一个有效的改进方向。
- **但这是第二优先级**。首先需要验证 v3 的跨层跨时间步是否使 self-source 提供了非 trivial 的监督信号（loss_self_only >> 0）。如果 loss_self_only 仍然 ≈ 0，那改控制器也无济于事。
- **计划**：v3 短验证后，如果 self-source 信号有效但 Router 仍偏向单源，我们会实施 synergy-aware reopen。

### 对 P0-2 "缓解机制是 reset 不是 recovery"

**审计观点**：mitigation 后一刀切重开 self，容易再塌缩。

**我们的回应**：
- **同意**。当前的 mitigation → full-reopen 确实缺少渐进过渡。
- **但在 v3 上下文中，情况可能根本不同**。v2 的塌缩是因为 pred ≈ ema_proj（loss_self ≈ 0），self 有绝对的 loss 优势。v3 的 self 不再有这个绝对优势——如果 loss_self ≈ 0.5 而 loss_dino ≈ 0.8，Router 的偏好差距小得多，reopen 后不一定会再次快速塌缩。
- **计划**：先观察 v3 的 reopen 行为，如果确实需要则实施 gradual ramp-up。

### 对 P1-1 "floor 只防零不防失活"

**审计观点**：alpha_dino 可以贴在 floor 上（名义双源，实质单源）。

**我们的回应**：
- **同意这是 v2 的问题**。v2 中 DINO loss >> self loss（1.3 vs 0.03），Router 有极强动机压低 DINO。
- **v3 预期缓解**。如果 v3 使 loss_self ≈ 0.5（而非 0.03），Router 的最优策略不再是极端偏向 self——一个 balanced 的融合可能比纯 self 更好（synergy > 0）。这正是"跨层跨时间步打破恒等退化"的目标。

### 对 P1-2 "warmup 退出是冷切换"

**审计观点**：缺少分阶段 reopen + 协同验收。

**我们的回应**：
- **部分同意**。warmup 结束时 Router 被重置为均匀（softmax([0,0])=[0.5,0.5]），然后自由学习。如果 self 很快展现优势，Router 会快速偏向 self——但这次偏向应该是基于真实的信号差异（v3），而非恒等退化（v1/v2）。
- **如果 v3 实验显示偏向仍然过快**：我们接受实施 two-stage unlock（先观察 synergy 再完全开放）。

### 对第 5 节 "为什么 cross-layer + cross-timestep 仍不够"

**审计观点**：解决了 identity shortcut 但没改控制器目标函数。

**我们的回应**：
- **这是审计中最有洞察力的批评**。我们完全同意控制器是第二阶段的改进目标。
- **但请注意因果顺序**：先验证 signal quality（v3 是否产生有意义的 self-source），再优化 control policy。如果 signal 仍然 trivial，控制器改动是徒劳的。
- **数学论证**：v3 打破了恒等退化的三个维度：
  1. 层差：block_6 output 和 block_20 output 是本质不同的特征层级
  2. 时间步差：高噪声 x_t 和低噪声 x_{t-k} 的信噪比不同
  3. Adapter 差：student_adapters["6"] 和 ema_adapters["20"] 是不同参数
  这使得 loss_self 不可能 ≈ 0，预期 ≈ 0.3-1.0。

---

## 2. 对审计评分的回应

### 审计矩阵

| 主张 | 审计评 | 我们的评 | 理由 |
|------|--------|---------|------|
| v3 去掉了同层同时间步捷径 | 支持 | **同意** | |
| v3 已建立稳定双源协同 | 不支持 | **无法判断** | 无 v3 实验数据 |
| v3 已形成可靠 routing | 不支持 | **无法判断** | 无 v3 实验数据 |
| v3 已建立 anti-collapse contract | 支持 | **同意** | |

**关键分歧**：审计说"不支持"，我们说"无法判断"。不支持需要反面证据；无法判断是因为证据缺失。它们不等价。

---

## 3. 我们接受的改进

| 来源 | 建议 | 接受？ | 时机 |
|------|------|--------|------|
| P0-1 | synergy 进入控制闭环 | ✅ | v3 验证后如需要 |
| P0-2 | mitigation 改为 gradual recovery | ✅ | v3 验证后如需要 |
| P1-2 | warmup 退出 two-stage unlock | ✅ | v3 验证后如需要 |
| P2-1 | dual_alive 定义更严格 | ✅ | 可以立即做 |
| P2-2 | 增加 "inconclusive" 档 | ✅ | 可以立即做 |
| §8 | 验证窗口扩展到 45k | ✅ | 已接受 |

---

## 4. 行动计划

### 立即（本周）
1. 按 `RUN_V3_VERIFY.md` 运行 v3 短验证（45k steps, seed 0）
2. 观察 step 25k+ 的 `loss_self_only`、`dual_alive`、`synergy_margin`
3. 结果判定：
   - `loss_self_only ≈ 0.3-1.0` → signal quality 验证通过
   - `dual_alive > 50%` 且 `synergy_margin > 0` → 双源成功，无需改控制器
   - `loss_self_only > 0.1` 但 `dual_alive < 30%` → signal 有效但控制器需改
   - `loss_self_only < 0.1` → v3 失败，转 DINO-only 路线

### 第二阶段（如 signal 有效但控制器不足）
4. 实施 synergy-aware reopen（reopen 条件加 synergy_margin > 0）
5. 实施 gradual ramp-up after mitigation
6. 再次短验证（45k, 2 seeds）

### 第三阶段（如双源成功）
7. 完整 400k run（3 seeds）
8. DINO-only 对照组
9. FID 评估

---

## 5. 对未来审计的请求

1. **区分版本**：审计报告应明确标注数据来自哪个 commit，不要将旧版本数据套用到新版本结论
2. **区分"不支持"和"无法判断"**：前者需要反面证据，后者是证据缺失
3. **承认架构改动的量级**：v2→v3 不是参数调优，是自源信号的根本重构（同层→跨层，同噪声→跨噪声）
