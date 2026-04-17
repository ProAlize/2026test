# S3A v3 多代理深度审计报告（2026-04-17）

## 1. 审计目标与范围
本审计回答两个问题：
1. 当前 v3 方案是否仍存在“双源失活”问题。
2. 若存在，问题是实现 bug、监控口径问题，还是目标函数/控制器层面的结构性问题。

审计输入材料：
- 设计与运行协议：`RUN_V3_VERIFY.md`
- 核心实现：`train_s3a_multisource_dinov2.py`、`run_s3a_multisource_dinov2.sh`
- 最新运行证据：`reports/s3a_bz128_400k_20260417/TRAINING_NOTE_S3A_BZ128_400K_20260417.md`
- 原始日志：`reports/s3a_bz128_400k_20260417/logs/s3a_bz128_400k_v4.log`

并行外部审查（`gpt-5.4 xhigh`）：
- Agent A（机制/代码路径）：`019d9a8f-cdcc-7b40-ac1f-23ea2162038a`
- Agent B（设计主张一致性）：`019d9a8f-ce16-74a1-a4a7-a2c450e502cf`
- Agent C（实验协议与监控）：`019d9a8f-ceef-7e03-9ad1-e287fefb227d`

## 2. 结论摘要（先给结论）
结论：**v3 仍存在“功能性双源失活”**。

需要区分两件事：
- 不存在“两个源都彻底死掉”的硬失活（source0 被合同保护）。
- 但存在“无法维持有效双源协同”的软失活（长期退化为单源主导/单源切换）。

更准确的系统行为是：
- 短暂 dual-open
- 进入 self-heavy（DINO 贴 floor）
- 达到报警阈值后触发 mitigation
- 回到 DINO-only
- 之后重复类似循环

因此目前可支持的声明是“anti-collapse 合同层面改进有效”，
不支持“已解决双源协同/可靠路由”。

## 3. 关键证据链

### 3.1 运行层证据（post-unlock）
来自训练笔记：
- `dual_alive = 2/44 (4.55%)`
- `avg_dual_synergy_margin = -0.3993`
- `max_dino_starved_alarm = 1`

这已经直接违反 v3 成功标准中的核心项（双源覆盖率与正协同）。

### 3.2 原始日志抽样证据（两个塌缩周期）
从 `s3a_bz128_400k_v4.log` 抽样：
- `26000`: `a_dino=0.136`, `a_self=0.864`, `dual_alive=1`, 但 `synergy_margin<0`
- `26500~27500`: `a_dino=0.05`, `a_dino_above_floor=0.0`, `a_self=0.95`
- `27500`: `dino_starved_alarm=1`, `mitigate=1`
- `28000`: 回到 `a_dino=1.0`, `a_self=0.0`
- `37000~39000`: 同模式再次出现
- `46500+`: 仍是 DINO-only，`dual_alive=0`, `synergy_margin≈-0.486`

该证据说明不是一次偶发噪声，而是可复现模式。

## 4. 机制层审计结果（按严重级别）

### P0-1 控制目标错位（最关键）
当前 gate/reopen 基于每源 utility EMA，
但系统成功目标（`dual_synergy_margin > 0`）没有进入控制闭环。

结果：控制器优化的是“选更好单源”，不是“维持协同双源”。

代码定位：
- gate 更新：`update_gate_state(...)`
- utility 估计：`policy_loo` 的 leave-one-out/add-one
- synergy 只计算不控制：日志窗口阶段

### P0-2 缓解机制是 reset，而不是 recovery
报警后通过 `set_self_mitigation_windows` 全面关 self 并清状态。
恢复后继续按同一 reopen 规则，无“协同准入”约束，容易再塌缩。

结果：形成“重置环路”，不是“稳定恢复环路”。

### P1-1 floor 只防零，不防失活
`source0_min_alpha` + `protect_source0_min_alpha` 确保 DINO 质量下界，
但不保证 DINO 的有效贡献或双源协同收益。

允许出现“名义双源，实质单源”状态：`a_dino≈floor`, `a_self≈0.95`。

### P1-2 warmup 退出是冷切换
warmup 期间 self 被 mask 且状态清空，退出时 router 近均匀重启，
缺少“分阶段 reopen + 协同验收”交接策略。

### P2-1 指标口径与声明不一致
代码里的 `dual_alive` 判定阈值偏宽（低门槛活性），
但文档将其当“协同成立”证据，存在误报风险。

### P2-2 协议缺少“未证实(inconclusive)”档
当前只有成功/失败，缺少中间档，导致边界 run 容易被叙述性放大。

## 5. 为什么 cross-layer + cross-timestep 仍不够
v3 的核心改动（跨层+跨时间步 self-source）解决的是
“literal identity shortcut”的几何近邻问题，
但没有同步改变控制器的目标函数与准入规则。

因此它降低了“直接复制型塌缩”的概率，
却没消除“utility 驱动单源化”的结构趋势。

## 6. 设计主张审计矩阵
| 主张 | 审计结论 | 证据 |
|---|---|---|
| v3 去掉了同层同时间步捷径 | 支持 | 实现与参数一致（offset, timestep offset） |
| v3 已建立稳定双源协同 | 不支持 | dual_alive 低、synergy 长期为负 |
| v3 已形成可靠 source reliability routing | 不支持 | 控制信号是单源 utility，不是协同目标 |
| v3 已建立 anti-collapse contract | 支持 | floor/protect/fallback/alarm-mitigate 生效 |

## 7. 最小修复包（建议顺序）
1. **联合准入/重开条件**
   - reopen 不只看 `U_self_off`，还要同时满足：
   - `alpha_dino_above_floor > eps`
   - `dual_synergy_margin > 0`
   - 连续 K 个窗口成立

2. **mitigation 从 reset 改为 quarantine recovery**
   - 不要一次性完全回开 self
   - 增加 re-entry cap/ramp + 分段放行

3. **将严格协同指标提升为控制与验收主指标**
   - 使用 per-layer floor-relative 指标
   - 把 `dual_alive_strict` 引入门控而不是仅日志

4. **warmup 退出改为 two-stage unlock**
   - Stage A: 只观察 self-off probe / synergy 证据
   - Stage B: 条件达标后才允许 self 实际参与策略

## 8. 验证协议升级（建议替换 RUN_V3_VERIFY）

### 8.1 审查窗口
- 保持当前 `log_every=500` 时，unlock 后至少看 10 个窗口（约 5k steps）
- 更稳妥建议短验证拉到 45k，以覆盖一次完整 reset loop

### 8.2 强双源窗口定义（建议）
同时满足：
- `a_self >= 0.10`
- `a_dino_above_floor >= 0.05`
- `alpha_dino_min_layer >= source0_floor_active + 0.02`
- `synergy_margin >= +0.02`
- `gate_self_state >= 0.70`
- `rGap <= 0.20`
- 且 `dino_starved=0`, `alarm=0`, `mitigate=0`

### 8.3 通过/失败门槛
- 通过：10 窗口中 >=8 强双源窗口，且 0 次 alarm/mitigate/starved_alarm
- 失败：任一窗口出现 `dino_starved_alarm=1` 或 `mitigate=1`，或强双源窗口 <=2
- 其余：`inconclusive`，不得升级到 400k

## 9. 最小实验包
1. `Baseline-v3-short`（45k, seed0）
2. `Fixed-v3-short`（只改 controller，其他不变，seed0）
3. `Fixed-v3-repeat`（seed1）

只有当 2/3 同时满足严格门槛，才可宣称“机制修复有效”。

## 10. 审计过程中的版本一致性注意
- `RUN_V3_VERIFY.md` 前置条件写的是 `9dcf858+`，
- 但当前证据 run 记录为 `0b466ad...`。

这不必然推翻结论，但说明协议需改为“精确 SHA 锁定 + resolved_args 固化”。

## 11. 最终路线建议
- 当前 route：`supplement / mechanism-fix`
- 允许对外表述：
  - anti-collapse engineering improved
- 暂不允许对外表述：
  - dual-source collaboration solved
  - robust source reliability routing established

---
本报告是“本地代码审计 + 三个 xhigh 代理交叉评审”的合并结论。
若需要，我可以下一步直接给出 `RUN_V3_VERIFY.md` 的“严格验收版”替换稿。
