# 实现口径冻结说明（2026-04-14）

目的：统一 `t-aware / SASA / S3A` 的代码口径，避免后续 agent 因历史文档产生歧义。

## 1. 生效优先级

- 本文件优先级高于 `docx` 下 2026-04-14 之前的历史评审草稿（如 `review*.md`、早期 root-cause 草稿）中与实现冲突的描述。
- 若本文件与代码冲突，以代码为准，并要求同步更新本文件。

## 2. 架构边界（冻结）

1. `t-aware`（真值分支：`exp_taware_adm_eval_20260410`）
- projector：REPA-faithful（`Linear -> SiLU -> Linear -> SiLU -> Linear`）。
- 默认宽度：`projector_dim=2048`（通过 `--repa-hidden-dim` 可覆盖）。
- 核心机制：per-sample diffusion timestep-aware gating（`g(t)`）。

注：`s3a` 分支内的 `train_2.py` 仅作为历史/兼容脚本，不再作为 `t-aware` 真值实现引用。

2. `SASA`（`train_sasa.py` / `train_sasa_dinov2.py`）
- projector：与 `t-aware` 完全一致（REPA-faithful，默认 2048）。
- `SASA` 相对 `t-aware` 的研究差异：时间/调度机制，不再包含 projector 差异。

3. `S3A`（`train_s3a_multisource_dinov2.py`）
- 保持外置对齐分支：multi-layer tap + router + gate + holistic loss。
- gate 的 utility 估计默认使用 `policy_loo`：
  与训练同一 `alpha` 策略（同 router + 同 floor 规则）下做 leave-one-out / add-one 反事实收益估计；
  `uniform/raw_alpha` 仅作为 legacy 对照模式保留。
- gate 控制器状态采用双轨 EMA：`active_utility_ema`（开门态）与 `inactive_utility_ema`（关门态）分离维护；
  gate flip 时重置 EMA 与计数器，避免 on/off estimand 混合导致控制语义漂移。
- 不与 `t-aware/SASA` 在 projector 结构上混淆比较。

## 3. 已清理的混淆点

- `train_sasa.py` 已移除 no-op `--repa-diff-schedule` 参数。
- `run_sasa.sh` 不再传递无效 diff schedule 参数。
- `run_sitxl_repa_dinov2_400k.sh` 已修复训练脚本文件名契约。

## 4. 对比实验口径

- `t-aware vs SASA`：projector 固定一致，仅比较时间/调度机制。
- `S3A`：作为外置灵活分支的独立方法比较，需配套消融而非直接把全部增益归因为单模块。

## 5. 文档职责边界（防止继续零散）

- 本文件：`实现契约`（参数、接口、行为边界、可恢复性约束）。
- `idea_s3a_with_audit_feedback.md`：`方法设计`（为什么这样设计、机制闭环）。
- `review_notes_s3a_20260414.md`：`审计记忆`（外部审计结论、风险清单、下一步）。

## 6. 2026-04-15 最新问题同步（Dual-EMA 审计）

以下为当前工程事实，必须在实现与实验中显式处理：

1. P0 `legacy resume` 语义污染（已修复）
- 已实现：legacy `source_utility_ema` 在迁移时丢弃，不再双拷贝到 `active/inactive` 两轨。
- 已实现：legacy 场景下 gate 计数器重置，并记录 migration notes。
- 当前行为：checkpoint format 升级到 `v5`，旧版本通过迁移路径兼容。

2. P1 `sticky-off` 风险（已实现保护，默认开启）
- 已实现：off-state add-one probe 支持最小探索权重 `--s3a-gate-reopen-probe-alpha-floor`。
- 默认值 `0.05`（训练脚本与 launcher 同步）；仅在 `policy_loo` 下生效。
- 已实现：参数契约校验（非 `policy_loo` 或 floor 组合不可行时 fail-fast）。

3. P1 `protect_source0` 仅保可用性，不保有效使用（已实现保护，默认开启）
- 已实现：source0 持久最小 alpha `--s3a-protect-source0-min-alpha`。
- 默认值 `0.05`（训练脚本与 launcher 同步）。
- 口径要求：若运行时显式覆盖为 `0.0`，不得宣称“DINO usage 下界被保证”。

4. P1 flip-reset 零起点偏置（已修复）
- 已实现：进入新 regime 后 invalidate 对应 EMA，并使用 first-sample seeding。
- 当前语义：避免 hard-zero 冷启动导致的 reopen 慢启动偏置。

5. P1 指标语义漂移风险（已修复）
- 已实现：metrics JSONL 增加 `metrics_schema_version`、`utility_ema_semantics`、`record_type`、`utility_probe_mode`。
- 已保留：`utility_self_ema` 作为兼容别名；并新增显式字段 `utility_self_active_ema` / `utility_self_inactive_ema`。

## 7. 当前可用论文口径（冻结）

- 允许：`guarded dual-source auxiliary alignment`。
- 不允许：`principled reliability routing`、`general multi-source routing`、`guaranteed DINO usage`。
