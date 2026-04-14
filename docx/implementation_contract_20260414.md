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
- 不与 `t-aware/SASA` 在 projector 结构上混淆比较。

## 3. 已清理的混淆点

- `train_sasa.py` 已移除 no-op `--repa-diff-schedule` 参数。
- `run_sasa.sh` 不再传递无效 diff schedule 参数。
- `run_sitxl_repa_dinov2_400k.sh` 已修复训练脚本文件名契约。

## 4. 对比实验口径

- `t-aware vs SASA`：projector 固定一致，仅比较时间/调度机制。
- `S3A`：作为外置灵活分支的独立方法比较，需配套消融而非直接把全部增益归因为单模块。
