# S3A Review Notes (Memory) - 2026-04-14

## Snapshot
- Branch: `s3a`
- Latest deep review report:
  - `/home/liuchunfa/2026qjx/2026test/RESEARCH_REVIEW_S3A_DEEP_AUDIT_20260414.md`

## Consensus Memory
1. Engineering hardening has materially improved vs unsafe legacy contract.
2. Method claim boundary must be narrowed now:
   - safe: guarded dual-source auxiliary alignment
   - unsafe: principled source reliability routing
3. Current status is still `REVISE-and-VERIFY`.
4. Priority is fresh causal evidence, not more static review.

## High-Impact Next Steps
1. Run causal source-quality intervention (True/Shuffle/Corrupt DINO).
2. Run drop-one bundle decomposition (warmup/floor/self-freeze/gate).
3. Run source necessity matrix (DINO-only/Self-only/Dual-guarded).
4. Enforce launch identity + resume equivalence regression gates.

## Writing Boundary
- Allowed now: stability and engineering contract claims.
- Forbidden now: over-claiming reliability routing or multi-source bank generality.

## 2026-04-15 Post-Patch Review Update
- New report: `/home/liuchunfa/2026qjx/2026test/RESEARCH_REVIEW_S3A_POSTPATCH_20260415.md`
- Reliability moved from ~4.5/10 to ~5.8/10 after contract hardening.
- Remaining blocker is method-level utility-policy mismatch + provenance closure.

## 2026-04-15 Policy-LOO Patch Review Update
- New report: `/home/liuchunfa/2026qjx/2026test/RESEARCH_REVIEW_S3A_POLICY_LOO_20260415.md`
- Core improvement: utility estimator switched to policy-consistent `policy_loo` (same alpha policy as train path).
- Consolidated reliability: ~6.9/10 (mechanism 7, engineering 8, AC 5.7).
- New remaining blocker: controller memory mixes on/off estimands in single EMA; thresholds need retune under new estimator scale.

## 2026-04-15 Dual-EMA Gate Patch Review Update
- New report: `/home/liuchunfa/2026qjx/2026test/RESEARCH_REVIEW_S3A_DUAL_EMA_20260415.md`
- Core improvement: gate controller now separates `active_ema` and `inactive_ema`, with gate-flip reset.
- Consensus: fresh-start semantics are materially improved; safe narrative remains `guarded dual-source auxiliary alignment`.
- New highest-priority blocker: legacy resume migration currently copies old mixed EMA into both new tracks, which may reintroduce biased controller state.

## 2026-04-15 问题台账（统一版）

1. P0: legacy resume 迁移污染
- 现状：已修复。
- 说明：legacy 单轨 EMA 在迁移中丢弃，避免双轨污染；迁移 notes 已写入 resume meta。

2. P1: sticky-off reopen 困难
- 现状：已加机制保护（默认关闭）。
- 说明：新增 off-state probe 最小探索权重 `--s3a-gate-reopen-probe-alpha-floor`（默认 0）。

3. P1: DINO usage 未被保证
- 现状：已加机制保护（默认关闭）。
- 说明：新增 `--s3a-protect-source0-min-alpha` 可提供 source0 使用下界。

4. P1: flip reset 零起点偏置
- 现状：已修复。
- 说明：已改为 entered-regime invalidate + first-sample seeding。

5. P1: 指标语义漂移
- 现状：已修复。
- 说明：metrics row 已加入 schema/version 与 `utility_ema_semantics`；保留兼容别名字段。

## 当前决策

- 研究口径：继续使用 `guarded dual-source auxiliary alignment`。
- 审稿口径：暂不使用“principled reliability routing”措辞。

## 2026-04-15 Final Delta Audit Update
- Summary doc: `/home/liuchunfa/2026qjx/2026test/RESEARCH_REVIEW_S3A_FINAL_DELTA_20260415.md`
- Converged status: `REVISE-and-VERIFY`.
- Newly confirmed still-open risks:
  1) legacy resume sterilization incomplete for stale `source_gate_mask` carry-over,
  2) sticky-off still possible under shipped defaults (`reopen_probe_alpha_floor=0.0`),
  3) missing-key allowlist is fail-open for non-default overrides,
  4) floor composition feasibility not jointly validated.
- Safe paper posture unchanged: `guarded dual-source auxiliary alignment` (not general reliability routing claim).

## 2026-04-15 DINO Deactivation Deep Audit
- Detailed report: `/home/liuchunfa/2026qjx/2026test/RESEARCH_REVIEW_S3A_DINO_DEACTIVATION_20260415.md`
- Key result:
  - fixed collapse-threshold equality dead-zone (`<` -> `<=`),
  - added fail-fast when `collapse_alpha_threshold < protect_source0_min_alpha`.
- Current status:
  - no P0/P1 blocker on this issue,
  - remaining concern is soft underuse observability (optional telemetry improvement, non-blocking).
