# Findings

## 2026-04-15 Result-to-Claim Gate (in-flight run)

- Run: `s3a_e7only_xl_s0_160k_v1`
- Log: `/home/liuchunfa/2026qjx/2026test/monitor_logs/s3a_e7only_xl_s0_160k_v1.log`
- Snapshot step: `0089000` (latest observed at `2026-04-15 17:50:47`)
- Reviewer: secondary agent (`gpt-5.4 xhigh`, id `019d908a-7d2f-7ba1-9e60-020d66887167`)

### Structured verdict

- claim_supported: `no`
- confidence: `high`

- what_results_support:
  - Training process is stable (loss band roughly `0.148-0.156`, no non-finite/runtime error).
  - Throughput stable around `0.81-0.82 steps/s`.
  - Checkpoints were saved at `20k/40k/60k/80k`.

- what_results_dont_support:
  - Dual-source effectiveness is not supported.
  - `a_dino` decays to near-zero early and stays effectively zero for long horizon.
  - `a_self=1.000` and `gate_self=1.000` dominate, consistent with effective self-only behavior.

- missing_evidence:
  - Sustained non-trivial source0 contribution beyond early warmup.
  - Evidence of source0 reactivation or measurable impact vs self-only baseline.
  - Higher-precision diagnostics for raw alpha and gate dynamics.

- suggested_claim_revision:
  - This run only supports: "training-stable but effectively self-only alignment under current configuration."

- next_experiments_needed:
  - Add high-precision logging for `raw_alpha_dino/alpha_dino` and gate transition counters.
  - Run matched self-only baseline to verify equivalence.
  - Run anti-starvation interventions (e.g., stronger source0 floor/balance regularization, selective-gate-off ablation), then re-gate claim.

### Route decision

- Route: `supplement` (do not claim dual-source effectiveness from this run).
- Immediate action: keep run monitored, but treat as failing dual-source mechanism objective until source0 contribution recovers in subsequent interventions.

## 2026-04-15 Result-to-Claim Gate (r5/r6 final smoke)

- Runs:
  - `20260415_s3a_fix_smoke_r5`
  - `20260415_s3a_fix_smoke_r6`
- Code:
  - `/home/liuchunfa/2026qjx/2026test/train_s3a_multisource_dinov2.py`
- Logs:
  - `/home/liuchunfa/2026qjx/2026test/monitor_logs/s3a_fix_smoke_20260415_r5.log`
  - `/home/liuchunfa/2026qjx/2026test/monitor_logs/s3a_fix_smoke_20260415_r6.log`
- Review note:
  - local fallback review; delegation unavailable in this session

### Structured verdict

- claim_supported: `partial`
- confidence: `high`

- what_results_support:
  - Safe dual-source contract now blocks exact-zero DINO collapse by construction.
  - The 600-step dual-source smoke is engineering-stable in both r5 and r6.
  - Current logging is strong enough to separate floor-supported survival from actual synergy.
  - r6 is better than r5 on short-horizon anti-starvation diagnostics.

- what_results_dont_support:
  - "Dual-source collaboration is solved."
  - Positive dual-source synergy.
  - Sustained above-floor DINO contribution after warmup.
  - Strong source-reliability-routing claims.

- missing_evidence:
  - A longer dual-source canary showing persistent `dual_source_alive=1`.
  - Positive `dual_synergy_margin` after warmup and at the final window.
  - Evidence that DINO contribution is not merely floor-supported.

- suggested_claim_revision:
  - Current evidence supports "guarded dual-source anti-collapse engineering" rather than "solved dual-source collaboration."

- next_experiments_needed:
  - One fresh `10k` dual-source canary at the r6 contract; only claim solved if `dual_source_alive=1`, `dual_synergy_margin>0`, and `dino_starved_alarm=0` hold for consecutive post-warmup windows and at the final window.

### Route decision

- Route: `supplement`
- Immediate action: keep the engineering claim, but do not round it up to a mechanism claim.

## 2026-04-15 Logic & Interface Review (`$research-review`)

- Reviewer: secondary agent (`gpt-5.4 xhigh`, id `019d91ad-dae5-7492-aa06-9170b0e2ed00`)
- Review record: `/home/liuchunfa/2026qjx/2026test/RESEARCH_REVIEW_LOGIC_INTERFACE_20260415.md`
- Corrected context: canary run used `s3a_self_warmup_steps=200` (resolved args), not parser default `5000`.

Final prioritized risks:
- P1 real bug: warmup exit is mask-only, not gate-controlled handoff.
- P1 real bug: starvation/collapse conditions are floor-blind for the target failure mode (`alpha_dino_above_floor ~ 0`).
- P1 objective-induced degeneration: current objective naturally converges to self-dominant + DINO floor-hugging when self-only dominates fused loss.
- P2 interface weakness: `gate_self` currently reflects effective mask, not pure controller gate state.
- P2 interface weakness: parser/launcher/resume contract drift (defaults/backfill policy).

Ship verdict (logic/interface): `No` until warmup-handoff and floor-relative controller criteria are fixed.

## 2026-04-15 Interrupt-or-Continue Decision (`$research-review`)

- Reviewer: secondary agent (`gpt-5.4 xhigh`, id `019d91ba-61e0-7ce2-a36f-bff5578b5f81`)
- Record: `/home/liuchunfa/2026qjx/2026test/RESEARCH_REVIEW_INTERRUPT_DECISION_20260415.md`
- Decision: `STOP_NOW`

Reason summary:
- Failure regime (`a_dino_above_floor≈0`, `dual_alive=0`, `synergy_margin<0`) appeared early and persisted through step 2000+.
- Existing step-2000 checkpoint is sufficient as defective-regime evidence.
- Additional running to next checkpoint is low decision value compared with patch-and-rerun.

## 2026-04-15 Logic/Interface Delta Recheck (`$research-review`)

- Reviewer: `gpt-5.4 xhigh` (agent `019d91dd-f14d-7ce3-a1ac-cdabbe30f885`)
- Record: `/home/liuchunfa/2026qjx/2026test/RESEARCH_REVIEW_LOGIC_INTERFACE_DELTA_20260415.md`

Summary:
- Round1 found one rerun blocker (collapse-threshold validation vs launcher defaults mismatch).
- We removed the blocking validation and clarified `--s3a-collapse-alpha-threshold` as diagnostic-only.
- Round2 verdict: rerun readiness `YES` (no remaining fresh-run P1 blocker).

## 2026-04-16 Design/Structure/Interface Review (`$research-review`)

- Reviewer: secondary agent (`gpt-5.4 xhigh`, id `019d9200-a738-7940-b495-93bbd9efed76`)
- Record: `/home/liuchunfa/2026qjx/2026test/RESEARCH_REVIEW_DESIGN_STRUCTURE_INTERFACE_20260416.md`
- Verdict: `CONDITIONAL`

Top risks:
- Mean-only starvation alarms can hide per-layer DINO starvation.
- Reopen path can re-enter floor-only regime (oscillation risk).
- Resume logs do not fully expose optimizer runtime truth after `opt_state` restore.
- Monitor/parser drift and launcher default reliance remain interface hygiene risks.

Recommended immediate path:
- Apply minimal 3-item mechanism patch (any-layer starvation, reopen hysteresis, layer-local starvation-to-gate hook), then re-run fresh 10k canary with explicit pass criteria.

## 2026-04-16 Math Logic Check (`$proof-writer`)

- Updated file: `/home/liuchunfa/2026qjx/2026test/PROOF_PACKAGE.md`
- Status: `PROVABLE AS STATED` (for contract-level claims under explicit runtime assumptions)

Main correction:
- Removed outdated claim that validator enforces
  `s3a_collapse_alpha_threshold >= s3a_protect_source0_min_alpha`.

What is proved now:
- Runtime `_build_alpha` simplex + floor guarantees.
- Source0 gate-lane hard protection under `protect_source0=True`.
- Existence of floor-relative (`dino_starved_alarm`) mitigation trigger path independent of absolute collapse-alpha threshold.

Not proved:
- Optimization-level dual-source effectiveness/synergy; still empirical.
