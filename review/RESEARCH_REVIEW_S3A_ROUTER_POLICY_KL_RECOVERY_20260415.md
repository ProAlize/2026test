# Research Review: S3A Router-Policy KL Recovery (2026-04-15)

## Scope
- Target file: `train_s3a_multisource_dinov2.py`
- Supporting launchers:
  - `run_s3a_multisource_dinov2.sh`
  - `scripts/run_e0_e7_single_seed.sh`
- Goal: fix persistent DINO starvation pattern (`raw_dino` collapse with floor-only survival), while keeping an engineering-first contract.

## Problem Snapshot (Before This Round)
- Prior smoke (`r4`) showed:
  - post-unlock: `raw_dino` collapsed to ~`0.006`
  - `a_dino` stayed non-zero mostly due floor
  - `dino_starved_alarm=1` and mitigation trigger fired
- This indicated raw router policy and deployed alpha policy were drifting apart.

## Code Changes Applied

### 1) Router-policy alignment term (main fix)
- Added `router_policy_kl_and_gap_per_sample(raw_alpha, policy_alpha)`.
- Added new training objective term inside `compute_s3a_alignment_loss()`:
  - `KL(raw_alpha || alpha_policy.detach())`
  - weighted by `--s3a-router-policy-kl-lambda`.
- Added telemetry fields:
  - `router_policy_kl`
  - `router_policy_gap`
- Added helper `make_empty_align_stats(...)` to remove duplicated large stats template blocks.

### 2) Config/contract/plumbing
- New CLI arg:
  - `--s3a-router-policy-kl-lambda` (default finalized to `0.1`)
- Added to resume critical contract key set.
- Bumped `METRICS_SCHEMA_VERSION: 3 -> 4`.
- Added contract row field and metric row fields for new router alignment stats.

### 3) Launcher alignment
- Added passthrough env var in both launchers:
  - `S3A_ROUTER_POLICY_KL_LAMBDA` (default `0.1`)
- Added this value to E0-E7 experiment suffix (`rkl...`).

### 4) Post-audit fixups
- Fixed resume backward-compatibility regression risk:
  - missing `s3a_router_policy_kl_lambda` in legacy ckpt now only auto-accepted when current value is legacy-equivalent `0.0`.
  - otherwise requires `--allow-legacy-resume-args`.
- Added unsafe flag identity into E0-E7 `contract_suffix`:
  - `uf...` and `uw...`

## Smoke Validation (A/B)

### Run R5 (lambda=0.02)
- Log: `monitor_logs/s3a_fix_smoke_20260415_r5.log`
- Key windows:
  - step300: `raw_dino=0.201`, `rGap=0.119`
  - step600: `raw_dino=0.025`, `dino_starved_alarm=1`, `mitigate=1`
- Verdict: improved vs old collapse, but still fails starvation alarm.

### Run R6 (lambda=0.1)
- Log: `monitor_logs/s3a_fix_smoke_20260415_r6.log`
- Key windows:
  - step300: `raw_dino=0.325`, `rGap=0.095`
  - step500: `raw_dino=0.081`, `dino_starved_windows=0`
  - step600: `raw_dino=0.089`, `dino_starved_alarm=0`, `mitigate=0`, `rGap=0.009`
- Verdict: raw/effective drift issue is materially closed in short-run smoke.

## 3-Agent Audit Rounds

### Round A (Mechanism)
- Agent: `Fermat`
- Finding: KL mechanism is self-consistent; `lambda=0.1` is effective for raw/effective drift.
- Residual risk: still floor-hugging (`alpha_dino_above_floor` tiny in short run).
- Verdict: `partial`.
- Report: `RESEARCH_REVIEW_S3A_ROUTER_POLICY_KL_FINAL_AUDIT_20260415.md`

### Round B (Engineering)
- Agent: `Planck`
- Initial finding (high): legacy resume could silently change objective for old checkpoints.
- Initial finding (low): unsafe flags missing in suffix identity.
- Actions taken:
  - tightened legacy default handling for `s3a_router_policy_kl_lambda`
  - encoded unsafe flags in `contract_suffix`
- Delta re-audit verdict: `can ship`.

### Round C (Result-to-Claim)
- Agent: `Mill`
- Allowed claims:
  - anti-collapse engineering path improved and stable
  - r6 better than r5 on starvation indicators
- Disallowed claims:
  - “dual-source synergy solved”
  - “positive dual synergy already proven”
- Verdict: `partial`.
- Report: `RESEARCH_REVIEW_S3A_RESULT_TO_CLAIM_R5_R6_20260415.md`

## Current Status
- Closed in this round:
  - raw router collapse/drift (short-run engineering objective)
  - resume-contract regression for new router KL key
  - launcher identity mismatch for unsafe flags
- Not yet closed:
  - strong collaboration claim (floor-above contribution and positive synergy still not demonstrated in short run)

## Recommended Next Minimal Step
- Run one fresh `10k` dual-source canary with current `r6` contract (`router KL = 0.1`), no extra mechanisms.
- Promote to “collaboration solved” only if post-warmup windows repeatedly satisfy:
  - `dino_starved_alarm = 0`
  - `mitigate = 0`
  - sustained non-trivial `alpha_dino_above_floor`
  - positive `dual_synergy_margin` over multiple windows.
