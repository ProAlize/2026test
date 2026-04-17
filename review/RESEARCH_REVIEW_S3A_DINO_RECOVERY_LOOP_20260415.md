# S3A DINO Recovery Loop (2026-04-15)

## Scope
- Goal: stop DINO source hard-deactivation and make S3A dual-source behavior engineering-safe.
- Principle: minimal, production-facing fixes; avoid stacking new controllers.

## Code Changes Applied

### 1) Trainer contract + migration hardening
- File: `train_s3a_multisource_dinov2.py`
- Changes:
  - Added `--s3a-allow-unsafe-zero-source0-floor` and enforced dual-source safety:
    - default dual-source now rejects `--s3a-protect-source0-min-alpha <= 0` unless explicit unsafe override.
  - Centralized source0 floor computation via `source0_min_alpha_at_step()` and reused in alpha builder.
  - Strengthened legacy migration/load sterilization:
    - `source0` gate lane is explicitly sterilized in migration and load paths.
  - Resume-contract compatibility:
    - backward-compatible missing-key handling covers late-added keys and keeps legacy resume usable.
  - Observability:
    - `metrics.jsonl` now writes `contract` row at startup.
    - Added metric fields: `source0_floor_active`, `alpha_dino_above_floor`, `dino_starved`, `dual_source_alive`.
    - Added synergy fields: `dual_synergy_margin`, `dual_synergy_supported`.
  - Added starvation-mitigation bridge:
    - track `dino_starved_windows`, emit `dino_starved_alarm`.
    - auto-mitigation now triggers on `(collapse_alarm OR dino_starved_alarm)`.
    - runtime state persistence now includes `dino_starved_windows`.

### 2) Launcher safety alignment
- Files:
  - `scripts/run_e0_e7_single_seed.sh`
  - `run_s3a_multisource_dinov2.sh`
- Changes:
  - Added env-managed unsafe flags:
    - `S3A_ALLOW_UNSAFE_ZERO_SOURCE0_FLOOR`
    - `S3A_ALLOW_UNSAFE_ZERO_WARMUP`
  - Fixed shell regression (critical):
    - removed empty-argv propagation by replacing scalar optional flag with array expansion.

### 3) Monitoring script alignment
- File: `scripts/monitor_training_20m.sh`
- Changes:
  - Parses new log keys: `a_dino_above_floor`, `dino_starved`, `dual_alive`.
  - Uses safe numeric defaults for awk comparisons (avoid empty-value syntax errors).
  - Adds `dual_not_alive_rounds` and integrates dual-alive based warning/critical rules.

## Multi-Agent Audit Loop

### Round-1 (3-agent)
- Engineering audit:
  - Found critical shell regression (empty argv in `run_e0_e7_single_seed.sh`) and flagged as blocker.
- Mechanism audit:
  - Confirmed exact `a_dino=0` is mostly closed by safety contract; warned about floor-supported pseudo dual-source.
- Monitoring/ops audit:
  - Requested monitoring alignment to newly added starvation fields.

Actions taken after Round-1:
- Fixed empty-argv launcher bug.
- Extended contract row fields.
- Added unsafe warmup env/flag pass-through.
- Updated monitor script to parse starvation fields.

### Round-2 (3-agent)
- Engineering audit:
  - No high/medium regressions; only low-severity wording mismatch in resume-contract error text.
  - Confirmed launcher unsafe flags and migration behavior are consistent.
- Mechanism audit:
  - Confirmed exact DINO deactivation is engineering-controlled.
  - Recommended minimal synergy metric (implemented: `dual_synergy_margin`).
- Monitoring audit:
  - Requested stronger integration of dual-alive signal and safer fallback parsing (implemented).

Actions taken after Round-2:
- Added synergy metric fields.
- Added `dino_starved_alarm` + mitigation coupling.
- Hardened monitor state machine (`dual_not_alive_rounds`, safe parsing).

### Final Delta Audit (3-agent)
- Engineering delta audit:
  - Raised concern that starvation-triggered mitigation is stronger than legacy collapse utility/probe gate.
- Mechanism delta audit:
  - Verified state-machine coherence (no contradictory or impossible states); bounded oscillation remains possible.
- Monitoring delta audit:
  - Found `inf` keyword false-positive risk in error regex.

Final closeout actions:
- Tightened starvation predicate with `raw_alpha_dino <= collapse_alpha_threshold` gate.
- Narrowed monitor error regex to avoid broad `inf` substring false alarms.
- Clarified resume-contract missing-key error wording.

## Smoke Validation Runs

### R1 (failed by design regression discovery)
- Log: `monitor_logs/s3a_fix_smoke_20260415.log`
- Result: failed at argparse due empty argv.
- Root cause: scalar empty optional flag in launcher.

### R2 (startup success, warmup too long for coexistence check)
- Log: `monitor_logs/s3a_fix_smoke_20260415_r2.log`
- Config: `warmup=5000, max_steps=600`
- Result: starts and runs; but self never unlocks, so not a valid coexistence test.

### R3 (coexistence observed)
- Log: `monitor_logs/s3a_fix_smoke_20260415_r3.log`
- Config: `warmup=200, max_steps=600`
- Key windows:
  - `step=300`: `a_dino=0.110`, `a_dino_above_floor=0.014`, `a_self=0.890`, `dino_starved=0`
  - `step=400`: `a_dino=0.096`, `a_dino_above_floor=0.001`, `a_self=0.904`, `dino_starved=1`
- Interpretation: exact zero avoided; pseudo-dual-source risk present (near-floor DINO).

### R4 (post starvation-mitigation patch)
- Log: `monitor_logs/s3a_fix_smoke_20260415_r4.log`
- Config: `warmup=200, max_steps=800`
- Key windows:
  - `step=400`: `a_dino=0.096`, `a_dino_above_floor=0.001`, `dino_starved=1`, `dino_starved_windows=1`, `mitigate=0`
  - `step=500`: `a_dino=0.094`, `a_dino_above_floor=0.001`, `dino_starved=1`, `dino_starved_windows=2`, `mitigate=0`
  - `step=600`: `a_dino=0.093`, `a_dino_above_floor=0.001`, `dino_starved_alarm=1`, `dino_starved_windows=3`, `mitigate=1`
  - `step=700`: `gate_self=0`, `a_dino=1.000`, `a_self=0.000` (mitigation hold window in effect)
  - `step=800`: `gate_self=0`, `a_dino=1.000`, `mitigateW=2`
- Interpretation:
  - Starvation alarm now connects to mitigation as designed.
  - DINO no longer stays in persistent near-floor mode once starvation persists.
  - `dual_synergy_margin` remained negative in this short run, so this smoke validates anti-starvation behavior, not final collaborative gain.

## Current Convergence Status
- Closed:
  - exact DINO hard-deactivation path (`a_dino=0`) under default safe contract.
  - launcher contract regression and unsafe-flag inconsistency.
  - legacy source0 gate lane sterilization gap.
  - starvation alarm now proven to trigger mitigation in smoke (`step=600 -> mitigate=1`, `step=700/800 -> self gated off`).
- Remaining risk:
  - short-run data still shows negative `dual_synergy_margin`; collaboration quality still needs longer-run tuning/validation.
- Added for closure:
  - explicit synergy observability (`dual_synergy_margin`).
  - starvation-to-mitigation bridge (`dino_starved_alarm`).
  - starvation predicate tightened with `raw_alpha_dino <= collapse_alpha_threshold` guard.
  - monitor false-positive fix for broad `inf` keyword matching.

## Files for external audit trail
- `RESEARCH_REVIEW_S3A_ENGINEERING_CONTRACT_AUDIT_ROUND1_20260415.md`
- `RESEARCH_REVIEW_S3A_ENGINEERING_CONTRACT_AUDIT_ROUND2_20260415.md`
- `RESEARCH_REVIEW_S3A_MECHANISM_ROUND1_AFTER_FIXES_20260415.md`
- `RESEARCH_REVIEW_S3A_MECHANISM_ROUND2_20260415.md`
- `RESEARCH_REVIEW_S3A_MONITOR_AUDIT_ROUND2_20260415.md`
- `RESEARCH_REVIEW_S3A_FINAL_DELTA_ENGINEERING_AUDIT_20260415.md`
- `RESEARCH_REVIEW_S3A_MECHANISM_FINAL_DELTA_AUDIT_20260415.md`
- `RESEARCH_REVIEW_S3A_MONITOR_DELTA_AUDIT_FINAL_20260415.md`
