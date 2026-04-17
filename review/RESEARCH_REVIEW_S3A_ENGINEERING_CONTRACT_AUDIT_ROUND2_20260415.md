## S3A Engineering Contract Audit Round 2 (2026-04-15)

### Mode

- `research-review` fallback local audit.
- Reason: current session does not expose `spawn_agent` / `send_input`, so this round was executed locally with the same review deliverable shape.

### Scope

- `/home/liuchunfa/2026qjx/2026test/train_s3a_multisource_dinov2.py`
- `/home/liuchunfa/2026qjx/2026test/scripts/run_e0_e7_single_seed.sh`
- `/home/liuchunfa/2026qjx/2026test/run_s3a_multisource_dinov2.sh`
- `/home/liuchunfa/2026qjx/2026test/scripts/monitor_training_20m.sh`

### Round-2 Questions

1. Is the `run_e0_e7` empty-argv regression fully fixed?
2. Are the new unsafe overrides for `zero source0 floor` and `zero warmup` consistent between launcher and trainer?
3. Do resume contract and legacy migration remain backward-compatible without new regressions?

### Evidence Collected

#### Static inspection

- `scripts/run_e0_e7_single_seed.sh` now uses `optional_contract_flags=()` and only appends unsafe flags when enabled at lines 268-293, then expands via `"${optional_contract_flags[@]}"` at line 346.
- `run_s3a_multisource_dinov2.sh` uses the same `OPTIONAL_ARGS=()` pattern for `--s3a-allow-unsafe-zero-source0-floor` and `--s3a-allow-unsafe-zero-warmup` at lines 168-185 and expands at line 257.
- `train_s3a_multisource_dinov2.py` exposes both unsafe flags in parser at lines 3142-3155.
- `validate_args()` rejects unsafe dual-source combinations unless the explicit override is present:
  - zero warmup blocked at lines 3292-3301
  - zero source0 floor blocked at lines 3312-3321
  - reopen-probe and floor consistency checks remain active at lines 3337-3375
- Resume-contract and migration logic still include:
  - backward-compatible missing-key handling at lines 1585-1594
  - source0 lane sterilization at lines 1661-1695
  - legacy migration plus post-load sterilization at lines 1698-1744 and 1835-1843
- `scripts/monitor_training_20m.sh` now consumes `a_dino_above_floor`, `dino_starved`, and `dual_alive` from the trainer log at lines 77-82 and 168-182.

#### Dynamic checks

1. Syntax:
   - `bash -n scripts/run_e0_e7_single_seed.sh`: pass
   - `bash -n run_s3a_multisource_dinov2.sh`: pass
   - `python -m py_compile train_s3a_multisource_dinov2.py`: pass

2. `run_e0_e7` default safe path:
   - Executed `START_FROM=E5 DRY_RUN=1` and inspected emitted E5/E6/E7 commands.
   - Result: no empty `""` argv is present on the safe default path.

3. `run_e0_e7` unsafe path:
   - Executed `START_FROM=E5 DRY_RUN=1 S3A_ALLOW_UNSAFE_ZERO_SOURCE0_FLOOR=1 S3A_ALLOW_UNSAFE_ZERO_WARMUP=1 S3A_SELF_WARMUP_STEPS=0 S3A_PROTECT_SOURCE0_MIN_ALPHA=0`.
   - Result: emitted commands include both `--s3a-allow-unsafe-zero-source0-floor` and `--s3a-allow-unsafe-zero-warmup`.

4. Main launcher unsafe path:
   - Executed `run_s3a_multisource_dinov2.sh` with `TORCHRUN_BIN=/bin/echo` and valid data/model paths.
   - Result: emitted command includes both unsafe flags when requested.

5. Trainer validation:
   - Parsed args through `build_parser()` and ran `validate_args()` on six cases.
   - Result:
     - safe zero warmup: rejected
     - unsafe zero warmup: allowed
     - safe zero source0 floor: rejected
     - unsafe zero source0 floor: allowed
     - safe both zero: rejected
     - unsafe both zero: allowed

6. Resume / migration spot checks:
   - `_validate_resume_contract()` accepts legacy arg dicts missing:
     - `s3a_protect_source0_min_alpha`
     - `s3a_gate_reopen_probe_alpha_floor`
     - `s3a_allow_unsafe_zero_source0_floor`
   - `_migrate_legacy_s3a_state()` test confirmed:
     - legacy `source_utility_ema` removed
     - source0 gate lane reset to active
     - source0 runtime controller tensors reset to expected values
     - migration notes include controller reset and source0 sterilization

### Findings

#### Low

1. Resume-contract error text still overstates the backfill rule.

Evidence:

- The code allows missing legacy float keys `s3a_protect_source0_min_alpha` and `s3a_gate_reopen_probe_alpha_floor` to backfill directly from current args at `/home/liuchunfa/2026qjx/2026test/train_s3a_multisource_dinov2.py:1585`.
- The error text still says missing keys are auto-backfilled only when current values match legacy defaults at `/home/liuchunfa/2026qjx/2026test/train_s3a_multisource_dinov2.py:1615`.

Impact:

- No correctness regression found.
- The wording can still mislead resume debugging.

### Verdict

- `run_e0_e7` empty-argv regression is fixed.
- Unsafe override contract is now consistent across both launchers and the trainer.
- Resume contract and legacy migration show no new functional regression in this round.
- I found no High or Medium issues in the requested scope.
- The only remaining issue in this audit is a Low-severity resume error-message wording mismatch.

### Final Assessment

- Requested blocker checks: `pass`
- Engineering launch contract: `pass`
- Legacy compatibility: `pass`
- Safe to proceed to the next validation stage, with the caveat that resume diagnostics text can still be clarified later.
