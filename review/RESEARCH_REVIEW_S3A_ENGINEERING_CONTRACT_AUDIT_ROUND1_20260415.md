## S3A Engineering Contract Audit Round 1 (2026-04-15)

### Scope

- `/home/liuchunfa/2026qjx/2026test/train_s3a_multisource_dinov2.py`
- `/home/liuchunfa/2026qjx/2026test/scripts/run_e0_e7_single_seed.sh`
- `/home/liuchunfa/2026qjx/2026test/run_s3a_multisource_dinov2.sh`

### Questions

1. Is `source0 floor=0` blocked by default in dual-source mode, with an explicit unsafe override?
2. Is the legacy resume source0 gate lane fully sterilized?
3. Does backward-compatible resume still work for old checkpoints?
4. Are there any new destructive regressions in arg validation, launchers, or logging/metrics?

### Findings

#### High

1. `scripts/run_e0_e7_single_seed.sh` introduces a launcher regression on the default safe path.

Evidence:

- The script builds `unsafe_zero_source0_floor_flag=""` by default at line 266.
- It always forwards the variable as a quoted argument at line 341:
  - `"$unsafe_zero_source0_floor_flag"`
- In bash, `"$empty_var"` is still a real empty argument, not "no argument".
- `argparse` treats that empty string as an unexpected positional argument and exits with code 2.

Impact:

- The E0-E7 launcher can fail even when the user keeps the safe default contract.
- This is a real engineering regression because the failure happens on the primary non-unsafe path.

Minimal fix:

- Stop forwarding the flag as a quoted scalar.
- Use an array for optional flags, matching the pattern already used in `run_s3a_multisource_dinov2.sh`.

Suggested patch shape:

- Replace `local unsafe_zero_source0_floor_flag=""` with `local optional_flags=()`
- Append `--s3a-allow-unsafe-zero-source0-floor` only when enabled
- Expand with `"${optional_flags[@]}"`

#### Medium

None found in the requested contract scope.

#### Low

1. Resume-contract error text is slightly stronger than the actual logic for newly added float keys.

Evidence:

- `_validate_resume_contract()` allows missing legacy keys `s3a_protect_source0_min_alpha` and `s3a_gate_reopen_probe_alpha_floor` to backfill from current args without checking a fixed legacy-equivalent default.
- The error text says missing keys are auto-backfilled only when current values match legacy defaults.

Impact:

- Not a correctness bug.
- It can mislead debugging when an old checkpoint resumes under new safe floor values.

Minimal fix:

- Either soften the message text, or log which missing keys were backfilled from current args during resume.

### Checks Against Requested Questions

#### 1. Dual-source source0 floor=0 default prohibition

Pass.

Evidence:

- Parser default is `--s3a-protect-source0-min-alpha=0.05` in `train_s3a_multisource_dinov2.py:3163`.
- `validate_args()` rejects dual-source runs with `source0_min_alpha <= 0` unless `--s3a-allow-unsafe-zero-source0-floor` is explicitly passed in `train_s3a_multisource_dinov2.py:3305`.
- Both launchers default to `S3A_PROTECT_SOURCE0_MIN_ALPHA=0.05` and only add the unsafe flag when the explicit environment toggle is set:
  - `scripts/run_e0_e7_single_seed.sh:39-40,287-289`
  - `run_s3a_multisource_dinov2.sh:42-43,179-181`

Conclusion:

- The default dual-source contract now blocks `source0 floor=0`.
- Unsafe override remains explicit and opt-in.

#### 2. Legacy resume source0 gate lane sterilization

Pass.

Evidence:

- `_migrate_legacy_s3a_state()` now calls `_sterilize_source0_gate_lane()` unconditionally after legacy migration work in `train_s3a_multisource_dinov2.py:1698-1744`.
- `load_checkpoint()` also calls `_sterilize_source0_gate_lane()` before loading state, even outside the format-version migration block, in `train_s3a_multisource_dinov2.py:1835-1843`.
- Sterilization forces `source_gate_mask[:, 0] = 1.0` and resets source0 controller lanes to expected-state values in `train_s3a_multisource_dinov2.py:1661-1695`.
- Runtime masking also hard-keeps source0 available on forward path in `train_s3a_multisource_dinov2.py:865-872`.

Conclusion:

- Old checkpoint residue in the source0 gate lane is now actively cleaned instead of passively tolerated.

#### 3. Backward-compatible resume for old checkpoints

Pass, with one caveat on messaging only.

Evidence:

- `_validate_resume_contract()` exempts missing legacy keys:
  - `s3a_protect_source0_min_alpha`
  - `s3a_gate_reopen_probe_alpha_floor`
  - `s3a_allow_unsafe_zero_source0_floor`
- This logic is in `train_s3a_multisource_dinov2.py:1585-1621`.
- Boolean unsafe flag is only auto-accepted when current args still equal the legacy-equivalent default `False`.

Conclusion:

- Old checkpoints should still resume without tripping the new arg contract on these added keys.
- The remaining issue is wording clarity, not compatibility.

#### 4. Destructive regressions in validation, launcher compatibility, logging/metrics

Mixed result.

Passes:

- Arg validation now enforces:
  - positive warmup unless explicitly unsafe
  - positive source0 persistent floor unless explicitly unsafe
  - reopen-probe floor consistency with source0 floor
- Contract row is now emitted to `metrics.jsonl` at launch in `train_s3a_multisource_dinov2.py:2249-2279`.
- Runtime metrics now include:
  - `source0_floor_active`
  - `alpha_dino_above_floor`
  - `dino_starved`
  - `dual_source_alive`
  in `train_s3a_multisource_dinov2.py:2731-2794`.

Regression:

- `scripts/run_e0_e7_single_seed.sh` default launch path forwards an empty optional argument and can fail before training starts.

### Minimal Action List

1. Fix `scripts/run_e0_e7_single_seed.sh` to use an optional-args array for the unsafe source0-floor flag.
2. Optionally soften or clarify the legacy resume backfill error text.

### Reviewer Verdict

- The core engineering contract fixes are mostly correct.
- The only material blocker found in this round is the `scripts/run_e0_e7_single_seed.sh` empty-argument launcher bug.
- After that fix, this contract patch set is substantially cleaner and closer to a usable S3A baseline.
