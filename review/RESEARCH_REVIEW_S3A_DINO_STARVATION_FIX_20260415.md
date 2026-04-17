# S3A DINO Starvation Fix Review (2026-04-15)

## Context
- Target problem: dual-source S3A degenerated into effective self-only alignment.
- Failed run: `20260414_s3a_full_probe_160k / E7_s3a_spread4_dualsrc_gate_holistic_cos`.
- Training was manually interrupted after confirming persistent DINO starvation.

## Evidence Snapshot
- Step `200`: `alpha_dino=0.3232`, `alpha_self=0.6768`.
- Step `400`: `alpha_dino=0.0203`.
- Step `1000`: `alpha_dino=7.19e-4`.
- Step `5000`: `alpha_dino=2.06e-5`.
- Step `9000`: `alpha_dino=2.47e-6`.
- Long-run training log shows `a_dino=0.000`, `a_self=1.000`, `gate_self=1.000` for thousands of steps.

Interpretation:
- `self` was not gated off.
- The router itself learned the self-source shortcut extremely early.
- Existing source0 floors in current code are contract protections, but this failed run came from an older contract and therefore does not validate the current default path.

## Local Multi-Agent Review

### Agent A: Mechanism
Finding:
- The failure is not primarily a selective-gate failure.
- The dominant failure mode is early router collapse: raw mixture rapidly saturates to self before gate statistics become decision-relevant.
- Once fused target is almost pure self, alignment losses shrink, router entropy drops, and DINO path receives too little optimization pressure to recover.

Minimal mechanism conclusion:
- Source0 must be protected at the training policy level, not only in recovery probes.
- The system also needs a simple notion of "dual-source contract alive" rather than only "collapse alarm harmful enough to mitigate".

### Agent B: Engineering Contract
Finding:
- Current code already logs many useful fields in `metrics.jsonl`, including `raw_alpha_dino`, `raw_alpha_self`, router entropy, per-layer alpha, and gate EMAs.
- What is still missing is a run-level contract row that makes it impossible to confuse a stale launcher or stale git revision with the intended safe configuration.
- Current validator prevents inconsistent floors and zero-warmup, but it does not reject weak dual-source defaults that are very likely to collapse immediately.

Minimal contract conclusion:
- Dual-source runs must emit one explicit contract record at launch and one explicit starvation status in metrics.
- Fail-fast should reject unsafe low-protection combinations, rather than silently allowing a run that is almost guaranteed to degrade to self-only.

### Agent C: Operations / Rollback
Finding:
- The fix should bias toward predictable training, not a large adaptive controller.
- The safest restart path is staged dual-source release: hold self closed early, then reopen under a nontrivial DINO floor, and monitor raw alpha separately from floored alpha.
- If this harms diffusion loss or throughput, the rollback path should disable only the aggressive protection layer, not delete the whole S3A stack.

Minimal ops conclusion:
- Keep the mechanism simple: staged self unlock + persistent source0 floor + starvation warning/fail-fast.
- Avoid adding entropy regularizers or new losses as the first-line fix.

## Engineering Fix Package

### 1. Run Contract
Must print at launch and write once into `metrics.jsonl` as `record_type=contract`:
- `git_revision`
- `metrics_schema_version`
- `s3a_use_ema_source`
- `s3a_enable_selective_gate`
- `s3a_self_warmup_steps`
- `s3a_dino_alpha_floor`
- `s3a_dino_alpha_floor_steps`
- `s3a_protect_source0_min_alpha`
- `s3a_gate_reopen_probe_alpha_floor`
- `s3a_utility_probe_mode`
- `s3a_gate_utility_off_threshold`
- `s3a_gate_utility_on_threshold`
- `s3a_gate_patience`
- `s3a_gate_reopen_patience`
- `s3a_feat_weight`
- `s3a_attn_weight`
- `s3a_spatial_weight`
- `s3a_layer_indices`
- `s3a_train_schedule`
- `s3a_diff_schedule`
- derived fields: `dual_source_contract_id`, `expected_self_unlock_step`, `max_source0_floor`, `dual_source_safe_default`

Must be written every metric window:
- `raw_alpha_dino`
- `alpha_dino`
- `alpha_dino_min_layer`
- `alpha_dino_max_layer`
- `router_entropy_norm`
- `gate_self`
- new: `source0_floor_active`
- new: `dino_starved` (`1` when `raw_alpha_dino` stays below threshold after unlock for N windows)
- new: `dual_source_alive` (`1` when `raw_alpha_dino` and per-layer min alpha remain above contract thresholds)

Fail-fast combinations:
- `use_ema_source=1 && self_warmup_steps < 2000`
- `use_ema_source=1 && protect_source0_min_alpha < 0.10`
- `use_ema_source=1 && dino_alpha_floor_steps < self_warmup_steps`
- `use_ema_source=1 && selective_gate=1 && utility_probe_mode != policy_loo`
- `use_ema_source=1 && protect_source0_min_alpha + gate_reopen_probe_alpha_floor > 1.0`
- `use_ema_source=1 && collapse_alpha_threshold < protect_source0_min_alpha`
- `use_ema_source=1 && attn_weight == 0 && spatial_weight == 0`

Rationale:
- The observed failure happened before any recovery logic mattered.
- The contract should therefore guarantee a nontrivial DINO training share through the self-unlock phase.

### 2. Minimal Code Changes (<= 5 points)

#### Change 1: Add a launch-time contract record
Files / functions:
- `train_s3a_multisource_dinov2.py`
- around the launch logging block and `metrics_jsonl` setup

Pseudo-patch:
- Build a small `contract_row` dict once after `validate_args()` and after `s3a_layer_indices` are finalized.
- Write it to `metrics.jsonl` before the first metric row.
- Include a derived `dual_source_contract_id` string.

Reason:
- This prevents re-running a stale launcher while thinking the safe contract is active.

#### Change 2: Add explicit starvation status metrics
Files / functions:
- `train_s3a_multisource_dinov2.py`
- metric aggregation path near the existing `metric_row`

Pseudo-patch:
- Maintain a small integer counter `dino_starve_windows` on rank 0.
- Increment when `train_steps >= self_warmup_steps` and `raw_alpha_dino < 0.02` and `alpha_dino_min_layer < 0.05`.
- Reset otherwise.
- Emit `dino_starved = float(dino_starve_windows >= 3)` and `dual_source_alive = float(not dino_starved)`.

Reason:
- Current logs can show collapse after the fact, but there is no contract-level boolean saying the dual-source premise is broken.

#### Change 3: Harden `validate_args()` for safe dual-source defaults
Files / functions:
- `train_s3a_multisource_dinov2.py`
- `validate_args()`

Pseudo-patch:
- Under `if args.s3a and args.s3a_use_ema_source:` add:
  - require `self_warmup_steps >= 2000` unless explicit unsafe override
  - require `protect_source0_min_alpha >= 0.10`
  - require `dino_alpha_floor_steps >= self_warmup_steps`
  - require at least one of `attn_weight` or `spatial_weight` to be positive

Reason:
- The failed trajectory shows that weak floors and early self competition are enough to kill source0 almost immediately.

#### Change 4: Stage self release in `get_source_mask()`
Files / functions:
- `train_s3a_multisource_dinov2.py`
- `S3AAlignmentHead.get_source_mask()`

Pseudo-patch:
- Keep current warmup logic.
- Add a second short ramp window after unlock, e.g. `self_ramp_steps=1000` derived from warmup or a fixed small constant.
- During the ramp, keep `mask[1]=0` for a subset of probe/training windows or equivalently force a capped self contribution by extra min-alpha to source0 in `_build_alpha`.
- Do not add a new optimizer loss; only constrain mixture policy during the vulnerable transition.

Reason:
- The observed crash happens exactly when self is available as a shortcut. A brief staged release is cheaper and simpler than adding new regularizers.

#### Change 5: Sync launcher safe defaults
Files / functions:
- `scripts/run_e0_e7_single_seed.sh`
- `run_s3a_multisource_dinov2.sh`

Pseudo-patch:
- Raise default `S3A_SELF_WARMUP_STEPS`, `S3A_PROTECT_SOURCE0_MIN_ALPHA`, and `S3A_DINO_ALPHA_FLOOR_STEPS`.
- Print the derived contract id in the launcher header.
- Refuse trailing overrides that weaken these values unless an explicit `ALLOW_UNSAFE_S3A=1` is set.

Reason:
- This failure came from a launcher/code contract mismatch. The launcher must not be a silent bypass around training-safe defaults.

### 3. Safe Default Restart Configuration
Recommended restart values for the next dual-source run:
- `s3a_use_ema_source = 1`
- `s3a_enable_selective_gate = 1`
- `s3a_utility_probe_mode = policy_loo`
- `s3a_self_warmup_steps = 4000`
- `s3a_dino_alpha_floor = 0.25`
- `s3a_dino_alpha_floor_steps = 6000`
- `s3a_protect_source0_min_alpha = 0.15`
- `s3a_gate_reopen_probe_alpha_floor = 0.10`
- `s3a_gate_patience = 500`
- `s3a_gate_reopen_patience = 200`
- `s3a_gate_utility_off_threshold = 0.002`
- `s3a_gate_utility_on_threshold = 0.005`
- `s3a_feat_weight = 1.0`
- `s3a_attn_weight = 0.5`
- `s3a_spatial_weight = 0.5`
- `s3a_collapse_alpha_threshold = 0.15`
- `s3a_collapse_self_threshold = 0.90`
- `s3a_collapse_windows = 3`
- `s3a_collapse_auto_mitigate = 1`

Reasoning:
- `self_warmup_steps=4000`: long enough to avoid the early self shortcut, shorter than a full phase redesign.
- `dino_alpha_floor=0.25` for `6000` steps: keeps source0 meaningfully present through and slightly beyond self unlock.
- `protect_source0_min_alpha=0.15`: persistent guarantee that dual-source does not collapse into numerically trivial DINO contribution.
- `reopen_probe_alpha_floor=0.10`: enough signal for add-one recovery probes without consuming the whole simplex budget.
- Keep gate thresholds unchanged initially to isolate the fix to mixture policy, not utility semantics.

### 4. Rollback Strategy
If the restart harms `loss_diff`, convergence speed, or final quality:
1. First rollback: keep dual-source but reduce persistence, not warmup.
- lower `protect_source0_min_alpha` from `0.15` to `0.10`
- keep `self_warmup_steps=4000`
- keep `dino_alpha_floor=0.25`, `floor_steps=6000`

2. Second rollback: shorten the transient floor, keep persistent floor.
- `dino_alpha_floor=0.20`
- `dino_alpha_floor_steps=4000`
- keep `protect_source0_min_alpha=0.10`

3. Third rollback: disable staged self release only.
- keep persistent floor metrics and starvation checks
- remove the extra post-unlock ramp logic if diffusion loss regresses

4. Last resort baseline:
- `use_ema_source=0` for matched self-off ablation
- verify whether the observed regression is coming from the second source or from unrelated instability

Rule:
- Never roll back the contract logging and starvation diagnostics.
- Roll back only protection strength, one layer at a time.

## Must-Have vs Optional

### Must-Have
- launch-time `contract_row`
- starvation booleans in metrics
- stronger `validate_args()` for dual-source safe defaults
- launcher default sync
- safe restart with longer warmup and stronger source0 floor

### Optional
- staged self-release ramp after warmup
- stricter launcher refusal of weakening overrides
- later entropy or balance regularization experiments

## Final Recommendation
Start with the four must-have engineering changes plus the safe restart config. Do not add a new regularization loss yet. The observed failure is already explained by an early policy shortcut, and the cheapest robust fix is to make the dual-source contract explicit, enforceable, and visibly alive in metrics.
