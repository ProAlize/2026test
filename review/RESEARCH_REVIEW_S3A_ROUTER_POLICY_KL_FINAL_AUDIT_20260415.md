# Research Review: S3A router-policy KL final audit (2026-04-15)

## Scope
- File reviewed: `train_s3a_multisource_dinov2.py`
- Runs compared:
  - `r5`: `monitor_logs/s3a_fix_smoke_20260415_r5.log`
  - `r6`: `monitor_logs/s3a_fix_smoke_20260415_r6.log`
- Question:
  - whether the new router-policy KL mechanism is mechanistically coherent
  - whether it actually fixes raw router collapse rather than only masking it
  - whether a residual "floor-only DINO alive" risk remains

## Review mode
- Local fallback for `$research-review`; this session did not expose subagent delegation APIs.

## Code-path audit

### What changed
- `source0_min_alpha_at_step()` centralizes the deployed DINO floor:
  - `train_s3a_multisource_dinov2.py:418`
- `router_policy_kl_and_gap_per_sample()` defines:
  - `KL(raw_alpha || policy_alpha)`
  - `L1 gap(raw_alpha, policy_alpha)`
  - `train_s3a_multisource_dinov2.py:432`
- The deployed policy `alpha` is still built after gate/floor correction:
  - `train_s3a_multisource_dinov2.py:1133`
  - `train_s3a_multisource_dinov2.py:1153`
- The KL target is detached policy alpha:
  - `train_s3a_multisource_dinov2.py:1161`
- The KL term is added directly into the layer loss:
  - `train_s3a_multisource_dinov2.py:1331`
- Raw/policy diagnostics are now logged:
  - `train_s3a_multisource_dinov2.py:1353`
  - `train_s3a_multisource_dinov2.py:2758`
  - `train_s3a_multisource_dinov2.py:2816`

### Mechanism judgment
- This is self-consistent.
- The detached target matters: the trainer is no longer only editing the executed `alpha`; it now explicitly teaches the latent router to match the already-deployed floor/gate policy.
- Therefore this patch addresses the exact old disconnect: "controller changes execution, but router never learns from those changes."

## Evidence from r5 vs r6

### r5 (`lambda=0.02`) does not fully repair raw collapse
- At step 600:
  - `raw_dino=0.025`
  - `a_dino=0.093`
  - `rKL=0.0380`
  - `rGap=0.136`
  - `dino_starved_alarm=1`
  - `mitigate=1`
- Evidence:
  - `monitor_logs/s3a_fix_smoke_20260415_r5.log:176`
  - `refine-logs/s3a_smoke/20260415_s3a_fix_smoke_r5/E7_s3a_spread4_dualsrc_gate_holistic_cos__w200_f0p1_fmin0p05_fs8000_rkl0p02_p10_upolicy_loo_rp0p05_cw3_m1/DiT-XL-2-seed0-20260415-214348-f8d91b-s3a-dinov2-lam0.1-trainconstant-diffcosine/metrics.jsonl:7`
- Interpretation:
  - lambda `0.02` is too weak; the raw router is still materially below the executed policy and still trips starvation mitigation.

### r6 (`lambda=0.1`) repairs the raw/effective split in the short run
- At step 500:
  - `raw_dino=0.081`
  - `a_dino=0.094`
  - `rKL=0.0016`
  - `rGap=0.026`
  - `dino_starved=0`
  - `mitigate=0`
- At step 600:
  - `raw_dino=0.089`
  - `a_dino=0.093`
  - `rKL=0.0002`
  - `rGap=0.009`
  - `dino_starved_alarm=0`
  - `mitigate=0`
- Evidence:
  - `monitor_logs/s3a_fix_smoke_20260415_r6.log:175`
  - `monitor_logs/s3a_fix_smoke_20260415_r6.log:176`
  - `refine-logs/s3a_smoke/20260415_s3a_fix_smoke_r6/E7_s3a_spread4_dualsrc_gate_holistic_cos__w200_f0p1_fmin0p05_fs8000_rkl0p1_p10_upolicy_loo_rp0p05_cw3_m1/DiT-XL-2-seed0-20260415-215850-a3111a-s3a-dinov2-lam0.1-trainconstant-diffcosine/metrics.jsonl:6`
  - `refine-logs/s3a_smoke/20260415_s3a_fix_smoke_r6/E7_s3a_spread4_dualsrc_gate_holistic_cos__w200_f0p1_fmin0p05_fs8000_rkl0p1_p10_upolicy_loo_rp0p05_cw3_m1/DiT-XL-2-seed0-20260415-215850-a3111a-s3a-dinov2-lam0.1-trainconstant-diffcosine/metrics.jsonl:7`
- Interpretation:
  - The improvement occurs while `gate_self=1` and `mitigate=0`.
  - That means the repair is acting on the latent router path itself, not only via post-hoc masking.
  - Compared with the earlier raw-collapse regime (`raw_dino=0.006` in `r4`), this is a real fix of the raw/effective drift bug.

## Residual risk

### The deeper "DINO alive only because of floor" risk is still present
- In both `r5` and `r6`, the effective DINO share above floor is nearly zero from step 400 onward:
  - `r5` step 600: `alpha_dino_above_floor=0.0006`
  - `r6` step 600: `alpha_dino_above_floor=0.0006`
- Evidence:
  - `refine-logs/s3a_smoke/20260415_s3a_fix_smoke_r5/E7_s3a_spread4_dualsrc_gate_holistic_cos__w200_f0p1_fmin0p05_fs8000_rkl0p02_p10_upolicy_loo_rp0p05_cw3_m1/DiT-XL-2-seed0-20260415-214348-f8d91b-s3a-dinov2-lam0.1-trainconstant-diffcosine/metrics.jsonl:7`
  - `refine-logs/s3a_smoke/20260415_s3a_fix_smoke_r6/E7_s3a_spread4_dualsrc_gate_holistic_cos__w200_f0p1_fmin0p05_fs8000_rkl0p1_p10_upolicy_loo_rp0p05_cw3_m1/DiT-XL-2-seed0-20260415-215850-a3111a-s3a-dinov2-lam0.1-trainconstant-diffcosine/metrics.jsonl:7`
- Also at step 600 in `r6`:
  - `dual_source_alive=0`
  - `dual_synergy_margin=-0.7569`
  - `utility_dino=-0.7569`
- Evidence:
  - `monitor_logs/s3a_fix_smoke_20260415_r6.log:176`
- Interpretation:
  - The new KL has largely taught `raw_alpha` to track the deployed floor-corrected policy.
  - It has not yet shown that DINO contributes meaningful above-floor share or positive dual-source gain.
  - So the raw collapse bug is repaired more cleanly, but the broader scientific claim "DINO is genuinely alive/useful in coexistence" is still unproven.

## Final judgment

### 1. Is the mechanism self-consistent and a real fix rather than a cover-up?
- Yes for the narrow raw-router-collapse bug.
- Evidence: the KL target is detached executed policy, the KL is in the optimization path, and `r6` improves `raw_dino` and eliminates mitigation without needing forced gate shutdown.

### 2. Is there still a residual "DINO effective only by floor" risk?
- Yes.
- Evidence: `alpha_dino_above_floor` is still essentially zero in both runs, and `utility_dino` remains negative.

### 3. Minimal next action
- No further code patch is justified yet.
- Run one unchanged `lambda=0.1` canary past the current floor-dominated window, and require:
  - `mitigate=0`
  - `raw_dino > source0_floor_active` for multiple consecutive windows once the floor decays further
  - preferably `alpha_dino_above_floor > 0.02`
- If that fails, the remaining problem is not raw/policy drift anymore; it is floor-only coexistence.

### 4. Verdict
- `partial`

Reason:
- `lambda=0.1` appears to close the raw/effective drift defect.
- But the stronger claim "DINO is alive without merely leaning on the floor contract" is still not established by these short runs.
