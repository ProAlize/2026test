# Research Review: S3A Logic & Interface (2026-04-15)

## Scope
- Target: `train_s3a_multisource_dinov2.py` and `scripts/run_e0_e7_single_seed.sh`
- Focus: logic correctness, interface/contract consistency, and operational auditability
- Reviewer mode: secondary agent (`gpt-5.4`, `xhigh`), id `019d91ad-dae5-7492-aa06-9170b0e2ed00`

## Round 1 (initial external audit)
Reviewer initially produced 4 findings and marked contract as fail:
1. P0: suspected warmup-contract mismatch (default warmup=5000 vs observed early self-on behavior)
2. P1: warmup exit not gate-controlled (self can become active immediately after warmup)
3. P1: starvation/collapse controller is floor-blind and uses raw alpha in a way that can miss floor-hugging failure
4. P2: parser/launcher/resume contract inconsistencies (especially gate thresholds and backfilled resume keys)

## Correction injected before Round 2
We provided hard evidence from the actual canary `resolved_args.json`:
- run path:
  `/home/liuchunfa/2026qjx/2026test/refine-logs/s3a_canary/20260415_s3a_canary10k_r6/E7_s3a_spread4_dualsrc_gate_holistic_cos__w200_f0p1_fmin0p05_fs8000_rkl0p1_uf0_uw0_p10_upolicy_loo_rp0p05_cw3_m1/DiT-XL-2-seed0-20260415-223109-105b02-s3a-dinov2-lam0.1-trainconstant-diffcosine/resolved_args.json`
- confirmed runtime args:
  - `s3a_self_warmup_steps=200`
  - `s3a_gate_utility_off_threshold=0.002`
  - `s3a_gate_utility_on_threshold=0.005`
  - `s3a_dino_alpha_floor=0.1`
  - `s3a_dino_alpha_floor_steps=8000`
  - `s3a_protect_source0_min_alpha=0.05`

## Round 2 (corrected final audit)
Reviewer withdrew previous P0 and converged to 5 prioritized items:

### 1) P1 real bug: warmup exit is not gate-controlled
- Mechanism: warmup suppresses self via mask only; controller gate state is initialized on and not explicitly kept closed through boundary.
- Risk: self may become active immediately at warmup exit without inactive-side reopen evidence.
- Minimal fix:
  - force `source_gate_mask[:,1]=0` during warmup and at warmup boundary,
  - allow reopen only via inactive-side `policy_loo` utility pathway.

### 2) P1 real bug: starvation/collapse logic is floor-blind
- Mechanism: alarms key off absolute `alpha_dino/raw_alpha_dino` thresholds while effective floor is time-varying and can mask true collapse-above-floor.
- Risk: controller can miss exactly the regime `alpha_dino_above_floor ~ 0`.
- Minimal fix:
  - drive `dino_starved`, `collapse_window_triggered`, `dual_source_alive` by `alpha_dino_above_floor`,
  - keep raw alpha diagnostics informational, not gate criterion.

### 3) P1 objective-induced degeneration (not code bug)
- Mechanism: current objective rewards fused loss minimization; if self-only dominates, optimum is DINO-at-floor.
- Observed behavior consistent: negative `synergy_margin`, DINO near floor.
- Recommendation:
  - either reframe claim as self-dominant routing with DINO floor protection,
  - or add explicit collaboration-driving term/constraint.

### 4) P2 interface weakness: `gate_self` metric naming/semantics
- Mechanism: current `gate_self` logs effective source mask, not pure controller gate state.
- Risk: operators cannot distinguish warmup/mask effects from actual gate controller decisions.
- Minimal fix:
  - log both `self_gate_state_mean` and `self_source_mask_mean`,
  - rename current `gate_self` to `self_mask_on`.

### 5) P2 interface weakness: parser/launcher/resume drift
- Mechanism:
  - parser default off-threshold (`0.0`) differs from launcher default (`0.002`),
  - resume still backfills some behavior-critical keys.
- Risk: same experiment label can map to different behavior across entrypoints/resume.
- Minimal fix:
  - unify defaults,
  - make floor/reopen/gate thresholds strict resume-contract keys.

## Final classification
- Real bugs:
  1. warmup exit gate semantics
  2. floor-blind starvation/collapse triggering
- Objective-induced behavior:
  3. DINO floor-hugging under current optimization target
- Interface/provenance weaknesses:
  4. gate metric semantics
  5. parser/launcher/resume drift

## Ship verdict (logic/interface only)
- Verdict: **No**
- Conditions before ship:
  1. fix warmup-exit gate semantics,
  2. fix floor-relative starvation/collapse detection,
  3. improve gate observability metric naming,
  4. tighten interface contract consistency.

## Actionable minimal package (max 3 engineering changes)
1. Gate-controlled warmup handoff for source1.
2. Floor-relative alarm/controller criteria.
3. Contract hardening + metric naming clarity (parser/launcher/resume + logs).
