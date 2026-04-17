# Research Review: S3A r5/r6 Result-to-Claim Final Gate (2026-04-15)

## Metadata
- Trigger: user requested `$research-review` final claim gate based on latest code and r5/r6 smoke logs.
- Mode: local fallback review because reviewer delegation APIs are not exposed in this session.
- Scope:
  - [`train_s3a_multisource_dinov2.py`](/home/liuchunfa/2026qjx/2026test/train_s3a_multisource_dinov2.py)
  - [`monitor_logs/s3a_fix_smoke_20260415_r5.log`](/home/liuchunfa/2026qjx/2026test/monitor_logs/s3a_fix_smoke_20260415_r5.log)
  - [`monitor_logs/s3a_fix_smoke_20260415_r6.log`](/home/liuchunfa/2026qjx/2026test/monitor_logs/s3a_fix_smoke_20260415_r6.log)

## Structured verdict
- claim_supported: `partial`
- confidence: `high`
- core paper claim status for "dual-source collaboration is solved": `no`

## Evidence

### 1. What the latest code now guarantees

The latest trainer has a real safe dual-source contract for source0 survival:

- source0 floor is enforced by `source0_min_alpha_at_step()` in [`train_s3a_multisource_dinov2.py:418`](/home/liuchunfa/2026qjx/2026test/train_s3a_multisource_dinov2.py#L418) and injected into deployed alpha in [`train_s3a_multisource_dinov2.py:1133`](/home/liuchunfa/2026qjx/2026test/train_s3a_multisource_dinov2.py#L1133)
- source0 cannot be selectively gated off in [`train_s3a_multisource_dinov2.py:808`](/home/liuchunfa/2026qjx/2026test/train_s3a_multisource_dinov2.py#L808) and is force-restored in [`train_s3a_multisource_dinov2.py:893`](/home/liuchunfa/2026qjx/2026test/train_s3a_multisource_dinov2.py#L893)
- unsafe zero-warmup and zero-source0-floor settings are rejected in [`train_s3a_multisource_dinov2.py:3358`](/home/liuchunfa/2026qjx/2026test/train_s3a_multisource_dinov2.py#L3358) and [`train_s3a_multisource_dinov2.py:3380`](/home/liuchunfa/2026qjx/2026test/train_s3a_multisource_dinov2.py#L3380)
- the trainer now logs `alpha_dino_above_floor`, `dino_starved`, `dual_source_alive`, and `dual_synergy_margin` in [`train_s3a_multisource_dinov2.py:2637`](/home/liuchunfa/2026qjx/2026test/train_s3a_multisource_dinov2.py#L2637) to [`train_s3a_multisource_dinov2.py:2777`](/home/liuchunfa/2026qjx/2026test/train_s3a_multisource_dinov2.py#L2777)

This is enough to support an engineering claim that exact-zero DINO collapse is blocked under the safe contract.

### 2. What the latest code still does not prove

The router is still not a strong "source reliability" mechanism:

- router inputs are pooled student tokens plus timestep/phase/layer embeddings only, in [`train_s3a_multisource_dinov2.py:617`](/home/liuchunfa/2026qjx/2026test/train_s3a_multisource_dinov2.py#L617) to [`train_s3a_multisource_dinov2.py:657`](/home/liuchunfa/2026qjx/2026test/train_s3a_multisource_dinov2.py#L657)
- it does not directly ingest per-source utility, per-source quality, or source-specific evidence before producing `raw_alpha`
- deployed `alpha` is post-processed by the floor logic, so non-zero effective DINO weight can still be floor-supported rather than router-chosen

Therefore current code can support guarded coexistence claims, but not a strong mechanism claim that it learns source reliability.

### 3. What r5 and r6 actually show

Both 2026-04-15 smoke runs complete 600 steps without runtime failure:

- r5 reaches final checkpoint in [`s3a_fix_smoke_20260415_r5.log:176`](/home/liuchunfa/2026qjx/2026test/monitor_logs/s3a_fix_smoke_20260415_r5.log#L176) to [`s3a_fix_smoke_20260415_r5.log:181`](/home/liuchunfa/2026qjx/2026test/monitor_logs/s3a_fix_smoke_20260415_r5.log#L176)
- r6 reaches final checkpoint in [`s3a_fix_smoke_20260415_r6.log:176`](/home/liuchunfa/2026qjx/2026test/monitor_logs/s3a_fix_smoke_20260415_r6.log#L176) to [`s3a_fix_smoke_20260415_r6.log:181`](/home/liuchunfa/2026qjx/2026test/monitor_logs/s3a_fix_smoke_20260415_r6.log#L176)

But the collaboration story is still not closed:

- both runs briefly enter above-floor coexistence at step 300, but `synergy_margin` stays negative:
  - r5 step 300: `dual_alive=1`, `synergy_margin=-2.9720` in [`s3a_fix_smoke_20260415_r5.log:173`](/home/liuchunfa/2026qjx/2026test/monitor_logs/s3a_fix_smoke_20260415_r5.log#L173)
  - r6 step 300: `dual_alive=1`, `synergy_margin=-4.2289` in [`s3a_fix_smoke_20260415_r6.log:173`](/home/liuchunfa/2026qjx/2026test/monitor_logs/s3a_fix_smoke_20260415_r6.log#L173)
- by steps 400-600, both runs are effectively floor-dominated:
  - r5 step 600: `a_dino_above_floor=0.001`, `raw_dino=0.025`, `dual_alive=0`, `synergy=0`, `dino_starved_alarm=1` in [`s3a_fix_smoke_20260415_r5.log:176`](/home/liuchunfa/2026qjx/2026test/monitor_logs/s3a_fix_smoke_20260415_r5.log#L176)
  - r6 step 600: `a_dino_above_floor=0.001`, `raw_dino=0.089`, `dual_alive=0`, `synergy=0`, `dino_starved_alarm=0` in [`s3a_fix_smoke_20260415_r6.log:176`](/home/liuchunfa/2026qjx/2026test/monitor_logs/s3a_fix_smoke_20260415_r6.log#L176)
- the metrics JSONL confirms `dual_synergy_supported=0.0` at every recorded post-warmup window for both runs:
  - r5 final row: [`metrics.jsonl`](/home/liuchunfa/2026qjx/2026test/refine-logs/s3a_smoke/20260415_s3a_fix_smoke_r5/E7_s3a_spread4_dualsrc_gate_holistic_cos__w200_f0p1_fmin0p05_fs8000_rkl0p02_p10_upolicy_loo_rp0p05_cw3_m1/DiT-XL-2-seed0-20260415-214348-f8d91b-s3a-dinov2-lam0.1-trainconstant-diffcosine/metrics.jsonl)
  - r6 final row: [`metrics.jsonl`](/home/liuchunfa/2026qjx/2026test/refine-logs/s3a_smoke/20260415_s3a_fix_smoke_r6/E7_s3a_spread4_dualsrc_gate_holistic_cos__w200_f0p1_fmin0p05_fs8000_rkl0p1_p10_upolicy_loo_rp0p05_cw3_m1/DiT-XL-2-seed0-20260415-215850-a3111a-s3a-dinov2-lam0.1-trainconstant-diffcosine/metrics.jsonl)

### 4. r6 is better than r5, but only on anti-starvation diagnostics

Relative to r5, r6 does improve short-horizon retention:

- final `raw_dino`: `0.025` -> `0.089`
- final `router_policy_gap`: `0.136` -> `0.009`
- final `dino_starved_alarm`: `1` -> `0`

That supports a narrow claim that higher router-KL regularization improved the smoke-run anti-starvation behavior.

It does not support a claim that collaboration is solved, because:

- `dual_alive=0` at the final window in both runs
- `dual_synergy_margin < 0` at every logged post-warmup window in both runs
- `loss_fused_probe` never beats the better single-source probe

## Allow claims

1. The latest safe dual-source contract blocks the old exact-zero DINO collapse mode by construction.
2. The latest implementation is engineering-stable for the 600-step dual-source smoke on 2026-04-15.
3. The trainer now has decision-grade observability for floor-supported DINO, starvation alarms, and probe-defined synergy.
4. r6 is better than r5 on short-horizon anti-starvation diagnostics, but only in that narrow smoke-run sense.
5. Brief above-floor dual-source coexistence appears after warmup, but it is not yet beneficial.

## Disallow claims

1. "Dual-source collaboration is solved."
2. "The method now demonstrates positive dual-source synergy."
3. "The router learns source reliability."
4. "DINO makes a sustained material contribution beyond the enforced floor after warmup."
5. "Selective gating has been validated as the reason for dual-source gains."
6. "The current dual-source path beats the best single-source path."
7. Any long-horizon or paper-level claim that the dual-source mechanism is already scientifically closed.

## Minimal supplementary experiment

Run one fresh `10k` dual-source canary with the current r6 contract and use the already-logged probe controls. Only upgrade the story to "dual-source collaboration is solved" if, after warmup, the run shows sustained:

- `dual_source_alive = 1`
- `dual_synergy_margin > 0`
- `dino_starved_alarm = 0`

for multiple consecutive log windows and still satisfies those conditions at the final window.

This is the smallest sufficient supplement because the code already computes the matched DINO-only and self-only probe losses inside the same run.

## Bottom line

As of 2026-04-15, the latest code plus r5/r6 logs support an **engineering anti-collapse** story, not a **solved dual-source collaboration** story.
