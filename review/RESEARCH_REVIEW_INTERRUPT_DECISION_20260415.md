# Research Review: Interrupt Decision for Current S3A Canary (2026-04-15)

## Reviewer setup
- Skill: `$research-review`
- External reviewer agent: `gpt-5.4` with `xhigh` reasoning
- Agent id: `019d91ba-61e0-7ce2-a36f-bff5578b5f81`

## Question
Should current run be interrupted now before code fixes, or continue collecting baseline evidence?

## Run under review
- Trainer: `/home/liuchunfa/2026qjx/2026test/train_s3a_multisource_dinov2.py`
- Session/log: `/home/liuchunfa/2026qjx/2026test/monitor_logs/s3a_canary_10k_r6.log`
- Resolved config:
  - `/home/liuchunfa/2026qjx/2026test/refine-logs/s3a_canary/20260415_s3a_canary10k_r6/E7_s3a_spread4_dualsrc_gate_holistic_cos__w200_f0p1_fmin0p05_fs8000_rkl0p1_uf0_uw0_p10_upolicy_loo_rp0p05_cw3_m1/DiT-XL-2-seed0-20260415-223109-105b02-s3a-dinov2-lam0.1-trainconstant-diffcosine/resolved_args.json`
- Key args confirmed:
  - `max_steps=10000`
  - `s3a_self_warmup_steps=200`
  - `s3a_dino_alpha_floor=0.1`
  - `s3a_dino_alpha_floor_steps=8000`
  - `s3a_protect_source0_min_alpha=0.05`
  - `s3a_router_policy_kl_lambda=0.1`
  - `s3a_gate_utility_off_threshold=0.002`
  - `s3a_gate_utility_on_threshold=0.005`

## Observed behavior summary
- After warmup, persistent regime:
  - `a_dino_above_floor ≈ 0.001`
  - `dual_alive = 0`
  - `synergy_margin < 0` for a long window
  - no meaningful recovery signal
- Checkpoint already saved at step 2000.

## Reviewer decision (verbatim class)
- **Decision:** `STOP_NOW`

## Reviewer rationale (condensed)
1. Failure regime appears early (~step 400) and persists through step 2000+; evidence already sufficient.
2. Warmup handoff bias + floor-blind alarm criteria imply further run time is low-value for decision making.
3. Waiting to next checkpoint (4000) is not decision-relevant because a valid defective-regime checkpoint already exists.

## Evidence to preserve (reviewer checklist)
1. Resolved args file.
2. Existing logic/interface review doc:
   - `/home/liuchunfa/2026qjx/2026test/RESEARCH_REVIEW_LOGIC_INTERFACE_20260415.md`
3. Key log window from steps `100/200/300/400/2000/2300`:
   - `/home/liuchunfa/2026qjx/2026test/monitor_logs/s3a_canary_10k_r6.log`
4. Metrics stream:
   - `/home/liuchunfa/2026qjx/2026test/refine-logs/s3a_canary/20260415_s3a_canary10k_r6/E7_s3a_spread4_dualsrc_gate_holistic_cos__w200_f0p1_fmin0p05_fs8000_rkl0p1_uf0_uw0_p10_upolicy_loo_rp0p05_cw3_m1/DiT-XL-2-seed0-20260415-223109-105b02-s3a-dinov2-lam0.1-trainconstant-diffcosine/metrics.jsonl`
5. Existing checkpoint + manifest:
   - `/home/liuchunfa/2026qjx/2026test/refine-logs/s3a_canary/20260415_s3a_canary10k_r6/E7_s3a_spread4_dualsrc_gate_holistic_cos__w200_f0p1_fmin0p05_fs8000_rkl0p1_uf0_uw0_p10_upolicy_loo_rp0p05_cw3_m1/DiT-XL-2-seed0-20260415-223109-105b02-s3a-dinov2-lam0.1-trainconstant-diffcosine/checkpoints/0002000.pt`
   - `/home/liuchunfa/2026qjx/2026test/refine-logs/s3a_canary/20260415_s3a_canary10k_r6/E7_s3a_spread4_dualsrc_gate_holistic_cos__w200_f0p1_fmin0p05_fs8000_rkl0p1_uf0_uw0_p10_upolicy_loo_rp0p05_cw3_m1/DiT-XL-2-seed0-20260415-223109-105b02-s3a-dinov2-lam0.1-trainconstant-diffcosine/checkpoints/0002000.pt.sha256.json`

## Action recommendation
- Stop current run now.
- Implement warmup-handoff + floor-relative controller fixes.
- Relaunch a fresh canary from step 0 (do not resume this run for main claim path).
