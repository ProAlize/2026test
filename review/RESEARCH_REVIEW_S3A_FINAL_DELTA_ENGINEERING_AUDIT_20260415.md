# S3A Final Delta Engineering Audit (2026-04-15)

## Scope
- Files inspected:
  - `train_s3a_multisource_dinov2.py`
  - `scripts/run_e0_e7_single_seed.sh`
  - `run_s3a_multisource_dinov2.sh`
  - `scripts/monitor_training_20m.sh`
  - `RESEARCH_REVIEW_S3A_DINO_RECOVERY_LOOP_20260415.md`
- Focus:
  - `dino_starved_alarm`
  - `dual_synergy_margin` / `dual_synergy_supported`
  - monitor `dual_not_alive_rounds` logic

## Method
- Local fallback audit for `$research-review` because this session does not expose `spawn_agent` / `send_input`.
- Reviewed final diffs and surrounding code contracts.
- Ran static sanity checks:
  - `python -m py_compile train_s3a_multisource_dinov2.py`
  - `bash -n scripts/run_e0_e7_single_seed.sh`
  - `bash -n run_s3a_multisource_dinov2.sh`
  - `bash -n scripts/monitor_training_20m.sh`

## Findings

### 1. High: `dino_starved_alarm` is now an actuator trigger without the utility/probe-improvement guards that defined the original collapse contract
- Evidence:
  - `dino_starved` is raised from alpha occupancy only:
    - `train_s3a_multisource_dinov2.py:2616`
  - `dino_starved_alarm` is then OR-ed directly into auto-mitigation:
    - `train_s3a_multisource_dinov2.py:2666`
    - `train_s3a_multisource_dinov2.py:2680`
  - The original collapse trigger required both positive DINO utility and fused-probe advantage over self-only:
    - `train_s3a_multisource_dinov2.py:2643`
    - `train_s3a_multisource_dinov2.py:2651`
    - `train_s3a_multisource_dinov2.py:2652`
  - The recovery note itself records that mitigation fired while `dual_synergy_margin` remained negative:
    - `RESEARCH_REVIEW_S3A_DINO_RECOVERY_LOOP_20260415.md:103`
    - `RESEARCH_REVIEW_S3A_DINO_RECOVERY_LOOP_20260415.md:109`
- Why this is a regression:
  - Final delta turns a monitoring symptom into the same control action used for evidence-backed collapse, but without preserving the original "DINO is actually helping" guardrails.
  - In the current code, persistent low above-floor DINO share is enough to force self shutdown, even when probe losses do not support DINO as the better or synergistic source.
- Impact:
  - The trainer can now enter mitigation windows that are inconsistent with the measured probe objective, creating a real risk of over-correcting into DINO-only behavior.

## Verdict
- Not clean.
- High finding count: `1`
- Additional Medium findings in the requested final-delta scope: `0`
