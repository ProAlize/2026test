# Research Review: S3A Failure Postmortem (2026-04-14)

## Metadata
- Trigger: user request `$research-review` to audit why current S3A design failed.
- Scope: current E7 run + E0-E7 experiment design + S3A code contract.
- Date: 2026-04-14.
- External reviewers (all `gpt-5.4`, `xhigh`):
  - Method/Narrative: `019d8c1f-fe83-7a70-a5b8-45d0a19119c5`
  - Causal Experiment Design: `019d8c20-3352-73a1-be74-ce538f89ce8a`
  - Code/Contract Engineering: `019d8c20-5389-7490-9689-a28f038fc4bb`

## Context Reviewed
- Current run log: `monitor_logs/s3a_e7only_xl_s0_160k_v1.log`
- Curve audit: `refine-logs/E7_CURVE_AUDIT_20260414.md`
- Previous audit note: `refine-logs/E7_RESEARCH_REVIEW_AUDIT_20260414.md`
- E0-E7 launcher: `scripts/run_e0_e7_single_seed.sh`
- Core trainer: `train_s3a_multisource_dinov2.py`
- Design docs: `docx/plan.md`, `docx/idea_s3a_with_audit_feedback.md`

## Round-by-Round Summary

## Round 1: Independent Critical Review
Common conclusion from all reviewers:
- Current E7 does not support the claim "effective DINO injection".
- Observed behavior is consistent with rapid collapse to self-dominant shortcut:
  - `alpha_dino` falls to near-zero very early.
  - `alpha_self` saturates to 1.
  - `feat/attn/spatial` near-zero thereafter only proves fused-target matching, not DINO transfer.

Main criticisms:
1. **Protocol confound**: launcher hardcodes `--s3a-self-warmup-steps 0` (step-0 self competition).
2. **Contract confound**: unconstrained softmax source competition has no DINO retention/floor/prior.
3. **Shortcut amplifier**: self branch is easier and has trainable target-side path (`ema_adapters`).
4. **Gate semantics mismatch**: protects DINO availability, not DINO usage.
5. **Causality gap**: missing matched controls (`no-align`, matched `DINO-only`, placebo DINO).

## Round 2: Convergence and Decision Gates
Reviewers converged to a minimum rescue strategy:
1. First fix protocol confound (warmup) and observability.
2. Then test whether DINO remains alive without adding many new components.
3. Only promote to longer runs if both mechanism and quality are directionally positive.

Consensus score impact:
- If minimum rescue succeeds: paper viability improves to roughly borderline (about 5.5-6/10).
- If rescue fails: DINO-injection headline should be terminated (about 2-3/10).

## Failure Tree (Symptom -> Direct Cause -> Design Root Cause)
1. Symptom: `alpha_dino` collapses (approx 0 by early stage).
- Direct cause: router minimizes fused alignment by selecting easiest source.
- Root cause: objective never enforces persistent DINO usage.

2. Symptom: `feat/attn/spatial` quickly approach tiny values.
- Direct cause: fused target rapidly becomes near-pure self source.
- Root cause: alignment metrics are fused-only; no source-specific anchoring.

3. Symptom: collapse occurs before DINO can anchor.
- Direct cause: self source enabled from step 0 under constant training schedule.
- Root cause: protocol violates staged teacher logic.

4. Symptom: gate does not rescue DINO.
- Direct cause: source0 protection is availability-only; alpha can still go to ~0.
- Root cause: gate objective mismatched with scientific goal.

## Responsibility Weights (Consensus)
- Method contract errors: **65%**
- Experimental protocol errors: **30%**
- Implementation bugs: **5%**

Root-cause breakdown:
- Unconstrained fused-source objective: 35%
- Step-0 self competition (`warmup=0`): 25%
- Self shortcut amplification (`ema_adapters` trainable): 20%
- Router not truly source-aware: 10%
- Availability-only source0 protection: 10%

## What Claims Are Safe Right Now
Allowed now:
- "Current dual-source unrestricted routing tends to collapse to self-dominant behavior under present E7 contract."

Not allowed now:
- "S3A has demonstrated effective DINO injection."
- "Holistic dual-source mechanism is validated."
- "Gate provides proven DINO-usefulness gains."

## Claim-Safe Matrix (Post-Fix Outcomes)
1. If `N2 > N1`, `N4 > current E7`, `N4 > placebo`:
- Allowed: real DINO contributes causally under corrected contract.
- Not allowed: default E7 already validated DINO injection.

2. If `N2 > N1`, but `N4 ~ placebo`:
- Allowed: DINO-only useful, current dual-source routing fails to utilize DINO.
- Not allowed: dual-source DINO injection success claim.

3. If `N2 ~ N1`, `N4 ~ N1`:
- Allowed: no evidence for DINO benefit in current pipeline.
- Not allowed: any external-DINO effectiveness headline.

## Prioritized TODO (with Cost)

## P0 (must do before any new long run)
1. Remove `warmup=0` hardcoding in launchers and add unsafe-config fail-fast.
2. Add collapse telemetry and alarms:
- `raw_alpha_*`, per-layer `alpha_dino`, router entropy,
- `loss_dino_only`, `loss_self_only`, collapse alarm.
3. Run causality-first minimum package (`N1`, `N2`, `N4`) at 10k, 1 seed.

Estimated cost on `4x RTX5880 Ada`:
- `N1`: 13-15 GPUh
- `N2`: 11-13 GPUh
- `N4`: 13-15 GPUh
- Total sequential: **37-43 GPUh** (about **9.4-10.6h wall-clock** on one 4-GPU node)

## P1 (if P0 indicates DINO can survive)
4. Add early DINO retention (alpha floor or KL prior for first 10k).
5. Freeze/bypass trainable `ema_adapters` early to reduce shortcut pressure.
6. Re-test with short canary, then promote winner to 40k.

## P2 (only if still unstable)
7. Upgrade gate from confidence proxy to utility-based source control.
8. Extend source-aware router input with source-specific detached stats.

## 48-Hour Minimum Verification Package
- `N1` NoAlign-match: `--s3a-lambda 0.0`
- `N2` DINO-only-match: `--no-s3a-use-ema-source --no-s3a-enable-selective-gate`
- `N4` DualSrc-warmup5k-Gate: `--s3a-self-warmup-steps 5000`

Decision table:
- Continue: `N2 > N1` and `N4 > N1` and no immediate post-warmup DINO collapse.
- Patch: `N2 > N1` but `N4 <= N2` or collapses again.
- Pivot: `N2 ~ N1` and `N4 ~ N1`.

## Mock Review (Consensus)
- Summary: Current evidence shows self-dominant shortcut collapse, not validated DINO injection.
- Strengths: good problem framing, visible instrumentation, structured ablation scaffold.
- Weaknesses: objective/claim mismatch, missing matched causal controls, confounded E7 protocol.
- Score now: **3/10**.
- Confidence: **4/5**.
- What moves toward accept: corrected contract + source-specific causal evidence + matched controls showing real DINO contribution.

## Final Consensus
This is not a random failure; it is a contract-level failure mode. 
Current E7 should be treated as a falsification signal for the existing design contract, not as evidence for DINO injection effectiveness.
