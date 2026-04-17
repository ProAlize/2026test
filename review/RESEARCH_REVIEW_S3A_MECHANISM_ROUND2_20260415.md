## Research Review Round-2: S3A mechanism audit after current fixes

- Date: 2026-04-15 20:23:19 CST
- Mode: local fallback review for `$research-review` because agent spawn/send APIs are not exposed in this session
- Scope: `train_s3a_multisource_dinov2.py`
- Question: whether DINO starvation is now engineering-controllable, and what minimal further action is justified

### Verdict

The current S3A contract has largely closed the exact `a_dino=0` failure mode in safe dual-source mode. However, it has not fully closed the broader "DINO is present only because of a floor, not because the router really wants it" boundary. In short:

1. Exact-zero starvation is now basically blocked under the safe dual-source contract.
2. Pseudo-collaboration is still possible: `raw_alpha_dino` can collapse near zero while `alpha_dino` stays at the enforced source0 floor.
3. The current monitoring can distinguish "not literally zero" from "above-floor contribution exists", but it still cannot prove true collaboration benefit.
4. A minimal additional metric is justified. It should be observational first, not another controller.

### Findings

#### 1. P1: exact `a_dino=0` is now engineering-blocked in the safe dual-source contract

Evidence:

- Source0 floor is defined by `max(decaying dino floor, persistent source0 floor)` in [`train_s3a_multisource_dinov2.py:418`](/home/liuchunfa/2026qjx/2026test/train_s3a_multisource_dinov2.py#L418) to [`train_s3a_multisource_dinov2.py:429`](/home/liuchunfa/2026qjx/2026test/train_s3a_multisource_dinov2.py#L429).
- Dual-source validation rejects `--s3a-protect-source0-min-alpha <= 0` unless the user explicitly opts into unsafe mode, in [`train_s3a_multisource_dinov2.py:3312`](/home/liuchunfa/2026qjx/2026test/train_s3a_multisource_dinov2.py#L3312) to [`train_s3a_multisource_dinov2.py:3321`](/home/liuchunfa/2026qjx/2026test/train_s3a_multisource_dinov2.py#L3321).
- Source0 is protected from selective gate shutoff in [`train_s3a_multisource_dinov2.py:761`](/home/liuchunfa/2026qjx/2026test/train_s3a_multisource_dinov2.py#L761) to [`train_s3a_multisource_dinov2.py:764`](/home/liuchunfa/2026qjx/2026test/train_s3a_multisource_dinov2.py#L764).
- `get_source_mask()` force-restores source0 availability and falls back to source0 if a mask would become empty, in [`train_s3a_multisource_dinov2.py:846`](/home/liuchunfa/2026qjx/2026test/train_s3a_multisource_dinov2.py#L846) to [`train_s3a_multisource_dinov2.py:874`](/home/liuchunfa/2026qjx/2026test/train_s3a_multisource_dinov2.py#L874).
- The runtime alpha builder injects the source0 minimum alpha after masking, in [`train_s3a_multisource_dinov2.py:1079`](/home/liuchunfa/2026qjx/2026test/train_s3a_multisource_dinov2.py#L1079) to [`train_s3a_multisource_dinov2.py:1102`](/home/liuchunfa/2026qjx/2026test/train_s3a_multisource_dinov2.py#L1102).

Conclusion:

- Under the intended dual-source contract, exact `a_dino=0` should not persist as a steady state.
- The practical exceptions are explicit unsafe override via `--s3a-allow-unsafe-zero-source0-floor`, or a hard upstream failure where DINO features are unavailable at all.

#### 2. P1: pseudo-collaboration is still possible, because the floor can mask raw-router collapse

Evidence:

- `raw_alpha` comes directly from a softmax router in [`train_s3a_multisource_dinov2.py:595`](/home/liuchunfa/2026qjx/2026test/train_s3a_multisource_dinov2.py#L595) to [`train_s3a_multisource_dinov2.py:610`](/home/liuchunfa/2026qjx/2026test/train_s3a_multisource_dinov2.py#L610).
- `alpha` used for training is not raw router output; it is post-processed by `_build_alpha()` with source0 min-alpha injection in [`train_s3a_multisource_dinov2.py:1084`](/home/liuchunfa/2026qjx/2026test/train_s3a_multisource_dinov2.py#L1084) to [`train_s3a_multisource_dinov2.py:1102`](/home/liuchunfa/2026qjx/2026test/train_s3a_multisource_dinov2.py#L1102).
- `dino_starved` correctly uses `alpha_dino_above_floor`, but `dual_source_alive` only checks for above-floor contribution larger than `0.01`, in [`train_s3a_multisource_dinov2.py:2613`](/home/liuchunfa/2026qjx/2026test/train_s3a_multisource_dinov2.py#L2613) to [`train_s3a_multisource_dinov2.py:2630`](/home/liuchunfa/2026qjx/2026test/train_s3a_multisource_dinov2.py#L2630).

Boundary cases that remain:

1. `raw_alpha_dino ~= 0`, `alpha_dino ~= source0_floor`, `alpha_self ~= 1 - source0_floor`.
2. `alpha_dino_above_floor > 0.01`, so `dual_source_alive=1`, but fused performance is still no better than the best single source.
3. Router entropy is nonzero and self gate is open, but the effective DINO contribution is only a tiny above-floor residual.

This means the current code can distinguish "DINO not literally zero" from "DINO has some above-floor share", but not from "DINO and self are truly collaborating in a useful way."

#### 3. P2: collapse mitigation is directionally aligned, but its trigger semantics are still looser than starvation semantics

Evidence:

- Per-layer collapse accounting triggers when `alpha_dino <= collapse_alpha_threshold`, `alpha_self > collapse_self_threshold`, `utility_dino > threshold`, and `fused_probe + margin < self_only`, in [`train_s3a_multisource_dinov2.py:1310`](/home/liuchunfa/2026qjx/2026test/train_s3a_multisource_dinov2.py#L1310) to [`train_s3a_multisource_dinov2.py:1320`](/home/liuchunfa/2026qjx/2026test/train_s3a_multisource_dinov2.py#L1320).
- Windowed collapse monitoring and auto-mitigation reuse `avg_alpha_dino`, not `alpha_dino_above_floor`, in [`train_s3a_multisource_dinov2.py:2634`](/home/liuchunfa/2026qjx/2026test/train_s3a_multisource_dinov2.py#L2634) to [`train_s3a_multisource_dinov2.py:2668`](/home/liuchunfa/2026qjx/2026test/train_s3a_multisource_dinov2.py#L2668).
- Validation enforces `collapse_alpha_threshold >= protect_source0_min_alpha`, so collapse alarm stays reachable, in [`train_s3a_multisource_dinov2.py:3368`](/home/liuchunfa/2026qjx/2026test/train_s3a_multisource_dinov2.py#L3368) to [`train_s3a_multisource_dinov2.py:3375`](/home/liuchunfa/2026qjx/2026test/train_s3a_multisource_dinov2.py#L3375).

Assessment:

- This is reasonably engineered. The mitigation path is not logically broken.
- But `dino_starved` and collapse alarm are not fully semantically identical:
  - `dino_starved` measures "DINO above-floor share is effectively gone."
  - `collapse_alarm` measures "self dominates and fused is still better than self-only."
- So starvation can be observable before collapse mitigation chooses to act.

This is not a correctness bug. It is a monitoring-versus-control semantic gap.

#### 4. P2: the code already records almost all ingredients needed for a minimal synergy metric

Evidence:

- The metrics row already logs `alpha_dino_above_floor`, `dino_starved`, `dual_source_alive`, `loss_fused_probe`, `loss_dino_only`, and `loss_self_only` in [`train_s3a_multisource_dinov2.py:2757`](/home/liuchunfa/2026qjx/2026test/train_s3a_multisource_dinov2.py#L2757) to [`train_s3a_multisource_dinov2.py:2773`](/home/liuchunfa/2026qjx/2026test/train_s3a_multisource_dinov2.py#L2757).

Implication:

- You do not need a new controller to judge collaboration quality.
- A minimal extra metric can be computed directly from already available probe losses.

### Answers

#### A. Will exact `a_dino=0` still occur?

- Under the safe dual-source contract: effectively no.
- Under unsafe override or upstream DINO-path failure: yes, it can return.

The important nuance is that "not zero" is not the same as "healthy collaboration."

#### B. What pseudo-collaboration boundaries still exist?

Main remaining boundaries:

1. Floor-supported pseudo-dual-source:
   - `raw_alpha_dino` collapses toward zero.
   - `alpha_dino` stays near the enforced floor.
   - Training logs no exact starvation, but DINO is not genuinely selected.

2. Above-floor but not truly synergistic:
   - `alpha_dino_above_floor > 0`.
   - `dual_source_alive=1`.
   - Yet `loss_fused_probe >= min(loss_dino_only, loss_self_only)`.

3. Self-dominant regimes that do not cross collapse trigger:
   - DINO still contributes something above floor.
   - `dino_starved` may stay 0.
   - Auto-mitigation may not trigger because the fused-versus-self comparison is not bad enough.

#### C. Should a minimal synergy metric be added?

- Yes.
- Recommended form:

`dual_synergy_margin = min(loss_dino_only, loss_self_only) - loss_fused_probe`

Interpretation:

- `> 0`: fused probe beats both single-source baselines, which is the cleanest operational definition of useful collaboration.
- `<= 0`: the two-source path is not outperforming the better single-source probe, even if alpha bookkeeping says both are alive.

This is the smallest high-value addition because:

1. It uses losses you already compute.
2. It closes the main interpretability gap.
3. It does not add a new controller, state machine, or resume-surface.

### Minimal recommendation

1. Add `dual_synergy_margin` to `metrics.jsonl` and trainer log lines.
2. Keep it observational first. Do not wire it into gate or mitigation yet.
3. Continue using:
   - `dino_starved` for hard anti-starvation monitoring
   - `dual_source_alive` for above-floor coexistence
   - `dual_synergy_margin` for actual collaboration quality

### Bottom line

For the user question "is DINO starvation now engineering-controllable?", the answer is:

- Yes for exact-zero starvation.
- Not fully for the broader collaboration objective.

For the user question "do we need another big patch?", the answer is:

- No.
- The minimal next step is one observational synergy metric, not another controller.
