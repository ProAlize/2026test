## Research Review: Final Delta Audit (Mechanism)

Date: 2026-04-15
Mode: local fallback for `$research-review` (no `spawn_agent` / `send_input` interface available in this session)
Scope: `train_s3a_multisource_dinov2.py`
Question: verify whether the new starvation-to-mitigation coupling is coherent and non-contradictory, focusing on `dino_starved_windows`, `dino_starved_alarm`, and the mitigation trigger condition.

### Findings

1. No P1 contradiction in the new coupling. `dino_starved_windows` increments only on windows where `dino_starved=1`, resets immediately otherwise, and `dino_starved_alarm` is a pure threshold on that counter. Auto-mitigation then triggers only when either `collapse_alarm` or `dino_starved_alarm` is active and both `mitigation_active_windows<=0` and `collapse_mitigation_cooldown_windows<=0` hold. This state machine is internally coherent and the trigger is reachable. Relevant code: `train_s3a_multisource_dinov2.py:2615`, `train_s3a_multisource_dinov2.py:2657`, `train_s3a_multisource_dinov2.py:2666`, `train_s3a_multisource_dinov2.py:2675`.

2. No impossible steady state where mitigation and starvation alarm keep reinforcing each other in the same window sequence. Once mitigation is active, the self source is forcibly masked off in both `get_source_mask()` and `update_gate_state()`. That pushes `avg_alpha_self` down, so `dino_starved` becomes false and `dino_starved_windows` resets on the next logging window instead of continuing to accumulate under mitigation. Relevant code: `train_s3a_multisource_dinov2.py:768`, `train_s3a_multisource_dinov2.py:861`, `train_s3a_multisource_dinov2.py:2621`, `train_s3a_multisource_dinov2.py:2657`.

3. There is still a bounded oscillation mode, but not a pathological zero-gap loop. If the router falls back into floor-only DINO usage after mitigation expires, the system can retrigger another mitigation cycle after `s3a_collapse_windows` windows, and only after cooldown reaches zero. This is a periodic rescue pattern, not an immediate self-exciting loop, because active mitigation and cooldown both hard-block retriggering. Relevant code: `train_s3a_multisource_dinov2.py:2680`, `train_s3a_multisource_dinov2.py:2690`, `train_s3a_multisource_dinov2.py:2828`.

4. The starvation path is intentionally stronger than the old collapse path. `dino_starved_alarm` depends only on post-warmup self dominance plus `alpha_dino_above_floor<=1e-3`, while `collapse_alarm` still also requires positive DINO utility and fused-vs-self evidence. Because mitigation triggers on `(collapse_alarm > 0 or dino_starved_alarm > 0)`, starvation-only mitigation is now possible even when the old collapse predicate is false. This is coherent with the engineering goal of preventing DINO starvation, but it does mean mitigation no longer implies a demonstrated fused-loss advantage. Relevant code: `train_s3a_multisource_dinov2.py:2616`, `train_s3a_multisource_dinov2.py:2643`, `train_s3a_multisource_dinov2.py:2675`.

### Verdict

- The starvation-to-mitigation coupling is coherent and non-contradictory.
- I do not see an impossible state or a tight pathological retrigger loop in the current implementation.
- The remaining limitation is semantic, not logical: starvation-triggered mitigation guarantees "do not let DINO collapse to floor-only forever", but it still does not prove true dual-source synergy.
