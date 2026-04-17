# Research Review: S3A Design / Structure / Interface (2026-04-16)

## Scope
- Request: `$research-review` for code design, structure, interface contracts.
- Target file: `/home/liuchunfa/2026qjx/2026test/train_s3a_multisource_dinov2.py`
- Supporting scripts: `scripts/monitor_training_20m.sh`, `run_s3a_multisource_dinov2.sh`, `scripts/run_e0_e7_single_seed.sh`
- Reviewer agent: `gpt-5.4` (`xhigh`), id `019d9200-a738-7940-b495-93bbd9efed76`

## Project Context Sent to Reviewer
- Single-file S3A training stack (~3482 lines) with router, dual-source alpha policy, selective gate, probes, checkpoint contract, and monitoring.
- Goal: avoid DINO (source0) deactivation and keep dual-source collaboration alive.
- Recent fixes already landed: warmup gate handoff reset, floor-relative starvation logic, resume contract hardening, gate observability split (`mask_self` vs `gate_self_state`).

## Round 1 Summary
### Verdict
- `CONDITIONAL`
- Interpretation: usable engineering baseline with floor+watchdog safeguards, but not yet a mechanism-closed robust dual-source solution.

### Key Findings
- `P1` Layer-average watchdog can hide single-layer DINO starvation.
- `P1` Floor-only oscillation risk remains (temporary mitigation then reopen back to floor-only regime).
- `P1` Resume contract does not fully expose optimizer runtime truth (CLI hyperparams vs restored `opt_state`).
- `P2` Monitor parser drift (`mask_self` now logged, monitor still expects `gate_self` text pattern/fallback).
- `P2` Canonical launcher relies on parser defaults for core booleans (`s3a_use_ema_source`, `s3a_enable_selective_gate`).
- `P2` Architectural coupling remains high (policy/probe/gate/resume/metrics in one main chain).

## Round 2 Follow-up (Minimal-Change Plan)
Reviewer was asked to provide a low-risk patch package (no big refactor).

### Minimal mechanism fix package (<=3 items)
1. Layer-local starvation signal should directly affect self gate update.
- Inject per-layer `alpha_dino_above_floor` starvation into pre-`update_gate_state` utility path.

2. Add reopen hysteresis gate.
- In add-one probe for source1 reopen, require DINO not to remain floor-only after adding self.

3. Upgrade watchdog from mean-only to include any-layer starvation.
- Add `alpha_dino_min_layer_above_floor` + `any_layer_dino_starved` and wire to alarms/mitigation.

### Why this package
- Addresses biggest residual mechanism risks without large restructuring.
- Expected code delta is small (roughly tens of lines each item).

## Consensus
- Current baseline is not blocked by hard correctness defects (`P0` none).
- Remaining risk is mechanism-quality (starvation detectability and reopen stability), not parser fatality.
- Recommended path: apply minimal 3-item mechanism patch first, then run fresh 10k canary before any large refactor.

## Results-to-Claims Matrix (Engineering)
| Canary outcome (post-warmup window) | Allowed claim |
|---|---|
| `any_layer_dino_starved=0`, `dino_starved_alarm=0`, `collapse_alarm=0`, `dual_source_alive=1` stable to end | "Engineering dual-source anti-starvation baseline is stable." |
| Alarms occasionally recover but mitigation triggers >1 | "Partially stabilized; still oscillatory, not robust baseline." |
| Any-layer floor-only starvation persists or end-window alarms remain | "Dual-source mechanism unresolved; do not claim collaboration readiness." |

## Pass Criteria (10k canary)
- Evaluate on `step >= warmup + 1000` (for default warmup 5000 => evaluate from 6000).
- Hard pass checks:
1. No 3 consecutive windows with `alpha_dino_min_layer_above_floor <= 1e-3`.
2. Final 3 windows: `any_layer_dino_starved=0`, `dino_starved_alarm=0`, `collapse_alarm=0`, `dual_source_alive=1`.
3. `collapse_mitigation_trigger_count <= 1`.
- Recommended (non-blocking) checks:
1. `median(alpha_dino_above_floor) >= 0.01`.
2. `median(alpha_dino_min_layer_above_floor) >= 0.003`.
3. `dual_synergy_margin` near non-negative in last windows.

## Prioritized TODO (with estimated cost)
1. Add any-layer starvation metrics + alarm wiring (`~0.1 dev-day`, `~0 GPU-day`).
2. Add reopen hysteresis guard in add-one probe (`~0.1 dev-day`, `~0 GPU-day`).
3. Add layer-local starvation-to-gate hook (`~0.2 dev-day`, `~0 GPU-day`).
4. Run fresh 10k canary and evaluate pass template (`~0.3-0.5 GPU-day`, depends on hardware).
5. Only after pass: do staged module split with golden tests (`1-2 dev-days`, optional compute for regression smoke).

## Notes
- This review intentionally prioritizes engineering landing and minimal behavior change.
- Large file refactor is recommended, but not on critical path to immediate mechanism stabilization.
