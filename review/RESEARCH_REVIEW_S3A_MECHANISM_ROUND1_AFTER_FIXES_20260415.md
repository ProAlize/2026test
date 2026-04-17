# Research Review: S3A Mechanism Audit Round-1 After Fixes (2026-04-15)

## Scope
- Target code: `/home/liuchunfa/2026qjx/2026test/train_s3a_multisource_dinov2.py`
- Audit mode: mechanism-only review after latest fixes
- Questions:
  - A) Can the system still enter `a_dino=0` with long self monopoly?
  - B) Are `source0` floor and `dino_starved` / `dual_source_alive` self-consistent, and do they separate true cooperation from floor-only nonzero alpha?
  - C) Do selective gate, collapse mitigation, and the new contract conflict?

## Review Mode
- `research-review` local fallback
- This interface did not expose `spawn_agent` / `send_input`, so the review was executed locally with code-anchored cross-check instead of a fresh delegated reviewer.

## Code Anchors
- Source0 floor builder: lines 418-429
- Gate state and source0 protection: lines 723-875
- Alpha construction and `policy_loo` counterfactual probes: lines 1043-1218
- Resume sterilization for source0 lane: lines 1661-1714 and 1833-1843
- Launch contract row: lines 2260-2279
- Monitoring / starvation / mitigation metrics: lines 2560-2800
- Argument contract checks: lines 3273-3369

## Conclusions

### A. `a_dino=0` and long self monopoly
Verdict:
- Exact `a_dino=0` is closed under the safe dual-source contract.
- A weaker failure mode still exists: router-level self dominance with DINO held above zero only by the source0 floor.

Why:
- `source0_min_alpha_at_step()` enforces a persistent DINO floor when `--s3a-protect-source0-min-alpha > 0` and can add a larger transient warmup floor on top.
- `_build_alpha()` applies that floor in the actual training alpha path, not only in probes.
- `update_gate_state(..., protect_source0=True)` keeps source0 logically ungated.
- `get_source_mask()` forcibly restores source0 availability even if a stale gate mask was resumed.
- Resume path additionally sterilizes the source0 gate lane before load.

Mechanism consequence:
- In safe dual-source mode, source0 cannot be numerically driven to zero by the router or by stale gate state.
- But the system can still settle into `raw_alpha_dino ~= 0`, `alpha_dino ~= source0_floor`, `alpha_self ~= 1 - source0_floor`, `gate_self ~= 1`. That is not exact zero, but it is still not the intended "dual-source cooperation".

Boundary conditions:
- This closure depends on `--s3a-use-ema-source=1`, `--s3a-protect-source0-min-alpha > 0`, and no explicit unsafe override.
- If the user enables `--s3a-allow-unsafe-zero-source0-floor`, exact starvation can return.
- This protection only guarantees nonzero train-time weight, not meaningful DINO contribution above the floor.

### B. Source0 floor vs `dino_starved` / `dual_source_alive`
Verdict:
- The metrics are internally coherent for distinguishing "floor-only nonzero DINO" from "some non-floor DINO share".
- They do not, by themselves, certify true cooperation.

Why they are coherent:
- `source0_floor_active` is derived from the same step-dependent floor function used by `_build_alpha()`.
- `alpha_dino_above_floor = max(0, alpha_dino - source0_floor_active)` explicitly removes the guaranteed floor mass.
- `dino_starved=1` is triggered when self is high after warmup and DINO has no meaningful mass above the floor.
- `dual_source_alive=1` requires both self activity and DINO mass above the floor.

What they do separate:
- `alpha_dino > 0` only because of the contract floor
- `alpha_dino > floor` because the router/training policy still assigns real mass to DINO

What they do not separate:
- "non-floor dual activity" versus "true cooperative gain"
- A run can have `dual_source_alive=1` even if fused supervision is not better than the best single source

Boundary conditions:
- `source0_floor_active` is computed as a single step-level scalar at log time, while `alpha_dino` is a window average. During floor decay, this is a monitoring approximation, not an exact per-sample decomposition.
- `dual_source_alive` uses an alpha threshold, not a utility or loss-improvement test.

Minimal extra suggestion:
- Keep the current metrics, but add one scalar such as
  - `dual_synergy_margin = min(loss_dino_only, loss_self_only) - loss_fused_probe`
- Then interpret:
  - `dual_source_alive`: non-floor dual activity exists
  - `dual_synergy_margin > 0`: fused target is actually better than both single-source probes

### C. Selective gate + collapse mitigation vs new contract
Verdict:
- No direct conflict is visible in the current implementation.
- The three pieces compose cleanly, with one important boundary condition around the temporary DINO floor.

Why there is no direct conflict:
- Selective gate is contract-bound to `policy_loo`, which matches the train-time alpha builder used by probes.
- Reopen probes can add a floor to the gated-off source, and `validate_args()` rejects impossible floor sums.
- Source0 is excluded from true gate-off behavior via `protect_source0=True`.
- Collapse mitigation acts by forcing the self lane off for a bounded number of windows, resetting its gate-side runtime state, and then handing recovery back to the normal selective-gate reopen logic.

Important boundary condition:
- Collapse alarm and auto-mitigation use `avg_alpha_dino <= collapse_alpha_threshold`.
- They do not use `alpha_dino_above_floor`.
- Therefore, when the transient DINO floor is larger than `collapse_alpha_threshold`, the auto-mitigation path is temporarily unreachable even if the router has already collapsed to pure floor-supported DINO.

Interpretation:
- This is not a logical contradiction.
- It means the current mitigation rule is a late safeguard, while `dino_starved` is the earlier detector.

Minimal extra suggestion:
- Choose one of these and keep the rest unchanged:
  - Trigger collapse mitigation on `alpha_dino_above_floor` instead of raw `alpha_dino`
  - Or explicitly validate/document that `collapse_alpha_threshold` is intended to be compared against the maximum active source0 floor

## Final Assessment
- Mechanism status: improved and materially safer
- Closed: exact `a_dino=0` from router collapse or stale source0 gate state under the safe contract
- Still open in mechanism terms: floor-supported pseudo-collaboration can persist without genuine dual-source gain
- Minimal next tightening, if desired: keep the controller as is and add one synergy metric or align collapse mitigation with `alpha_dino_above_floor`
