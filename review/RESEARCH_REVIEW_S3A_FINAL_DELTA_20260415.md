# Research Review: S3A Final Delta Audit (2026-04-15)

## Scope
- Target code: `/home/liuchunfa/2026qjx/2026test/train_s3a_multisource_dinov2.py`
- Launch defaults checked:
  - `/home/liuchunfa/2026qjx/2026test/run_s3a_multisource_dinov2.sh`
  - `/home/liuchunfa/2026qjx/2026test/scripts/run_e0_e7_single_seed.sh`
- Goal: confirm whether latest fixes fully close prior audit risks (legacy compatibility + probe-floor consistency).

## Reviewer Setup (research-review skill)
- Mechanism reviewer (gpt-5.4 xhigh): `019d8d16-0697-72b0-9457-46118c132b6e`
- Engineering reviewer (gpt-5.4 xhigh): `019d8d16-06e7-7041-8f51-519d323bcdfc`
- AC/narrative reviewer (gpt-5.4 xhigh): `019d8d16-0799-7701-95cf-95306e2bce3f`

## Local Cross-Check Anchors
- Dual-track utility EMA + regime split gate updates: lines 664-830.
- Policy-consistent `policy_loo` probe path and reopen-floor logic: lines 1125-1191.
- Resume arg contract + backward-compatible missing-key set: lines 1507-1594.
- Legacy migration path: lines 1597-1646.
- Parser defaults:
  - `--s3a-protect-source0-min-alpha=0.0` (line 2993)
  - `--s3a-gate-reopen-probe-alpha-floor=0.0` (line 3059)
- Launcher defaults also keep both values at `0.0`.

## Converged Findings

### Closed (relative to previous round)
1. Fresh-run mixed-estimand bug is closed:
- active/off regimes use separate EMA states.
- gate flip invalidates entering-regime EMA for fresh seeding.
2. Policy mismatch criticism is largely closed for default `policy_loo` path:
- utility probe uses train-consistent alpha builder.
3. Missing-arg hard fail for newly added keys is softened:
- legacy checkpoints missing two new keys no longer fail by default.
4. Metrics semantic version marker exists (`metrics_schema_version` + utility semantics tag).

### Still Open (material)
1. `P1` legacy resume sterilization still incomplete:
- migration discards old `source_utility_ema` but does not sanitize legacy `source_gate_mask` if present.
- stale gate state can affect pre-probe behavior after resume.
2. `P1` sticky-off remains possible under shipped defaults:
- reopen probe floor default is `0.0` in parser and launchers.
- gated-off source can stay near-zero influence if router mass collapses.
3. `P1` resume allowlist is fail-open for non-default overrides:
- missing-key allowlist skips checks unconditionally for two keys.
- legacy checkpoint can resume even when current run sets those keys non-default.
4. `P1` floor composition can violate requested reopen floor:
- sequential min-alpha applications do not guarantee simultaneous floor constraints.
- cross-arg feasibility is not validated.
5. `P2` backward compatibility is not fully default-smooth for older `uniform` checkpoints:
- `s3a_utility_probe_mode` remains strict mismatch key.

## Final Verdict
- Engineering/mechanism verdict: **real improvement, but still contract-level hardening**.
- Paper posture verdict: **still not mechanism-claim safe; weak-reject risk remains without fresh causal evidence**.
- Recommended narrative remains: **guarded dual-source auxiliary alignment** (not principled/general reliability routing).

## Updated Claims Matrix

### Allowed
1. Training-time guarded dual-source alignment with policy-aware utility gating is implemented.
2. Controller correctness and observability improved versus earlier unsafe versions.
3. Inference path unchanged.

### Borderline
1. Dual-guarded consistently outperforms strongest DINO-only baseline.
2. Gate/reopen policy (not just warmup/floor heuristics) drives gains.

### Forbidden (current evidence/code)
1. Proven/general source reliability routing.
2. Guaranteed anti-sticky-off recovery by default settings.
3. Guaranteed source0 effective contribution after early stage.

## Minimal Patch Package (highest closure per LOC)
1. Legacy migration sterilization:
- when discarding `source_utility_ema`, also reset `source_gate_mask` and mitigation window state to expected defaults.
- optionally enforce source0 pass-through in `get_source_mask()` after selective gate multiply.
2. Tighten missing-key allowlist semantics:
- only allow missing new keys when current value equals intended legacy backfill default.
3. Add cross-arg validation:
- reject impossible floor combinations (e.g., source0 floor + reopen floor > 1 in 2-source case).
- reject nonzero reopen floor outside `policy_loo` mode.
4. Clarify resume mismatch guidance for legacy `uniform` checkpoints.

## Minimal Evidence Package (7-10 days)
1. `True / Shuffle / Corrupt DINO` causal mini-pack at 10k, 1 seed each.
2. `DINO-only / Self-only / Dual-guarded` at 10k, 1 seed each.
3. Promote best `DINO-only` and `Dual-guarded` to 40k, 3 seeds each.
4. Only if logs show starvation/no-reopen: short default-vs-floors-on diagnostic.

## Status
- Decision: **REVISE-and-VERIFY**
- This round closed multiple objections, but not the last contract risks.
