# Research Review: S3A Dual-EMA Gate Patch Audit (2026-04-15)

## 1) Scope
- Branch: `s3a`
- Audited patch goal:
  - fix mixed-estimand controller memory by splitting gate utility EMA into active/off regimes,
  - add gate-flip reset to avoid stale controller carry-over.
- Primary code:
  - `train_s3a_multisource_dinov2.py`
  - `docx/implementation_contract_20260414.md`

## 2) What Changed (audited)
1. Gate utility memory split:
- `source_utility_active_ema`
- `source_utility_inactive_ema`

2. Gate update semantics separated by regime:
- gate ON: update active EMA; off decision from active EMA
- gate OFF: update inactive EMA; reopen decision from inactive EMA

3. Gate flip reset added:
- on ON->OFF or OFF->ON, reset EMA/counters to prevent regime contamination.

4. Utility estimator remains policy-consistent:
- `policy_loo` under same alpha policy (`_build_alpha`) as train path.

5. Logging/metrics extended:
- `utility_self_active_ema`
- `utility_self_inactive_ema`
- legacy `utility_self_ema` retained as alias.

6. Checkpoint schema bumped:
- `format_version: 4`
- legacy migration path added for old `source_utility_ema`.

## 3) Reviewer Setup (multi-agent)
- Reviewer M (mechanism/math): `019d8ce7-b3e3-7a52-b213-05c15b3d1758`
- Reviewer E (engineering/repro): `019d8ce7-b43c-7bb1-8c72-622e137061bf`
- Reviewer A (AC/paper): `019d8ce7-b493-7653-ad63-d6d5757f64d8`
- Model/reasoning: `gpt-5.4` / `xhigh`

## 4) Round Findings

### 4.1 Mechanism/Math (Reviewer M)
- Positive:
  - Fresh-start semantics repair is valid: active/off estimands are no longer mixed in one EMA.
- Core risks:
  1. Legacy resume contamination still possible: old mixed EMA is copied into both new EMAs.
  2. Sticky-off risk: when source is off, add-one utility can collapse if router mass for that source is near zero.
  3. `protect_source0` protects availability, not effective usage (alpha may still go near zero).
  4. Full zero-reset on each flip may induce slow reopen/chatter due zero-start bias.
- Score:
  - `7.1/10` (fresh-start scope)
  - `6.3/10` (including legacy-resume semantics)

### 4.2 Engineering/Repro (Reviewer E)
- Positive:
  - Interface and telemetry are materially clearer.
- Core risks:
  1. Legacy migration currently semantically unsafe for old checkpoints (same point as M).
  2. Resume contract is stricter due probe-mode default changes (safe-by-fail but operationally brittle).
  3. Metric semantic drift: `utility_self_ema` alias meaning changed silently.
  4. Flip-reset + default thresholds/momentum may delay reopen and appear no-op.
  5. Probe overhead increases (extra counterfactual + multiple all-reduce).
- Score:
  - `5/10`

### 4.3 AC/Paper (Reviewer A)
- Positive:
  - This patch removes a major logical weakness and improves narrative defensibility.
- Core limits:
  1. Still not enough to claim principled reliability routing.
  2. Asymmetric setup (DINO protected anchor) remains guarded dual-source, not general routing.
  3. Evidence chain still missing (multi-seed and direct old-vs-new controller ablation).
- Acceptance-lift estimate:
  - `+0.5 ~ +0.8` on 10-point reviewer scale.
- Position:
  - from “mechanistically flawed” -> “mechanistically coherent but heuristic / borderline”.

## 5) Converged Assessment
- Consensus:
  - This patch is a real mechanism improvement and fixes the main probe-memory mismatch for fresh starts.
  - Current safe paper posture remains:
    - `guarded dual-source auxiliary alignment`
    - not `principled reliability routing`.

- Remaining blocker (highest priority):
  - legacy resume migration should not copy old mixed EMA into both new tracks.

## 6) Results-to-Claims Matrix

| Outcome | Allowed claim | Not allowed |
|---|---|---|
| Strong gains (3 seeds + direct old-vs-new wins) | Dual-EMA gate improves stability/performance in guarded dual-source setup | General source reliability routing is solved |
| Mixed gains | Semantics fix improves controller interpretability; quality gain depends on config | Mechanism is primary cause of all gains |
| No gains | Semantics hardening improves correctness/observability only | Mechanism is effective for quality |

## 7) Prioritized TODO (with rough cost)

### P0 (must fix before strong promotion runs)
1. Legacy migration fix:
- do not map old mixed `source_utility_ema` into both new EMAs.
- migrate by reset/invalidate both, or map only active side by historical gate and zero inactive side.
- Cost: low (code 0.5 day) + sanity run (0.5 day).

2. Add explicit migration regression tests:
- old-format resume with gate-off/mitigation edge cases.
- Cost: low (0.5 day).

### P1
1. Reduce flip-reset bias:
- prefer regime-entry invalidate + first-sample seeding over hard zero-start for both EMAs.
- Cost: medium (0.5-1 day + quick sweep).

2. Add anti-sticky-off exploration safeguard:
- periodic trial-open or minimum probe alpha on off-source.
- Cost: medium (1-2 days including sweep).

3. Metric schema annotation:
- add `metrics_schema_version` and explicit `utility_ema_semantics` marker.
- Cost: low (0.5 day).

### P2
1. Threshold retune under dual-EMA policy_loo scale.
2. Probe-overhead benchmark (`probe_every=1/5/10`).
3. Causal package:
- DINO true/shuffle/corrupt + DINO-only/Self-only/Dual-old/Dual-new.
- Cost: medium-high (multi-GPU days).

## 8) Final Recommendation
- Status: `REVISE-and-VERIFY`
- Next best action:
  1. fix migration semantics,
  2. run minimal old-vs-new controller ablation + 3 seeds,
  3. keep narrative strictly under guarded dual-source scope.
