# Research Review: S3A Post-Patch Audit (2026-04-15)

## Scope
- Target: latest post-patch code on branch `s3a`
- Audited files:
  - `/home/liuchunfa/2026qjx/2026test/train_s3a_multisource_dinov2.py`
  - `/home/liuchunfa/2026qjx/2026test/run_s3a_multisource_dinov2.sh`
  - `/home/liuchunfa/2026qjx/2026test/scripts/run_e0_e7_single_seed.sh`
  - `/home/liuchunfa/2026qjx/2026test/scripts/launch_e0_e7_single_seed_tmux.sh`
  - `/home/liuchunfa/2026qjx/2026test/docx/idea_s3a_with_audit_feedback.md`
  - `/home/liuchunfa/2026qjx/2026test/docx/implementation_contract_20260414.md`

## Reviewer Setup
- Reviewer M (mechanism/math): `019d8cbb-d5a9-7eb2-bc20-cd3af70a12ff`
- Reviewer E (engineering/repro): `019d8cbb-d5eb-7b63-b843-189a6d9b6e9f`
- Reviewer A (AC/paper): `019d8cbb-d655-71d2-8fe3-c494eab82f20`
- Model/reasoning: `gpt-5.4` / `xhigh`

## This Round’s Code Changes
1. Strict resume arg checking with explicit unsafe bypass:
- `--allow-legacy-resume-args`
2. Lineage artifact output:
- `resolved_args.json` (git revision, argv, resume source)
3. Structured failure telemetry:
- non-finite loss/grad event rows into `metrics.jsonl`
4. Collapse stale-state reset for zero-probe windows
5. Launcher safety hardening:
- block managed trailing arg overrides in `run_s3a_multisource_dinov2.sh`
6. E0-E7 experiment naming improvement:
- contract suffix in S3A run name/path

## Round-1 Findings
### Reviewer M (mechanism/math)
- Still sees method-level `P0`: training policy and utility/audit policy are not mathematically unified.
- Core issue: gate/alarm utility still depends on probe policy (`uniform` by default), not the exact training policy.
- Reliability score: `6/10`.

### Reviewer E (engineering/repro)
- Engineering improved to roughly `6.0~6.5/10`.
- Remaining blocker for audit-grade reproducibility:
  - dirty-tree provenance not sealed by artifact identity.
- Remaining gaps:
  - `run_s3a` header/argv can still drift for some trailing args (e.g., `--allow-legacy-resume-args` not blocked in this review run).
  - identity not fully injective for all contract-critical knobs.

### Reviewer A (AC/paper)
- Confirms best current narrative is:
  - `guarded dual-source auxiliary alignment`
  - not `principled source reliability routing`.
- Mock review stance remains weak reject without fresh causal evidence.
- Reliability score on audit conclusion: `5/10`.

## Converged Assessment
- Engineering contract: significantly better than previous unsafe baseline.
- Method claim strength: still not enough for strong routing/reliability claims.
- Paper posture: continue mainline, but with narrowed claim boundary and causal evidence-first plan.

## Post-Patch Reliability Score
- Consolidated reliability score: **`~5.8/10`**

Interpretation:
- Better than prior deep audit (`~4.5/10`), but not yet “submission evidence-grade”.
- Major remaining risk moved from pure crash-risk to mechanism interpretation + provenance closure.

## Updated Claim Boundary
### Allowed
1. Training-only guarded dual-source auxiliary alignment (DINO anchor + optional EMA self) is implemented and stabilized.
2. Current revision improves runtime safety and observability.
3. Inference path remains unchanged (no added runtime branch).

### Borderline
1. Utility gate improves final quality (not just stability).
2. Guarded dual-source consistently beats strongest single-source baseline.
3. Resume equivalence in practice across full training trajectories.

### Forbidden
1. “S3A has proven source reliability routing.”
2. “General multi-external-source bank `{E_k}` is validated.”
3. “Attention-map alignment is implemented” (current code uses affinity loss).

## Minimal Next Actions
1. Mechanism unification patch:
- unify train/gate/alarm utility under one policy (e.g., leave-one-source-out marginal utility on route policy).
2. Provenance closure:
- add code-state fingerprint (dirty bit + tracked file hash digest) to resolved artifacts/checkpoints.
3. Fresh current-code causal pack:
- True vs Shuffle vs Corrupt DINO
- DINO-only / Self-only / Dual-guarded
- drop-one guard bundle

## Recommendation
- Continue mainline.
- Immediately narrow paper language to guarded dual-source contract.
- Do not claim reliability-routing mechanism until policy-consistent utility + causal experiments pass.
