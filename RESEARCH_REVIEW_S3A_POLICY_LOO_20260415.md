# Research Review: S3A `policy_loo` Patch Audit (2026-04-15)

## 1) Scope
- Branch: `s3a`
- Baseline commit before this patch: `507b7e1`
- Audited patch (working tree):
  - `train_s3a_multisource_dinov2.py`
  - `run_s3a_multisource_dinov2.sh`
  - `scripts/run_e0_e7_single_seed.sh`
  - `docx/implementation_contract_20260414.md`
  - `docx/idea_s3a_with_audit_feedback.md`

## 2) What Changed
1. Utility estimator default switched to `policy_loo`:
- `--s3a-utility-probe-mode` choices now include `policy_loo`; default is `policy_loo`.
2. Utility computation in S3A alignment now uses policy-consistent counterfactuals:
- active source: `U_k = L(without k) - L(full)`
- inactive source: `U_k = L(current) - L(add k)`
- both under same `_build_alpha(mask)` policy as training.
3. Legacy `uniform/raw_alpha` probe estimators kept for backward/ablation.
4. Launcher defaults updated from `uniform` to `policy_loo`.
5. Launcher trailing-arg guard extended to include `--allow-legacy-resume-args`.
6. E0-E7 naming now includes utility estimator tag in contract suffix.
7. Design/contract docs synced to `policy_loo` semantics.

## 3) Reviewer Setup
- Reviewer M (mechanism/math): `019d8cce-6a45-7ca2-886b-984e18f18519`
- Reviewer E (engineering/repro): `019d8cce-6ab3-7163-b39c-05ce7ba33b91`
- Reviewer A (AC/paper): `019d8cce-6b1b-7142-a579-7789f69b8b12`
- Model/reasoning: `gpt-5.4` / `xhigh`

## 4) Round Findings

### 4.1 Mechanism/Math (Reviewer M)
- Positive:
  - Confirms old probe-policy mismatch is fixed at instantaneous estimator level.
- Main risk:
  - Remaining estimator-controller mismatch in controller memory:
    - same `source_utility_ema` mixes two estimands (on-state LOO and off-state add-one), then used for both off/on hysteresis decisions.
- Additional risks:
  - mitigation windows can preserve stale utility memory.
  - thresholds were not re-calibrated for new estimator scale.
  - possible sticky-off behavior due router-starvation while gated off.
- Reliability score: **7/10**.

### 4.2 Engineering/Repro (Reviewer E)
- Positive:
  - static checks pass (py_compile + bash -n), checkpoint schema not broken.
- Main risks:
  - implicit resume behavior changes because default mode changed (`uniform -> policy_loo`) while resume contract validates this key.
  - mixed-estimand EMA + unchanged thresholds = controller non-stationarity.
  - probe compute overhead rises under `policy_loo` (up to ~5 probe losses vs ~3 legacy).
  - metric semantics drift (`loss_fused_probe`, `utility_*`) without schema/version marker.
- Reliability score: **8/10**.

### 4.3 AC/Paper (Reviewer A)
- Positive:
  - patch improves contract-level claim hygiene and defensibility.
- Main limits:
  - still cannot claim “principled source reliability routing”.
  - no fresh post-patch causal evidence yet.
  - mechanism still asymmetric (DINO anchor protected).
- Mock acceptance trajectory: ~2.6/10 -> ~3.2/10.
- Reliability score: **5.7/10**.

## 5) Converged Assessment
- This patch is a **real mechanism hardening step**.
- It resolves a key previous criticism (probe-policy mismatch), but does **not** fully close mechanism risk due to mixed-estimand EMA controller state.
- Current best paper posture remains:
  - `guarded dual-source auxiliary alignment`
  - not `principled source reliability routing`.

Consolidated reliability score (mean): **6.9/10**.

## 6) Updated Claim Boundary

### Allowed
1. S3A uses policy-consistent utility estimation (`policy_loo`) under the same training alpha policy.
2. Launcher/docs now align with this utility contract.
3. Training-only guarded dual-source branch remains inference-free.

### Borderline
1. `policy_loo` is better than `uniform/raw_alpha` in quality.
2. Gate improves final quality (not only stability).
3. Dual-source robustly beats strongest single-source baseline.

### Forbidden (for now)
1. “S3A proves source reliability routing.”
2. “General multi-external-source bank routing is validated.”
3. “Attention-map alignment is implemented” (code is affinity-style token loss).

## 7) Priority Fixes Before Large Promotion Runs

### P0
1. Split utility memory by regime (active/off) or reset EMA on each gate flip.
2. Clear self utility EMA when entering/exiting forced mitigation windows.

### P1
1. Re-tune gate thresholds for `policy_loo` scale.
2. Add metric schema/version and log estimator mode per row to avoid cross-run silent mixing.
3. Add a resume note/guardrail for old `uniform` checkpoints.

### P2
1. Benchmark probe overhead (`uniform` vs `policy_loo`).
2. Decide whether `self_only_count` should include warmup/mitigation steps; align code/analysis semantics.

## 8) Minimal Regression + Evidence Pack
1. Resume contract test: old-uniform checkpoint resume under new defaults, then explicit mode override.
2. Gate reopen sanity: forced off -> reopen behavior under `policy_loo`.
3. Metric semantics sanity: warmup/mitigation windows and `utility_*` interpretation.
4. Probe performance benchmark: steps/s and memory cost.
5. Fresh causal mini-pack:
- True/Shuffle/Corrupt DINO
- DINO-only / Self-only / Dual-guarded
- `policy_loo` vs `uniform/raw_alpha`

## 9) Decision
- Status: **REVISE-and-VERIFY**.
- Recommendation: keep current direction, fix controller-memory semantics and metric contract first, then run focused causal package.
