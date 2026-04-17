# Research Review: S3A Deep Audit (2026-04-14)

## 1) Scope
- Branch: `s3a`
- Audited implementation:
  - `/home/liuchunfa/2026qjx/2026test/train_s3a_multisource_dinov2.py`
  - `/home/liuchunfa/2026qjx/2026test/scripts/run_e0_e7_single_seed.sh`
  - `/home/liuchunfa/2026qjx/2026test/run_s3a_multisource_dinov2.sh`
  - `/home/liuchunfa/2026qjx/2026test/scripts/launch_e0_e7_single_seed_tmux.sh`
- Context docs:
  - `/home/liuchunfa/2026qjx/2026test/docx/idea_s3a_with_audit_feedback.md`
  - `/home/liuchunfa/2026qjx/2026test/docx/implementation_contract_20260414.md`
  - `/home/liuchunfa/2026qjx/2026test/RESEARCH_REVIEW_MODIFIED_S3A_20260414.md`
  - `/home/liuchunfa/2026qjx/2026test/MODIFIED_S3A_RESEARCH_REVIEW_20260414.md`

## 2) Reviewer Setup
- Reviewer A (mechanism/math): `019d8ca1-44d3-7d00-bf93-7a89cbfb351b`
- Reviewer B (engineering/reproducibility): `019d8ca1-4526-7452-b83e-9187502f4584`
- Reviewer C (AC/paper quality): `019d8ca1-45a7-7a00-80a2-61d9273b09b7`
- Model/reasoning: `gpt-5.4` / `xhigh`

## 3) Round 1 Findings
### 3.1 Reviewer A (mechanism/math)
- Core verdict: engineering guardrails improved, but method-level P0 remains.
- Main critique:
  - Training objective uses effective `alpha_train` (includes gate/floor/mask), but gate/alarm utility relies on probe policy; estimator-controller is not mathematically unified.
  - Router is not source-evidence-aware; "reliability router" wording is too strong.
  - Design doc formulas and code truth are not aligned.
- Reliability score on existing audit conclusion: `3/10`.

### 3.2 Reviewer B (engineering/reproducibility)
- Core verdict: code hardening is real, but evidence chain is weak and experiment identity is not fully audit-grade.
- Main critique:
  - Current online long run evidence is old contract (unsafe warmup) and cannot validate latest fixes.
  - Resume contract is still fail-open for missing keys in legacy checkpoints.
  - Launcher can show header config but run overridden trailing args; experiment identity can drift.
- Reliability score on existing audit conclusion: `4/10`.

### 3.3 Reviewer C (AC-level)
- Core verdict: current implementation supports a narrower story (`guarded dual-source auxiliary alignment`), not full "source reliability routing" story.
- Main critique:
  - `source bank {E_k}` and attention-map alignment claims are over-claimed vs code.
  - Selective stop narrative in doc (FID/Recall/grad-conflict) does not match implementation (utility EMA gate).
  - Need claim contraction + causal evidence package before top-venue posture.
- Reliability score on existing audit conclusion: `6/10`.

## 4) Round 2 Convergence
### 4.1 Targeted Follow-up Outputs
- Reviewer A gave a minimal unification patch concept:
  - split `alpha_route` vs `alpha_train`;
  - compute gate/alarm utility via leave-one-source-out marginal utility under the same route policy;
  - deprecate probe mode mismatch in mechanism interpretation.
- Reviewer B separated:
  - Code defects (must fix) vs Evidence gaps (must run);
  - proposed 3 minimum release gates to raise audit trust >= 7/10.
- Reviewer C provided:
  - narrowed title direction;
  - safe claim boundaries;
  - reviewer Q&A strategy.

### 4.2 Converged Position
- Engineering contract quality: improved significantly vs earlier unsafe state.
- Method validity: still `REVISE` for top-venue claim strength.
- Current correct posture:
  - "guarded dual-source training contract with anti-collapse safeguards".
  - Not yet: "principled source reliability routing".

## 5) Reliability of Current Audit Conclusions
- Reviewer scores: `3/10`, `4/10`, `6/10`
- Consolidated score: **`4.5/10`**

Interpretation:
- Directional conclusions are mostly right.
- But conclusions are not yet robust enough to be used as final evidence in paper claims.
- Main reason: static code audit advanced faster than fresh causal evidence.

## 6) Mechanism/Math Risk Register
### P0 (method-level)
1. Utility-policy mismatch risk:
- gate/alarm utility and training policy are not from a single estimator/controller definition.
- Impact: claim-level mechanistic interpretation is fragile.

### P1
1. Router claim overreach:
- router inputs are context-only (`student tokens + timestep/phase/layer`) rather than source-specific evidence.
2. Asymmetric gate semantics:
- practical behavior is one-sided self control with protected DINO anchor, not symmetric source reliability control.
3. Doc-code mismatch:
- formula/narrative drift (`attn map`, `source bank`, `selective stop criterion`) vs actual code.

### P2
1. Recovery observability blind spots during mitigation windows.
2. Audit document drift (old defaults/comments inconsistent with latest code).

## 7) Claim Boundary (for writing)
### Allowed
1. Training-only dual-source auxiliary alignment branch with multi-layer taps and utility-driven self gating exists and is implemented.
2. Current revision adds explicit anti-collapse safeguards and observability.
3. Inference path remains backbone-only (no extra runtime branch at sampling).

### Borderline (needs targeted evidence)
1. Guarded dual-source outperforms strongest single-source baseline under matched budget.
2. Utility gate contributes quality (not only stability).
3. Holistic loss decomposition contributes beyond feature-only.

### Forbidden (currently)
1. "S3A learns source reliability routing" as a proven mechanism.
2. "General multi-external-source bank" claim.
3. "Attention-map alignment" claim if only affinity loss is implemented.
4. Any causal claim based only on old unsafe runs.

## 8) Results-to-Claims Matrix
| Outcome | Allowed claim | Forbidden claim |
|---|---|---|
| Dual-guarded > DINO-only > NoAlign, Placebo-DINO drops, 3 seeds stable | DINO semantic source is useful under guarded contract | Proven reliability routing |
| Dual-guarded ≈ DINO-only > NoAlign | Gain mainly from DINO alignment; self/gate incremental gain unclear | Multi-source routing is core gain driver |
| Dual-guarded only survives via frequent mitigation | Guardrails improve stability | Router has learned robust source switching |
| Placebo-DINO ≈ True-DINO | Effect likely regularization/branch effect | External semantic teacher is core causal factor |
| High variance across seeds, no clear margin | REVISE-track exploratory result | Submission-grade main claim |

## 9) Prioritized TODO (acceptance lift / GPU week)
Assumption from recent log (`~0.82 steps/s` on 4 GPUs):
- 10k steps ≈ 3.4 hours wall-clock ≈ 13.6 GPU-hours.
- 40k steps ≈ 13.6 hours wall-clock ≈ 54.2 GPU-hours.

### T1 (Highest ROI)
- Causal source-quality intervention:
  - True-DINO vs Shuffled-DINO vs Corrupt-DINO.
  - 10k x 1 seed each.
- Cost: ~41 GPU-hours.
- Purpose: distinguish reliability routing vs schedule prior.

### T2
- Bundle decomposition (drop-one):
  - remove warmup / floor / self-freeze / utility-gate one at a time.
  - 10k x 1 seed x 4 runs.
- Cost: ~54 GPU-hours.
- Purpose: identify true gain source.

### T3
- Source necessity matrix:
  - DINO-only vs Self-only vs Dual-guarded.
  - 10k x 1 seed x 3 runs.
- Cost: ~41 GPU-hours.
- Purpose: prove dual-source necessity.

### T4
- Promotion runs for paper signal:
  - best 1-2 configs to 40k, then 3 seeds for top-1 candidate.
- Cost: 54~325 GPU-hours depending selection depth.
- Purpose: convert mechanism trend into robust quality claim.

## 10) Minimal Regression Gate (engineering)
Before using any new results in paper:
1. Launch identity gate:
- resolved argv/config digest must be saved and match executed command.
2. Fresh current-code canary:
- run with current code; verify new metrics schema fields exist.
3. Resume equivalence gate:
- uninterrupted vs resume split consistency + fail-closed negative test.

## 11) Recommended Paper Reframe (current stage)
- Suggested direction:
  - `Guarded Dual-Source Auxiliary Alignment for Stable Representation Guidance in DiT/SiT`.
- One-line safe claim:
  - "A training-only guarded dual-source auxiliary alignment contract improves stability and can improve quality under matched baselines, without inference overhead."

## 12) Final Decision
- **Modification reasonableness**: reasonable as robust engineering hardening.
- **Audit conclusion reliability**: not yet high; needs fresh causal and reproducibility evidence.
- **Current status**: `REVISE-and-VERIFY`, not submission-ready.
