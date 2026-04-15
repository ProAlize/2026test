# Proof Package

## Claim
For the current S3A implementation in `/home/liuchunfa/2026qjx/2026test/train_s3a_multisource_dinov2.py`, under dual-source mode (`num_sources=2`) and validated arguments (`validate_args` passes), define:

- source0 as DINO, source1 as EMA-self,
- training/probe fusion weights by `alpha = _build_alpha(mask_vec, extra_min_alpha_by_source)`,
- persistent DINO floor $m := \texttt{s3a\_protect\_source0\_min\_alpha}$,
- optional reopen-probe source1 floor $r := \texttt{s3a\_gate\_reopen\_probe\_alpha\_floor}$.

Then the following hold:

1. For every call to `_build_alpha`, $alpha$ is a valid probability vector:
$$
\forall i,\ \alpha_i \ge 0,\qquad \sum_i \alpha_i = 1.
$$

2. If `source_ready[0] > 0`, then training/probe alpha builder enforces:
$$
\alpha_0 \ge d,\quad d := \max\!\big(m,\ d_{\text{sched}}\big),
$$
where $d_{\text{sched}}$ is the time-decayed floor from `s3a_dino_alpha_floor` when active.
In particular, $\alpha_0 \ge m$ whenever source0 is ready.

3. For add-one reopen probe (`extra_min_alpha_by_source={1: r}`), if $d+r \le 1$, then both lower bounds hold simultaneously:
$$
\alpha_0 \ge d,\qquad \alpha_1 \ge r.
$$

4. Collapse-alarm alpha condition is boundary-reachable at floor equality: because implementation uses `<=` and validator enforces
$$
\texttt{s3a\_collapse\_alpha\_threshold} \ge m,
$$
the alarm is not structurally blocked by the source0 floor at equality.

## Status
PROVABLE AS STATED

## Assumptions
- `validate_args(args)` has passed.
- Dual-source branch is active (`s3a_use_ema_source=True`, hence `num_sources=2`).
- Source0 is available in the considered step/layer (`source_ready[0] > 0`).
- For part (3), either `r=0` or validator feasibility check ensures $d+r\le 1$ in the relevant regime.
- We reason over the implemented code paths:
  - `_apply_joint_min_alpha` and `_build_alpha` in `compute_s3a_alignment_loss`.
  - `validate_args` consistency checks.
  - collapse predicates in per-probe and window-level logic.

## Notation
- Let $\tilde{\alpha}\in\mathbb{R}_{\ge0}^2$ denote normalized masked router output before floor projection.
- Let floor vector $f\in\mathbb{R}_{\ge0}^2$ with components set by source-wise minimum-alpha constraints.
- Let $F := \sum_i f_i$.
- Let $\Delta^2 := \{\alpha\in\mathbb{R}_{\ge0}^2:\sum_i\alpha_i=1\}$.

## Proof Strategy
Direct proof from implementation semantics:
1. Read `_build_alpha` as "masked-softmax normalization + floor projection".
2. Analyze `_apply_joint_min_alpha` branch by branch.
3. Specialize to source0/source1 floors used in training and add-one reopen probes.
4. Use validator inequalities plus inclusive predicate (`<=`) to prove collapse condition reachability.

## Dependency Map
1. Main claim (1) depends on normalization and final renormalization in `_apply_joint_min_alpha`.
2. Main claim (2) depends on `_compute_dino_floor` and floor insertion `min_alpha_by_source[0]=d`.
3. Main claim (3) depends on joint floor map merge in `_build_alpha` and feasibility assumption $d+r\le1$.
4. Main claim (4) depends on:
   - per-probe/window predicates using `alpha_dino <= threshold`,
   - validator check `collapse_alpha_threshold >= s3a_protect_source0_min_alpha`.

## Proof
Step 1. `_build_alpha` first constructs
$$
\tilde{\alpha} = \frac{\texttt{raw\_alpha}\odot \texttt{mask}}{\sum_j \texttt{raw\_alpha}_j\texttt{mask}_j},
$$
with denominator clamped below by a positive constant in code. Therefore $\tilde{\alpha}_i\ge0$ and $\sum_i\tilde{\alpha}_i=1$.

Step 2. `_apply_joint_min_alpha` receives $\tilde{\alpha}$ and floor vector $f$.
If $F\le0$, function returns $\tilde{\alpha}$, so simplex validity is preserved.

Step 3. If $0<F<1$, code computes residual
$$
r_i := \max(\tilde{\alpha}_i-f_i,0),
$$
normalizes residual direction (or falls back to normalized $\tilde{\alpha}$ when residual is identically zero), then outputs
$$
\alpha_i = f_i + (1-F)\,\hat{r}_i,
$$
followed by normalization by its own sum (which is already positive). Since $\hat{r}_i\ge0$ and $\sum_i\hat{r}_i=1$, we have:
$$
\alpha_i \ge f_i,\qquad \sum_i\alpha_i=1,\qquad \alpha_i\ge0.
$$
So $\alpha\in\Delta^2$ and all requested floors are satisfied.

Step 4. If $F\ge1$, code returns normalized floor vector. This still yields $\alpha\in\Delta^2$. In this branch per-source floor guarantees may be weakened by normalization, so strict floor-preservation requires $F<1$. The active S3A dual-source contract avoids problematic $F>1$ via validator feasibility checks for configured floors.

Step 5. For source0, `_compute_dino_floor` returns
$$
d=\max(m,d_{\text{sched}})
$$
when source0 is ready and dual-source mode is active. `_build_alpha` injects this as floor on source0. By Step 3, when $F<1$, $\alpha_0\ge d\ge m$.

Step 6. In add-one reopen probe, `_build_alpha(..., extra_min_alpha_by_source={1:r})` merges floors by sourcewise max, so floor vector includes at least $(d,r)$. Under $d+r\le1$, Step 3 gives:
$$
\alpha_0\ge d,\qquad \alpha_1\ge r.
$$

Step 7. Collapse boundary reachability:
- per-probe and window-level predicates use
$$
\alpha_{\text{dino}} \le \texttt{s3a\_collapse\_alpha\_threshold},
$$
not strict $<$;
- validator enforces
$$
\texttt{s3a\_collapse\_alpha\_threshold}\ge m.
$$
Hence floor-level equality ($\alpha_{\text{dino}}=m$) is admissible for the alpha-threshold clause, so no structural dead-zone remains at equality.

Therefore claims (1)-(4) follow. ∎

## Corrections or Missing Assumptions
- Strict simultaneous floor guarantees in part (3) rely on $d+r\le1$ in the effective regime. This is consistent with validator intent but should remain an explicit contract assumption.
- This proof is about alpha/gate contract correctness, not about guaranteeing global optimization quality or preventing every form of DINO underuse.

## Open Risks
- The proof does not establish that collapse alarm catches all gradual underuse regimes; it only proves the floor-equality dead-zone is removed.
- If future code paths bypass `_build_alpha`/`_apply_joint_min_alpha`, these guarantees need re-verification.
