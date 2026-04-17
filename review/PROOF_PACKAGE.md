# Proof Package

## Claim
For the current implementation in `/home/liuchunfa/2026qjx/2026test/train_s3a_multisource_dinov2.py`, under:

- dual-source mode (`s3a_use_ema_source=True`, source0=DINO, source1=self),
- `validate_args(args)` passed,
- analysis restricted to runtime call paths inside `compute_s3a_alignment_loss(...)` and the main logging/mitigation block,

define:

- $m := \texttt{s3a\_protect\_source0\_min\_alpha}$,
- $d_{\text{sched}}(k)$ as decayed DINO floor at step $k$ from `source0_min_alpha_at_step`,
- $d(k) := \max\{m, d_{\text{sched}}(k)\}$,
- $r := \texttt{s3a\_gate\_reopen\_probe\_alpha\_floor}$.

Then:

1. Every runtime `_build_alpha` invocation in this graph returns $\alpha \in \Delta^2$:
$$
\alpha_i \ge 0,\qquad \sum_i \alpha_i = 1.
$$

2. For train/probe alpha builds where source0 is ready:
$$
\alpha_0 \ge d(k) \ge m.
$$

3. For add-one reopen probes using `extra_min_alpha_by_source={1:r}`, if $d(k)+r\le 1$:
$$
\alpha_0 \ge d(k),\qquad \alpha_1 \ge r.
$$

4. Under selective gate update with `protect_source0=True`, source0 controller lane is forcibly kept on:
$$
\texttt{source\_gate\_mask}[l,0] = 1,\ \forall\ \text{updated layer slot } l.
$$

5. Auto-mitigation has a floor-relative trigger channel independent of `s3a_collapse_alpha_threshold`: if windowed `dino_starved_alarm=1`, mitigation can trigger without using that absolute alpha threshold.

## Status
PROVABLE AS STATED

## Assumptions
- `validate_args(args)` passed.
- Runtime follows existing call guards in `compute_s3a_alignment_loss`.
- Source0 is available (`source_ready[0] > 0`) on the analyzed path.
- For Claim 3, validator feasibility condition holds when $r>0$:
$$
\max\{\texttt{s3a\_dino\_alpha\_floor},\,m\}+r\le1.
$$

## Notation
- $\hat{\alpha}$: masked router output before normalization.
- $\tilde{\alpha}$: normalized masked alpha before floor projection.
- $f$: per-source floor vector used by `_apply_joint_min_alpha`.
- $F:=\sum_i f_i$.
- $\Delta^2:=\{x\in\mathbb{R}_{\ge0}^2:\sum_i x_i=1\}$.

## Proof Strategy
Direct proof by code-path invariants and branch analysis:

1. Prove mask positivity on all runtime `_build_alpha` call sites.
2. Prove normalization + joint floor projection preserve simplex and floor lower bounds.
3. Prove source0 gate protection is hard-coded in gate update.
4. Prove mitigation has a threshold-independent sufficient trigger via `dino_starved_alarm`.

## Dependency Map
1. Mask positivity lemma depends on `get_source_mask` fallback and probe call guards.
2. Simplex lemma depends on `_build_alpha` normalization and `_apply_joint_min_alpha` branches.
3. Floor lemma depends on `_compute_dino_floor` and floor merge rule.
4. Source0 gate lemma depends on `update_gate_state(..., protect_source0=True)`.
5. Mitigation lemma depends on definitions of `dino_starved`, `dino_starved_alarm`, and mitigation trigger logic.

## Proof
Step 1. Runtime mask positivity.

Main training/probe call uses:
$$
\alpha=\_build\_alpha(\texttt{source\_mask}),
$$
where `get_source_mask` enforces fallback: if mask sum is non-positive, set source0 mask to $1$.

For policy-LOO probes:
- leave-one-out branch executes only when `source_mask.sum() > 1`;
- add-one branch executes only when modified mask sum is $>1$.

Hence all runtime `_build_alpha` calls here have strictly positive masked mass.

Step 2. Simplex preservation in `_build_alpha`.

Given positive mass, `_build_alpha` computes:
$$
\tilde{\alpha}=\frac{\hat{\alpha}}{\sum_j \hat{\alpha}_j},
$$
thus $\tilde{\alpha}\in\Delta^2$.

If no floor is active, output is $\tilde{\alpha}$.

If floor is active, `_apply_joint_min_alpha` handles three cases:
- $F\le0$: identity return.
- $0<F<1$: output is
$$
\alpha=f+(1-F)\hat{r},
$$
with $\hat{r}\in\Delta^2$, so $\alpha\in\Delta^2$ and $\alpha_i\ge f_i$.
- $F\ge1$: output is normalized floor vector, still in $\Delta^2$.

Therefore Claim 1 holds.

Step 3. Source0 floor lower bound.

`_compute_dino_floor` returns $d(k)$ when source0 is active and dual-source is enabled. `_build_alpha` inserts this as floor for source0 before projection. By Step 2:
$$
\alpha_0\ge d(k)\ge m.
$$
So Claim 2 holds.

Step 4. Add-one reopen floor bound.

In add-one branch, floor map merges source0 floor and `extra_min_alpha_by_source={1:r}` by componentwise max. Under $d(k)+r\le1$, the feasible floor branch applies and Step 2 gives:
$$
\alpha_0\ge d(k),\qquad \alpha_1\ge r.
$$
So Claim 3 holds.

Step 5. Source0 hard gate protection.

In `update_gate_state`, when `protect_source0=True` and source index is $0$, code sets gate mask to $1$ and resets counters without evaluating off/on utility conditions. Therefore source0 controller lane is forced on at each update. Claim 4 holds.

Step 6. Threshold-independent mitigation channel.

Main loop defines:
$$
\alpha_{\text{dino,above-floor}}:=\max(0,\alpha_{\text{dino}}-d(k)),
$$
and
$$
\texttt{dino\_starved}=1
$$
when post-warmup, self alpha is high, and $\alpha_{\text{dino,above-floor}}\le\varepsilon$ (with $\varepsilon=10^{-3}$). Window aggregation gives `dino_starved_alarm`.

Mitigation trigger condition includes:
$$
(\texttt{collapse\_alarm}>0)\ \lor\ (\texttt{dino\_starved\_alarm}>0).
$$
Hence `dino_starved_alarm` alone is a sufficient trigger path, which does not involve `s3a_collapse_alpha_threshold`. Claim 5 holds. $\square$

## Corrections or Missing Assumptions
- Previous proof text claimed validator enforced
$$
\texttt{s3a\_collapse\_alpha\_threshold}\ge m,
$$
which is no longer true in current code. That statement has been removed.
- Current proof is purely contract-level; it does not prove optimization-level collaboration.

## Open Risks
- Proven invariants prevent exact zero-collapse by contract but do not guarantee meaningful DINO contribution above floor.
- If future code adds new `_build_alpha` call paths without current mask guards, this proof must be revalidated.
