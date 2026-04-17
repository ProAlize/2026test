# Research Review: S3A DINO Starvation Recovery (2026-04-15)

## Metadata
- Trigger: user requested multi-agent analysis after interrupting the failed run.
- Run status: no active `train_s3a_multisource_dinov2.py` process detected at review time.
- Delegation note: direct `spawn_agent` tooling was unavailable in this session, so the `research-review` workflow was executed as three independent local review tracks with the same deliverable structure.
- Scope:
  - `train_s3a_multisource_dinov2.py`
  - `scripts/run_e0_e7_single_seed.sh`
  - `monitor_logs/s3a_e7only_xl_s0_160k_v1.log`
  - `findings.md`
  - previous audit docs under `RESEARCH_REVIEW_S3A_*`

## Evidence Snapshot
- The interrupted long run is a failure for the intended dual-source story:
  - `a_dino` falls from `0.323` at step `200` to rounded `0.000` by step `1400`, then stays there for the rest of the run.
  - `a_self=1.000`, `gate_self=1.000` for the long-horizon regime.
  - `loss_align` remains tiny, which only proves the fused target became easy, not that DINO remained useful.
- Current launcher defaults are already safer than that failed run:
  - `S3A_SELF_WARMUP_STEPS=5000`
  - `S3A_DINO_ALPHA_FLOOR=0.1`
  - `S3A_DINO_ALPHA_FLOOR_STEPS=8000`
  - `S3A_PROTECT_SOURCE0_MIN_ALPHA=0.05`
- Therefore the interrupted run should be treated as a stale-contract falsification signal, not as evidence against the latest launcher defaults.

## Reviewer A: Mechanism

### Findings
1. The core failure is still objective-level, not gate-level.
- `get_source_mask()` guarantees that source0 is available.
- `_build_alpha()` can guarantee an effective floor on source0.
- Neither mechanism guarantees that the router actually wants DINO, or that DINO improves the fused objective once self is present.

2. Floor-only rescue can create a false-alive regime.
- If `raw_alpha_dino` collapses but `alpha_dino` stays non-zero only because of the floor, then DINO is mechanically present but not genuinely chosen by the router.
- In that regime, the training story is still weak: the contract is enforcing exposure, not demonstrating true dual-source cooperation.

3. Gate logic is not the shortest rescue path.
- The self gate acts on source1 utility.
- The main symptom is source0 starvation.
- Using more gate logic to rescue source0 is indirect and likely to produce more engineering complexity than scientific clarity.

### Recommendation
Use a staged rescue order:
1. Prove that DINO alone is useful under the current target/alignment stack.
2. Reintroduce self only after a real DINO anchoring period.
3. Only if raw router preference still collapses after that, add one minimal raw-router anti-starvation prior for the early coexistence window.

### Minimal mechanism patch if config-only rescue fails
Add an early raw-router prior on `raw_alpha_dino`, not on effective `alpha_dino`:
- preferred form: a hinge penalty such as `max(0, p_min - raw_alpha_dino)^2`
- active only during early coexistence, for example the first `5k-10k` steps after self warmup
- removed afterward so the late policy can still adapt

Reason:
- this directly penalizes the actual hidden failure mode (`raw_alpha_dino -> 0`),
- it is smaller and cleaner than adding new adapters / extra controllers / more gate states.

## Reviewer B: Engineering

### Findings
1. The current code already logs enough primary fields to avoid blind debugging.
Available in `metrics.jsonl`:
- `alpha_dino`, `alpha_self`
- `raw_alpha_dino`, `raw_alpha_self`
- `router_entropy_norm`
- `loss_fused_probe`, `loss_dino_only`, `loss_self_only`
- `utility_dino`, `utility_self`
- `alpha_dino_min_layer`, `alpha_dino_max_layer`, and per-layer `alpha_dino_l*`

2. What is still missing is not raw telemetry but decision-grade derived metrics.
The next validation loop should compute three derived quantities from existing logs:
- `dino_clamp_gap = alpha_dino - raw_alpha_dino`
- `dual_gain_vs_best_single = min(loss_dino_only, loss_self_only) - loss_fused_probe`
- `layer_floor_coverage = alpha_dino_min_layer / max(alpha_dino, 1e-8)` or simply the pair `(alpha_dino_min_layer, alpha_dino_max_layer)`

3. A long dual-source run should not be relaunched until a short run proves all three of these are healthy.
Otherwise the system can look numerically stable while still being scientifically dead.

### Recommendation
Prefer a small number of run-contract changes before new code patches:
1. Keep `use_trainable_ema_adapters=0`.
2. Keep `self_warmup_steps >= 5000` for real runs.
3. Strengthen the early DINO floor for the first rescue attempt:
- `s3a_dino_alpha_floor=0.20`
- `s3a_dino_alpha_floor_steps=10000`
- `s3a_protect_source0_min_alpha=0.10`
4. Do not use the old failed run to evaluate the new contract.

Reason:
- these are config-level interventions that target the known failure mode without expanding code surface area.

## Reviewer C: Validation / Result-to-Claim

### Findings
1. The first question is not "does `alpha_dino` stay above zero?"
The first question is "does dual-source beat the best single source under the same probe?"

2. A non-zero effective DINO share is insufficient.
A run is still a failure if:
- `alpha_dino` is held up only by the floor,
- `raw_alpha_dino` remains near zero,
- `loss_fused_probe` is not better than the best single source,
- or the improvement disappears as soon as self is enabled.

3. The shortest trustworthy decision tree is three-step:
- Step 1: `DINO-only`
- Step 2: `Dual, no selective gate`
- Step 3: `Dual, selective gate on`

This isolates whether the failure comes from:
- DINO itself,
- raw source competition,
- or gate policy.

### Recommendation: shortest recovery package
Run these in order and stop as soon as one stage fails.

1. `R1: DINO-only canary`
- `--no-s3a-use-ema-source`
- goal: verify that the DINO target and alignment stack actually improve probe behavior vs no-align

2. `R2: Dual-source coexistence canary without selective gate`
- self enabled after warmup
- `--no-s3a-enable-selective-gate`
- stronger early DINO floor as above
- goal: test whether the starvation is already solved by staged coexistence alone

3. `R3: Full S3A guarded run`
- same as `R2`, then enable selective gate
- only valid if `R2` already shows non-trivial DINO use and positive dual gain

## Consensus
The shortest credible solution is not "add more rescue logic".
It is:
1. establish DINO usefulness in isolation,
2. establish stable dual coexistence without gate confound,
3. then reintroduce selective gate,
4. and only add one small raw-router prior if coexistence still collapses.

## Decision Rules

### Safe to continue without code patch
If the current launcher defaults plus stronger early floors yield all of the following in the short canary:
- `raw_alpha_dino >= 0.10` after self warmup
- `alpha_dino >= 0.15`
- `dual_gain_vs_best_single > 0`
- `alpha_dino_min_layer >= 0.05`
- no collapse alarm / mitigation trigger

then the starvation problem is likely fixed at the run-contract level.

### Must patch code
If either of the following occurs after self warmup under the stronger config:
- `raw_alpha_dino < 0.05` for multiple consecutive log windows
- `dual_gain_vs_best_single <= 0` while `alpha_dino` is held up by the floor

then config-only rescue is insufficient, and the next patch should be the minimal raw-router early prior described above.

## What Not To Do
- Do not relaunch a long 40k-160k dual-source run before passing a short canary.
- Do not interpret non-zero `alpha_dino` alone as success.
- Do not add multiple new mechanisms at once; otherwise failure attribution becomes ambiguous again.
- Do not use gate behavior as the primary evidence that DINO is alive.

## Recommended Next Action
1. Launch `DINO-only` short canary.
2. If that passes, launch `Dual no-gate` short canary with stronger early floors.
3. If that passes, launch full guarded S3A short canary.
4. Only if step 2 fails, implement the single raw-router anti-starvation prior and rerun step 2.
