# Modified S3A Research Review (2026-04-14)

## Verdict

The revised S3A is no longer a clearly broken contract. It is now a guarded, partially defensible training contract that explicitly blocks the previously observed catastrophic failure mode. However, it is still not a cleanly defensible "source reliability routing" method in the NeurIPS/ICML sense.

Current status:
- old failure mode is blocked by construction
- observability is much better
- but the central method claim is still stronger than what the implementation currently justifies

## Score

- Contract defensibility: 6/10

Interpretation:
- above "patch-only"
- below "methodologically clean"

## Main Remaining Risks

1. Router is still not truly source-aware.
   - It conditions only on student tokens, layer, timestep, and phase.
   - It does not ingest source-specific evidence.
   - This weakens any strong claim that it learns source reliability rather than a learned prior.

2. Default self path is still co-moving with the student adapter.
   - `pred = student_adapter(student_tokens)`
   - default self target = `student_adapter(ema_tokens)` under `no_grad`
   - The self target is not fixed; it shares the same evolving projector geometry.
   - This still favors the self branch relative to fixed DINO targets.

3. DINO alpha floor is a guardrail, not a principled routing solution.
   - It is applied after routing.
   - Gate updates still use `raw_alpha`.
   - The router can remain internally collapsed while the effective alpha is artificially kept alive.

4. Launcher now bundles multiple fixes at once.
   - This is good for safety.
   - It is bad for causal attribution in a paper unless isolated ablations are run.

## What Can Be Claimed If Minimal Experiments Pass

- staged source release prevents immediate DINO death
- weak DINO retention plus self-shortcut suppression can maintain nontrivial DINO participation
- the revised S3A is a safeguarded dual-source curriculum, not yet a proven reliability router

## What Should Not Be Claimed Yet

- that the router intrinsically learns source reliability
- that the default dual-source objective naturally preserves DINO
- strong interpretability claims from alpha curves alone

## Required Minimal Experiments

1. Old E7 vs new guarded E7 at 10k with mechanism logs
2. Warmup-only ablation
3. Warmup + alpha-floor ablation
4. Warmup + alpha-floor + trainable-EMA toggle ablation
5. Promote the best variant to 40k or longer and report FID + source diagnostics

Without these, the revised method is still a reasonable implementation revision, not a submission-ready evidence package.

## Final Recommendation

- Continue the line only as REVISE-AND-VERIFY
- Do not treat the current code revision alone as enough to stabilize the paper story
