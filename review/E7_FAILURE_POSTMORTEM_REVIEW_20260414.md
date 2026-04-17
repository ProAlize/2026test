# E7 S3A Failure Postmortem Review (2026-04-14)

## Verdict

The current S3A design did not fail because DINO was unavailable. It failed because the training contract made DINO optional and made EMA-self the easiest target from step 0.

Under the observed E7 run, DINO injection was effectively dead by step ~800:
- step 1: `alpha_dino=0.4941`, `alpha_self=0.5059`
- step 200: `alpha_dino=0.323`, `alpha_self=0.677`
- step 400: `alpha_dino=0.020`, `alpha_self=0.980`
- step 800: `alpha_dino=0.001`, `alpha_self=0.999`
- step 27000: `loss_align=0.0003`, `a_dino=0.000`, `a_self=1.000`, `gate_self=1.000`

So the mechanism claim "DINO is being injected through dynamic multi-source routing" is unsupported in this run. What the system actually learned is an EMA-self shortcut.

## Core Evidence

- Launcher hardcodes self warmup to zero:
  - `scripts/run_e0_e7_single_seed.sh:271`
- EMA source is enabled from step 0 when warmup is zero:
  - `train_s3a_multisource_dinov2.py:685-686`
- Router outputs softmax over sources; masked alpha is only renormalized, not lower-bounded:
  - `train_s3a_multisource_dinov2.py:814-830`
- DINO protection only keeps gate mask open; it does not enforce nonzero DINO weight:
  - `train_s3a_multisource_dinov2.py:653-655`
- EMA source goes through trainable adapters:
  - `train_s3a_multisource_dinov2.py:611-621`
  - `train_s3a_multisource_dinov2.py:799-800`
- Entire S3A head is optimized jointly:
  - `train_s3a_multisource_dinov2.py:1200-1202`
- EMA starts as a deepcopy of the student and is synced with `decay=0` before training:
  - `train_s3a_multisource_dinov2.py:1108`
  - `train_s3a_multisource_dinov2.py:1275-1276`
- Current E7 log shows fast alpha collapse while `gate_self` stays at `1.000`:
  - `monitor_logs/s3a_e7only_xl_s0_160k_v1.log:137-145`
  - `monitor_logs/s3a_e7only_xl_s0_160k_v1.log:275`

## Failure Tree

### Symptom A
`alpha_dino` collapses to near zero by step ~800 and never recovers.

Direct cause:
- Router is trained to minimize alignment loss against the fused target `z* = sum alpha_k z_k`.
- If one source is consistently easier, the optimizer pushes alpha to that source boundary.

Design-layer root cause:
- The method contract does not encode "inject DINO".
- It encodes "pick whichever source minimizes the alignment objective".
- DINO is an optional competitor, not a required anchor.

### Symptom B
`feat/attn/spatial` all collapse rapidly toward zero.

Direct cause:
- The fused target becomes nearly pure EMA-self.
- Student-vs-EMA structural matching is trivial relative to student-vs-DINO transfer.

Design-layer root cause:
- The alignment metrics certify agreement with the fused target, not DINO contribution.
- Once the fused target becomes self-like, near-zero alignment loss is meaningless as evidence for external semantic transfer.

### Symptom C
Selective gate does not save DINO.

Direct cause:
- `protect_source0` only forces DINO gate mask to remain `1`.
- It does not impose an alpha floor.
- In logs, `gate_self` remains `1.000`, so collapse happens with both sources available.

Design-layer root cause:
- The gate mechanism was solving the wrong problem.
- It protects source availability, not source usage.

### Symptom D
Collapse happens extremely early, before any stable DINO anchoring phase.

Direct cause:
- `--s3a-self-warmup-steps 0` exposes EMA-self from step 0.
- `--s3a-train-schedule constant` keeps alignment pressure fully on from step 0.

Design-layer root cause:
- The experiment protocol introduced the shortcut before the desired teacher had any chance to shape the representation.

### Symptom E
EMA-self is overwhelmingly easier than DINO.

Direct cause:
- EMA starts from a copy of the student.
- EMA tracks the student continuously.
- EMA source also passes through trainable `ema_adapters`.

Design-layer root cause:
- The self source is structurally advantaged and over-parameterized.
- In this setup, router collapse toward self is not an accident; it is the expected optimum.

## Responsibility Weights

1. Optional-teacher objective: fused target plus unconstrained alpha makes DINO disposable.
   - Weight: 35%
   - Confidence: 0.97
2. Step-0 shortcut exposure: `self_warmup=0` with full-strength alignment from the first step.
   - Weight: 25%
   - Confidence: 0.95
3. Self source overpowered: EMA copy initialization plus trainable EMA adapters.
   - Weight: 20%
   - Confidence: 0.92
4. Router is not source-aware in the strong sense; it learns a generic preference, not true source reliability.
   - Weight: 10%
   - Confidence: 0.78
5. DINO "protection" is mis-specified: availability without usage constraints.
   - Weight: 10%
   - Confidence: 0.96

Total: 100%

## Bug vs Contract vs Protocol

- Method contract error: 65%
  - The loss function does not force DINO transfer and naturally rewards self-collapse.
- Experiment protocol error: 30%
  - The launcher hardcodes `self_warmup=0`, turning on the shortcut immediately.
- Implementation bug: 5%
  - Mostly the misleading safety assumption around source-0 protection and the hardcoded override of a safer parser default.

Bottom line: this is not mainly a coding bug. The code is largely doing what the method asked it to do.

## Minimal Rescue Plan

### P0
Protocol-only rescue. No code change.

- Keep E0-E4 unchanged.
- Re-run only E5-E7 as a shadow ladder:
  - `E5w = E5 + --s3a-self-warmup-steps 5000`
  - `E6w = E6 + --s3a-self-warmup-steps 5000`
  - `E7w = E7 + --s3a-self-warmup-steps 5000`
- Use the same seed and the same 10k screening budget first.

Why this is first:
- Minimal intervention
- Preserves causal comparability
- Highest acceptance lift per GPU week
- Directly tests whether the failure was mostly protocol-driven

### P1
Add a weak early-phase DINO retention constraint.

Minimal code option:
- Introduce `--s3a-dino-alpha-floor 0.1` for steps `< 10000`, then disable it.

Alternative:
- Early-phase router prior toward DINO, e.g. KL toward `[0.7, 0.3]` before 10k.

Why:
- If P0 still collapses right after warmup ends, the contract is still wrong.
- You need a mechanism that makes DINO non-optional during the anchoring phase.

### P2
Reduce self shortcut strength.

Minimal code option:
- Freeze `ema_adapters` until step 10000, or bypass them entirely in early training.

Why:
- Right now the self source is both close to the student and trainably re-shaped to be even closer.
- That is too strong a shortcut for a mechanism paper centered on external injection.

## What Not To Do First

- Do not spend another full 160k run on current E7.
- Do not start by tuning `lambda`, `attn_weight`, `spatial_weight`, or gate threshold.
- Do not claim any DINO-based mechanism result from the current E7 trajectory.

## 48-Hour Validation Package

### Exp 0: Offline source-decomposition probe on current E7 checkpoint

Hypothesis:
- H0: Router collapsed because EMA-self is already lower-loss than DINO at the per-source level.

Log:
- `raw_alpha`
- masked `alpha`
- per-layer `L_dino_only`
- per-layer `L_self_only`
- `L_fused`
- router entropy
- fraction of samples with `alpha_dino > 0.1`
- metrics binned by layer and diffusion timestep

Expected result if diagnosis is correct:
- `L_self_only << L_dino_only`
- `L_fused ~= L_self_only`
- alpha collapse mirrors the source-loss ordering

Time:
- 2 to 4 hours

### Exp 1: E7 control short run to 10k with richer logging

Hypothesis:
- H1: Current collapse is reproducible and happens before 1k even under a fresh restart.

Change:
- No method change
- Only add richer logging if missing

Success criterion:
- Reproduces `alpha_dino < 0.01` by ~1k

Time:
- About 3.5 hours

### Exp 2: E7w with `self_warmup=5000`

Hypothesis:
- H2: A large part of the failure is protocol: step-0 self competition kills DINO before anchoring.

Change:
- Only `--s3a-self-warmup-steps 5000`

Readout:
- `alpha_dino` at 200, 1k, 5k, 6k, 10k
- `L_dino_only` slope from 0 to 5k
- post-warmup collapse speed

Success criterion:
- `alpha_dino` stays near `1.0` before 5k
- After 5k, it should not instantly crash below `0.05`

Time:
- About 3.5 hours

### Exp 3: E7w plus early DINO alpha floor

Hypothesis:
- H3: If collapse resumes right after warmup, then the root problem is the method contract, not only the protocol.

Change:
- `self_warmup=5000`
- DINO alpha floor `0.1` until 10k

Readout:
- same as Exp 2
- plus whether `L_dino_only` actually improves relative to Exp 2

Success criterion:
- DINO occupancy survives after warmup release
- `L_dino_only` improves materially over Exp 2

Time:
- About 3.5 hours

### Exp 4: E7w plus frozen EMA adapters

Hypothesis:
- H4: Trainable EMA adapters are amplifying the shortcut.

Change:
- `self_warmup=5000`
- freeze or bypass `ema_adapters` until 10k

Readout:
- compare `L_self_only` gap versus Exp 2
- compare post-warmup `alpha_dino`

Success criterion:
- smaller self advantage gap
- slower or absent DINO collapse

Time:
- About 3.5 hours

### Promotion rule

Only promote one variant to 40k or 160k if all three are true by 10k:
- `alpha_dino` is still alive after self unlock
- `L_dino_only` is decreasing
- `loss_align` is not trivially near zero for purely self-like reasons

If none pass, stop. The method story is not ready for expensive FID runs.

## Mock Review

### Summary

This paper proposes S3A, a multi-source alignment mechanism for class-conditional diffusion that dynamically fuses DINO and EMA-self supervision. However, the central mechanism is not empirically supported in the provided experiments. In the main E7 setting, routing rapidly collapses to the EMA-self source, with DINO weight becoming negligible by about 800 training steps. As a result, the method does not demonstrate meaningful DINO injection in the claimed configuration.

### Strengths

- The paper asks a valid question: whether external semantic structure and self-consistency can be combined in diffusion training.
- The implementation is instrumented enough to expose a real mechanism failure rather than hiding it.
- The ladder from E0 to E7 is a reasonable starting ablation scaffold.

### Weaknesses

- The objective does not force DINO transfer; it only rewards whichever source is easiest.
- The self source is introduced at step 0 and is structurally much easier than DINO.
- The claimed dynamic routing mechanism is not validated as dynamic or source-reliability aware.
- Near-zero alignment loss is not evidence of successful external semantic injection.
- The main experimental configuration cannot support the headline claim.

### Questions

1. Why is `self_warmup` hardcoded to zero when the parser default is 5000?
2. Why should source-0 protection be interpreted as DINO retention if alpha is unconstrained?
3. What are `L_dino_only` and `L_self_only` over training?
4. Does any configuration keep nontrivial DINO occupancy after self is enabled?
5. If the best FID comes from self-collapse, what is the actual contribution of DINO?

### Score

3 / 10

### Confidence

4 / 5

### What Would Move Toward Accept

- A corrected protocol showing nontrivial DINO occupancy after self is enabled
- Source-specific loss decomposition proving DINO is not a dead expert
- A minimal causal ablation showing which fix prevents collapse
- Updated claims that match the evidence if true DINO injection still cannot be shown
