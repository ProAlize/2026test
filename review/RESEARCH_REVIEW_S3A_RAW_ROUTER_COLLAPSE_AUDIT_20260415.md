# Research Review: S3A raw-router collapse audit (2026-04-15)

## Scope
- File audited: `train_s3a_multisource_dinov2.py`
- Question: why the current fixes still yield "DINO effectively stays alive, but raw router preference collapses", and what is the minimal engineering repair
- Mode: local fallback for `$research-review` because this session does not expose subagent spawn/send APIs

## Evidence summary

Latest smoke `r4` shows the split clearly in [`monitor_logs/s3a_fix_smoke_20260415_r4.log:171`](/home/liuchunfa/2026qjx/2026test/monitor_logs/s3a_fix_smoke_20260415_r4.log#L171) to [`monitor_logs/s3a_fix_smoke_20260415_r4.log:176`](/home/liuchunfa/2026qjx/2026test/monitor_logs/s3a_fix_smoke_20260415_r4.log#L176):

- step 300: `a_dino=0.110`, `raw_dino=0.030`, `a_self=0.890`
- step 400-600: `a_dino≈0.093-0.096`, `raw_dino=0.006`, `a_self≈0.904-0.907`, `dino_starved=1`
- step 600: `dino_starved_alarm=1`, `mitigate=1`
- step 700-800: `gate_self=0`, `a_dino=1.000`, but `raw_dino` remains `0.006`

This is not a logging bug. It is a real split between:

- `raw_alpha`: router output before floor/gate/mitigation
- `alpha`: effective training-time mixture after floor/gate/mitigation

## Root-cause chain

### 1. The main alignment objective trains the router only through the post-processed fused target

The router emits `raw_alpha` in [`train_s3a_multisource_dinov2.py:1024`](/home/liuchunfa/2026qjx/2026test/train_s3a_multisource_dinov2.py#L1024), but the training loss uses `alpha = _build_alpha(source_mask)` in [`train_s3a_multisource_dinov2.py:1084`](/home/liuchunfa/2026qjx/2026test/train_s3a_multisource_dinov2.py#L1084) to [`train_s3a_multisource_dinov2.py:1104`](/home/liuchunfa/2026qjx/2026test/train_s3a_multisource_dinov2.py#L1104), and then constructs the fused target from that `alpha` in [`train_s3a_multisource_dinov2.py:1106`](/home/liuchunfa/2026qjx/2026test/train_s3a_multisource_dinov2.py#L1106) to [`train_s3a_multisource_dinov2.py:1129`](/home/liuchunfa/2026qjx/2026test/train_s3a_multisource_dinov2.py#L1129).

There is no explicit router-side loss on `raw_alpha`. Probe utilities are computed under `torch.no_grad()` in [`train_s3a_multisource_dinov2.py:1142`](/home/liuchunfa/2026qjx/2026test/train_s3a_multisource_dinov2.py#L1142) to [`train_s3a_multisource_dinov2.py:1268`](/home/liuchunfa/2026qjx/2026test/train_s3a_multisource_dinov2.py#L1268), and the state machine only consumes those diagnostics in [`train_s3a_multisource_dinov2.py:1322`](/home/liuchunfa/2026qjx/2026test/train_s3a_multisource_dinov2.py#L1322) to [`train_s3a_multisource_dinov2.py:1366`](/home/liuchunfa/2026qjx/2026test/train_s3a_multisource_dinov2.py#L1366).

So the router is only indirectly trained by the fused loss, not by the controller signals.

### 2. The self source is a much easier shortcut than DINO once warmup ends

The self source uses EMA tokens projected through the same student adapter path under `no_grad` by default in [`train_s3a_multisource_dinov2.py:997`](/home/liuchunfa/2026qjx/2026test/train_s3a_multisource_dinov2.py#L997) to [`train_s3a_multisource_dinov2.py:1010`](/home/liuchunfa/2026qjx/2026test/train_s3a_multisource_dinov2.py#L1010).

That makes self-target matching far easier than DINO-target matching:

- at step 300, `Ldino=12.2229`, `Lself=0.0776`
- at step 400, `Ldino=12.8393`, `Lself=0.0949`

from [`monitor_logs/s3a_fix_smoke_20260415_r4.log:171`](/home/liuchunfa/2026qjx/2026test/monitor_logs/s3a_fix_smoke_20260415_r4.log#L171) and [`monitor_logs/s3a_fix_smoke_20260415_r4.log:172`](/home/liuchunfa/2026qjx/2026test/monitor_logs/s3a_fix_smoke_20260415_r4.log#L172).

Given the current objective, the raw router is behaving rationally when it drives toward self after unlock.

### 3. The source0 floor prevents exact death, but also creates a gradient dead zone for raw recovery

The source0 floor is enforced by `source0_min_alpha_at_step()` in [`train_s3a_multisource_dinov2.py:418`](/home/liuchunfa/2026qjx/2026test/train_s3a_multisource_dinov2.py#L418) to [`train_s3a_multisource_dinov2.py:429`](/home/liuchunfa/2026qjx/2026test/train_s3a_multisource_dinov2.py#L429), and then injected by `_apply_joint_min_alpha()` inside `_build_alpha()` in [`train_s3a_multisource_dinov2.py:1043`](/home/liuchunfa/2026qjx/2026test/train_s3a_multisource_dinov2.py#L1043) to [`train_s3a_multisource_dinov2.py:1102`](/home/liuchunfa/2026qjx/2026test/train_s3a_multisource_dinov2.py#L1102).

When `raw_alpha_dino < floor`, the two-source output becomes almost constant:

- forward `alpha` is pinned near `[floor, 1-floor]`
- `raw_alpha` changes no longer materially change the fused target
- so the fused loss stops giving useful recovery gradient back to the router

That is exactly the observed regime at step 400-600:

- `raw_dino=0.006`
- `a_dino≈0.093-0.096`

The floor saves execution-time alpha, but not router preference learning.

### 4. Mitigation changes the executed mask, not the router objective

Mitigation is implemented by forcing self off in [`train_s3a_multisource_dinov2.py:698`](/home/liuchunfa/2026qjx/2026test/train_s3a_multisource_dinov2.py#L698) to [`train_s3a_multisource_dinov2.py:710`](/home/liuchunfa/2026qjx/2026test/train_s3a_multisource_dinov2.py#L710), and then masking self inside `get_source_mask()` in [`train_s3a_multisource_dinov2.py:846`](/home/liuchunfa/2026qjx/2026test/train_s3a_multisource_dinov2.py#L846) to [`train_s3a_multisource_dinov2.py:874`](/home/liuchunfa/2026qjx/2026test/train_s3a_multisource_dinov2.py#L874).

The windowed trigger is in [`train_s3a_multisource_dinov2.py:2616`](/home/liuchunfa/2026qjx/2026test/train_s3a_multisource_dinov2.py#L2616) to [`train_s3a_multisource_dinov2.py:2696`](/home/liuchunfa/2026qjx/2026test/train_s3a_multisource_dinov2.py#L2616).

Under mitigation, the forward policy becomes effectively DINO-only, so `a_dino=1.0`. But the router still has no dedicated correction signal. Therefore the log can show:

- `gate_self=0`
- `a_dino=1.0`
- `raw_dino=0.006`

as in [`monitor_logs/s3a_fix_smoke_20260415_r4.log:175`](/home/liuchunfa/2026qjx/2026test/monitor_logs/s3a_fix_smoke_20260415_r4.log#L175) to [`monitor_logs/s3a_fix_smoke_20260415_r4.log:176`](/home/liuchunfa/2026qjx/2026test/monitor_logs/s3a_fix_smoke_20260415_r4.log#L176).

### 5. The current state machine therefore guarantees "effective non-death", not "raw preference restoration"

The current contract is:

- source0 may not fully disappear in effective alpha
- starvation can trigger temporary self shutdown
- but none of that rewrites the router's internal preference model

So the present behavior is expected, not contradictory:

- floor/mitigation repair the forward path
- router collapse persists on the latent path

## Minimal engineering repair

### Recommendation

Add one explicit router-policy alignment loss in `compute_s3a_alignment_loss()`.

Core idea:

- keep the current forward policy `alpha_policy = _build_alpha(source_mask)` exactly as-is
- keep the current gate/floor/mitigation controller exactly as-is
- add a detached auxiliary loss that teaches `raw_alpha` to absorb persistent controller corrections

Concretely:

1. In [`train_s3a_multisource_dinov2.py:1104`](/home/liuchunfa/2026qjx/2026test/train_s3a_multisource_dinov2.py#L1104), rename current `alpha` to `alpha_policy`.
2. Right after it, add:

   - `router_policy_kl = KL(raw_alpha || alpha_policy.detach())`
   - or equivalently `KL(alpha_policy.detach() || raw_alpha)`; use one direction consistently and log it

3. Add `args.s3a_router_policy_kl_lambda * router_policy_kl` into `layer_loss` before accumulation.
4. Log one extra metric:

   - `router_policy_gap = mean(abs(alpha_policy[:, 0] - raw_alpha[:, 0]))`

This is the smallest coherent repair because it does not add a new controller. It connects the existing controller to router learning.

### Why this is the right minimal fix

- It addresses the exact observed failure:
  - today the controller edits `alpha` after routing
  - tomorrow the router learns from those edits
- It is local to `compute_s3a_alignment_loss()`
- It does not require a new state machine, new buffers, or resume-format churn
- It is easy to verify from existing logs

### Why not make the repair bigger in patch 1

There is a deeper asymmetry problem: self source is still much easier than DINO because it is projected through the student adapter path. But that is a second-order contract issue. Do not mix that redesign into the first patch whose goal is only to close the raw/effective split.

If router-policy KL is added and raw still immediately re-collapses after mitigation ends, that will be strong evidence that the remaining problem is the self-source shortcut itself, not the controller/router disconnect.

## Function-level modification points

### `compute_s3a_alignment_loss()`

Primary edit site: [`train_s3a_multisource_dinov2.py:934`](/home/liuchunfa/2026qjx/2026test/train_s3a_multisource_dinov2.py#L934)

Make these changes:

1. Keep `raw_alpha` as the pure router output.
2. Treat current `_build_alpha(source_mask)` result as `alpha_policy`.
3. Build fused target from `alpha_policy`, not from a new routing rule.
4. Add:
   - `router_policy_kl_acc`
   - `router_policy_gap_acc`
5. Add auxiliary loss term:
   - `layer_loss = weighted_alignment_loss + lambda * router_policy_kl`
6. Expose the new stats in the returned `stats` dict and global logger.

### `S3AAlignmentHead`

No structural state change is required for patch 1.

Do not add another mitigation buffer or another gate flag here. The bug is not missing state. The bug is that state changes do not supervise the router.

### `validate_args()`

Primary edit site: [`train_s3a_multisource_dinov2.py:3290`](/home/liuchunfa/2026qjx/2026test/train_s3a_multisource_dinov2.py#L3290)

Add validation for one new scalar:

- `--s3a-router-policy-kl-lambda >= 0`

### Parser / config row

Primary edit sites:

- parser area around [`train_s3a_multisource_dinov2.py:3180`](/home/liuchunfa/2026qjx/2026test/train_s3a_multisource_dinov2.py#L3180)
- startup contract row around [`train_s3a_multisource_dinov2.py:2267`](/home/liuchunfa/2026qjx/2026test/train_s3a_multisource_dinov2.py#L2267)

Add:

- `--s3a-router-policy-kl-lambda`
- contract/metrics logging for the new value

## Hyperparameters

Only one new hyperparameter is needed for patch 1.

### `--s3a-router-policy-kl-lambda`
- default: `0.02`
- reason:
  - large enough to matter when controller correction is extreme, especially mitigation windows like `alpha=[1,0]` vs `raw=[0.006,0.994]`
  - small enough that the main alignment loss remains primary when raw and policy are already close

I do not recommend adding a second hyperparameter in patch 1. Keep it simple.

## Explicitly rejected "fixes"

### 1. More floors, longer mitigation windows, lower alarm thresholds

Reject.

Reason:

- these only strengthen downstream correction
- they do not teach the router anything
- they make the raw/effective split worse, not better

### 2. Remove `source0` floor or allow unsafe zero-floor by default

Reject.

Reason:

- this reopens exact DINO death
- it solves the symptom by deleting the safety contract

### 3. Force raw router directly to one-hot DINO everywhere

Reject.

Reason:

- that is not "recovering preference"
- that is hard-coding a preference
- it will create false-positive recovery metrics and hide whether DINO is actually useful

### 4. Disable self source permanently or stretch warmup until the run is basically DINO-only

Reject.

Reason:

- that avoids the dual-source problem instead of solving it
- it makes the experiment claim weaker

### 5. Use `--no-s3a-router-detach-input` as the main fix

Reject as primary repair.

Reason:

- it changes feature coupling
- it does not connect controller output back to router learning
- it may change dynamics, but it does not address the identified mechanism break

### 6. Turn on trainable EMA adapters as a quick default fix

Reject for patch 1.

Reason:

- it adds capacity to the self path before fixing the policy-learning disconnect
- it can make the self shortcut even easier

## Short-run success criteria (500-2k steps)

Use the same smoke shape as `r4`, with short warmup and current logging.

### Must-pass criteria

1. During any mitigation window, `raw_dino` must move materially upward instead of staying flat.
   - target: within the first mitigation hold window, `raw_dino` rises from pre-trigger collapse level by at least `+0.05`
   - example: from `0.006` to `>= 0.056`

2. The raw/effective gap must shrink.
   - target: `|a_dino - raw_dino| < 0.15` within 1-2 log windows after mitigation trigger
   - today the gap is about `0.994` at step 700

3. Reopen should not immediately snap back to the exact old latent collapse.
   - target: after self gate reopens, `raw_dino` should remain clearly above the collapse floor for at least one full log window
   - practical target: `raw_dino >= 0.05`

4. Existing safety must remain intact.
   - still no exact `a_dino=0`
   - still no loss/grad instability

### Nice-to-have but not required for patch 1

- `dual_synergy_margin` improves
- `raw_dino` and `a_dino` stay close even outside mitigation

Those belong to the deeper source-contract question, not the first repair.

## Bottom line

The present code is behaving consistently with its current mechanism:

- floor and mitigation keep DINO alive in the forward path
- but raw router collapse persists because controller corrections do not train the router

The minimal repair is not another alarm or another floor. It is one auxiliary loss that makes the router learn the already-decided policy corrections.
