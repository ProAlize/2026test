# S3A DINO Collaboration Audit Protocol (2026-04-15)

- Mode: local fallback review for `$research-review` because agent spawn/send APIs are not exposed in this session.
- Scope: `train_s3a_multisource_dinov2.py`
- Target claim: whether the current S3A implementation has actually solved "DINO collaboration" rather than merely preventing exact `a_dino=0`.

## Bottom-line verdict on the referenced run

Verdict: **not passed** for the claim "DINO collaboration is solved."

Why:

1. The code's own collaboration metric is `dual_synergy_margin = min(loss_dino_only, loss_self_only) - loss_fused_probe`, and it stayed negative in every logged window of the referenced run.
2. After self unlock, `raw_alpha_dino` collapsed from `0.030` at step 300 to about `0.006` by steps 400 to 600, so the router itself was no longer meaningfully selecting DINO.
3. `alpha_dino_above_floor` fell from `0.014` at step 300 to about `0.001` by steps 400 to 600, so DINO contribution was almost entirely floor-supported rather than learned.
4. `dino_starved_alarm=1` and `collapse_mitigation_triggered=1` appeared at step 600, meaning the system needed rescue rather than demonstrating stable synergy.

Evidence:

- Referenced run configuration and windows: `warmup=200`, `log_every=100`, `probe_every=10` in `monitor_logs/s3a_fix_smoke_20260415_r4.log:18-29` and `:123-143`.
- Failure windows: `monitor_logs/s3a_fix_smoke_20260415_r4.log:171-176`.
- Metric definitions:
  - DINO floor: `train_s3a_multisource_dinov2.py:418-429`
  - raw router alpha: `train_s3a_multisource_dinov2.py:1024-1029`
  - post-floor training alpha: `train_s3a_multisource_dinov2.py:1084-1104`
  - starvation / synergy / mitigation: `train_s3a_multisource_dinov2.py:2616-2695`

## 1. Minimal sufficient metric set

Do **not** judge from `a_dino` alone. The minimum sufficient set is:

| Metric | Why it is necessary | Definition in code |
| --- | --- | --- |
| `dual_synergy_margin` | Only direct evidence that fused dual-source target is better than the better single-source target. | `train_s3a_multisource_dinov2.py:2637-2643` |
| `alpha_dino_above_floor` | Separates real DINO contribution from floor-forced contribution. | `train_s3a_multisource_dinov2.py:2611-2617` |
| `raw_alpha_dino` | Detects router collapse hidden by the DINO floor. | `train_s3a_multisource_dinov2.py:1024-1029`, logged at `:2789-2791` |
| `dual_source_alive` | Quick coexistence check; useful but not sufficient by itself. | `train_s3a_multisource_dinov2.py:2628-2635` |
| `dino_starved_alarm` and `collapse_mitigation_triggered` | Hard fail sentinels: if rescue fires, collaboration is not solved. | `train_s3a_multisource_dinov2.py:2668-2695`, logged at `:2783-2785`, `:2811-2817` |

Not sufficient on their own:

- `a_dino`: can look healthy because of the enforced source0 floor.
- `gate_self`: can look healthy because the gate simply turned self off.
- `loss_align`: can go down in self-shortcut regimes and therefore is not a collaboration proof.

## 2. Thresholds and window lengths for 500 to 2k-step audits

Use the existing short-run logging contract:

- `log_every = 100` steps
- `probe_every = 10` steps
- One decision window = one 100-step metric window
- Judge collaboration only on **post-unlock** windows

Important interpretation rule:

- `500` to `800` steps is enough to declare an **early fail**
- `1000` to `2000` steps is the minimum range to declare a **pass**

Recommended thresholds:

| Metric | Pass threshold | Fail threshold | Window rule |
| --- | --- | --- | --- |
| `dual_synergy_margin` | median `> 0.02` and at least `2/3` post-unlock windows `> 0` | all first `3` post-unlock windows `<= 0`, or any `2` consecutive windows `< -0.05` | first `3` post-unlock windows |
| `alpha_dino_above_floor` | median `>= 0.02` | `<= 0.005` for `2` consecutive windows | first `3` post-unlock windows |
| `raw_alpha_dino` | median `>= 0.10` | `< 0.05` for `2` consecutive windows | first `3` post-unlock windows |
| `dual_source_alive` | `1` in at least `2/3` post-unlock windows | `0` in all first `3` post-unlock windows | first `3` post-unlock windows |
| `dino_starved_alarm` | always `0` | any `1` | any window |
| `collapse_mitigation_triggered` | always `0` | any `1` | any window |

Calibration against the referenced run:

- step 300: `raw_alpha_dino=0.030`, `alpha_dino_above_floor=0.014`, `dual_synergy_margin=-1.0065`
- steps 400 to 600: `raw_alpha_dino≈0.006`, `alpha_dino_above_floor≈0.001`, `dual_synergy_margin` stays negative, and rescue fires at step 600

This run therefore fails by every primary metric, not by a single fragile threshold.

## 3. Three-stage minimal experiment protocol

Hold everything else constant with the current E7 recipe from `scripts/run_e0_e7_single_seed.sh:256-370`:

- same model / data / seed
- same `layer_indices=auto`
- same `s3a_lambda=0.1`
- same holistic loss weights `feat=1.0`, `attn=0.5`, `spatial=0.5`
- same `diff_schedule=cosine`
- same `probe_every=10`, `log_every=100`

Only vary source/gate switches.

### Stage A: DINO-only

Purpose:

- prove the DINO branch itself is trainable and establish a DINO-only baseline

Switches:

- `--no-s3a-use-ema-source`
- `--no-s3a-enable-selective-gate`

Equivalent launcher setting:

- `run_s3a(... use_ema_source=0, use_gate=0, attn_weight=0.5, spatial_weight=0.5, diff_schedule=cosine)`

Decision:

- If DINO-only is unstable or `loss_align` does not improve at all across the run, stop here and debug the DINO path before talking about collaboration.
- This stage does **not** prove collaboration; it only establishes the DINO baseline `Ldino`.

### Stage B: dual-no-gate

Purpose:

- isolate whether the failure is in router / unlock policy rather than gate control

Switches:

- `--s3a-use-ema-source`
- `--no-s3a-enable-selective-gate`

Equivalent launcher setting:

- `run_s3a(... use_ema_source=1, use_gate=0, attn_weight=0.5, spatial_weight=0.5, diff_schedule=cosine)`

Decision:

- If this stage already fails the post-unlock thresholds above, the root problem is **not** the gate. It is the router + self-unlock mixture policy.
- If this stage passes but Stage C fails, the regression is introduced by gate logic.

### Stage C: full-gate

Purpose:

- test the actual current claim: gated dual-source training achieves stable DINO/self collaboration without rescue

Switches:

- `--s3a-use-ema-source`
- `--s3a-enable-selective-gate`

Equivalent launcher setting:

- current E7-style run, i.e. `run_s3a(... use_ema_source=1, use_gate=1, attn_weight=0.5, spatial_weight=0.5, diff_schedule=cosine)`

Decision:

- Must satisfy the same collaboration thresholds as Stage B.
- In addition, `dino_starved_alarm` must remain `0` and `collapse_mitigation_triggered` must remain `0`.
- If Stage C only "passes" because self is shut off and the run reverts to DINO-only, that is **not** a collaboration pass.

### Minimal runtime budget

- `<= 800` steps: fail-fast only
- `1000` to `1200` steps: minimum pass/fail audit
- `1500` to `2000` steps: preferred short-run confidence audit

### Claim logic by stage outcome

| Outcome | Allowed claim |
| --- | --- |
| A fails | DINO path itself may be broken; no collaboration claim allowed |
| A passes, B fails | self source overwhelms DINO even without gate; collaboration not solved |
| A passes, B passes, C fails | gate / mitigation logic breaks collaboration |
| A passes, B passes, C passes | short-run evidence supports "current S3A achieves nontrivial DINO/self collaboration" |

## 4. If the protocol fails, what code point to change first

Priority is based on the current failure pattern, where collapse happens immediately after self unlock and **before** gate logic helps.

### Priority 1: self unlock and post-unlock alpha policy

Code points:

- `train_s3a_multisource_dinov2.py:846-874`
- `train_s3a_multisource_dinov2.py:1079-1104`
- `train_s3a_multisource_dinov2.py:418-429`

Why first:

- In the referenced run, `raw_alpha_dino` collapses to near-zero right after unlock while `gate_self=1`.
- That means the first break is the mixture policy, not the gate threshold.

What to change first:

- add a short post-warmup release ramp for self
- or temporarily enforce a stronger DINO reservation immediately after unlock
- do this in alpha construction, not by adding another controller first

### Priority 2: router behavior itself

Code points:

- `train_s3a_multisource_dinov2.py:1024-1029`
- `train_s3a_multisource_dinov2.py:576-610` if you decide to touch `SourceReliabilityRouter`

Why second:

- If Stage B still fails after fixing unlock policy, the router is intrinsically learning the self shortcut too aggressively.

What to change:

- only then consider a router bias / temperature / mild regularization around unlock

### Priority 3: gate utility off/on logic

Code points:

- `train_s3a_multisource_dinov2.py:724-844`
- `train_s3a_multisource_dinov2.py:1322-1350`

Why third:

- Gate is not the first failure in the current trace.
- Only tune this first if Stage B passes and Stage C fails.

### Priority 4: collapse / mitigation state machine

Code points:

- `train_s3a_multisource_dinov2.py:2617-2695`

Why fourth:

- This block is rescue logic and monitoring.
- It can make logs easier to interpret, but it does not fix the primary mechanism if the router already collapsed.

## 5. PASS / FAIL template for an audit markdown

```md
## S3A DINO Collaboration Audit

- Run: <run_name>
- Git revision: <git_sha>
- Stage: <DINO-only | dual-no-gate | full-gate>
- Steps: <total_steps>
- Decision: <PASS | FAIL>

### Evidence windows
- Post-unlock windows used: <e.g. steps 300, 400, 500>
- dual_synergy_margin: <list>
- alpha_dino_above_floor: <list>
- raw_alpha_dino: <list>
- dual_source_alive: <list>
- dino_starved_alarm: <list>
- collapse_mitigation_triggered: <list>

### Pass criteria check
- synergy positive in at least 2/3 windows: <yes/no>
- DINO above-floor share nontrivial: <yes/no>
- raw router DINO share nontrivial: <yes/no>
- dual source actually alive: <yes/no>
- no rescue fired: <yes/no>

### Allowed claim
- <One sentence allowed claim>

### Not allowed claim
- <One sentence explicitly disallowed claim>
```

### PASS wording

Use this only if all Stage C thresholds pass:

> The current S3A implementation shows short-run evidence of genuine DINO/self collaboration: after self unlock, the fused probe loss beats both single-source probes in multiple windows, DINO contribution remains above the enforced floor by a nontrivial margin, the router assigns DINO nontrivial raw mass, and no starvation rescue is triggered.

### FAIL wording

Use this if any hard-fail condition appears:

> The current S3A implementation does not yet support the claim that DINO collaboration is solved. The observed behavior is floor-supported DINO presence or mitigation-driven fallback, not stable positive dual-source synergy.

## Practical summary

For the current referenced run:

- `A`: not evaluated here
- `B`: not evaluated here
- `C`: **fail**

Reason:

- the run demonstrates DINO preservation / rescue semantics, but not DINO+self synergy
- therefore the strongest defensible claim today is:

> "Current S3A can detect and mitigate DINO starvation in short runs."

Not yet defensible:

> "Current S3A has solved DINO collaboration."
