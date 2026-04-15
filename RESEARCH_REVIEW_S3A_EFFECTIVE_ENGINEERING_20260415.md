# S3A Effective Engineering Closure Log (2026-04-15)

## 1. Goal
Build an **effective engineering S3A version** with minimal patching, then run multi-round `$research-review` audits (3 agents, xhigh) until no blocking issues remain.

## 2. Implemented Changes

### 2.1 Core trainer (`train_s3a_multisource_dinov2.py`)
1. **Source0 safety in runtime mask**
- In `get_source_mask()`, source0 is re-enforced after selective-gate multiply.
- Ref: `/home/liuchunfa/2026qjx/2026test/train_s3a_multisource_dinov2.py:851`

2. **Joint floor composition (no sequential floor override bug)**
- Added `_apply_joint_min_alpha()` and upgraded `_build_alpha()` to support multi-source floor constraints in one pass.
- `policy_loo` add-one probe now uses `extra_min_alpha_by_source` through unified alpha builder.
- Refs:
  - `/home/liuchunfa/2026qjx/2026test/train_s3a_multisource_dinov2.py:1027`
  - `/home/liuchunfa/2026qjx/2026test/train_s3a_multisource_dinov2.py:1078`
  - `/home/liuchunfa/2026qjx/2026test/train_s3a_multisource_dinov2.py:1194`

3. **Resume contract tightened (fail-closed for non-legacy defaults)**
- Missing-key compatibility now allowed only when current values equal legacy defaults (`0.0`) for:
  - `s3a_protect_source0_min_alpha`
  - `s3a_gate_reopen_probe_alpha_floor`
- Added clearer guidance for `s3a_utility_probe_mode` mismatch.
- Ref: `/home/liuchunfa/2026qjx/2026test/train_s3a_multisource_dinov2.py:1578`

4. **Legacy migration sterilization expanded**
- Added `_reset_selective_gate_runtime_state()`.
- When legacy `source_utility_ema` exists, reset full selective-gate runtime state (mask/counters/EMAs/init flags/mitigation window) to expected defaults.
- Refs:
  - `/home/liuchunfa/2026qjx/2026test/train_s3a_multisource_dinov2.py:1624`
  - `/home/liuchunfa/2026qjx/2026test/train_s3a_multisource_dinov2.py:1652`

5. **Safer shipped defaults**
- `--s3a-protect-source0-min-alpha`: `0.05`
- `--s3a-gate-reopen-probe-alpha-floor`: `0.05`
- Refs:
  - `/home/liuchunfa/2026qjx/2026test/train_s3a_multisource_dinov2.py:3036`
  - `/home/liuchunfa/2026qjx/2026test/train_s3a_multisource_dinov2.py:3102`

6. **Arg-contract hardening**
- `--s3a-enable-selective-gate` now requires `--s3a-utility-probe-mode=policy_loo`.
- Nonzero reopen floor requires `policy_loo` and `use_ema_source`.
- Floor-feasibility check now uses step-aware DINO floor activation (`dino_alpha_floor_steps > 0`).
- Refs:
  - `/home/liuchunfa/2026qjx/2026test/train_s3a_multisource_dinov2.py:3195`
  - `/home/liuchunfa/2026qjx/2026test/train_s3a_multisource_dinov2.py:3211`

### 2.2 Launchers
1. `run_s3a_multisource_dinov2.sh`
- Synced defaults:
  - `S3A_PROTECT_SOURCE0_MIN_ALPHA=0.05`
  - `S3A_GATE_REOPEN_PROBE_ALPHA_FLOOR=0.05`
- Ref: `/home/liuchunfa/2026qjx/2026test/run_s3a_multisource_dinov2.sh:42`

2. `scripts/run_e0_e7_single_seed.sh`
- Synced defaults to `0.05`.
- Added per-run `reopen_probe_alpha_effective`:
  - if `use_ema_source != 1`, force `0.0`.
- Applied effective value consistently to:
  - CLI arg forwarding
  - contract suffix tag
  - header note (auto-zero behavior)
- Refs:
  - `/home/liuchunfa/2026qjx/2026test/scripts/run_e0_e7_single_seed.sh:266`
  - `/home/liuchunfa/2026qjx/2026test/scripts/run_e0_e7_single_seed.sh:320`

### 2.3 Contract doc sync
- Updated default/contract statements for protect/reopen floors and fail-fast behavior.
- Ref: `/home/liuchunfa/2026qjx/2026test/docx/implementation_contract_20260414.md:58`

## 3. Multi-Agent Audit Rounds

## Round 1 (after first code patch)
Agents (gpt-5.4, xhigh):
- Mechanism: `019d8d2f-2a21-7991-9379-8d519bf7af35`
- Engineering: `019d8d2f-2a65-7073-b4c5-546e9f6d7bf9`
- AC: `019d8d2f-2afc-7ce0-9700-a8f81e2daef4`

### Round-1 consensus
- Trainer core mostly fixed.
- Remaining blockers found:
1. `E0-E7` launcher could pass nonzero reopen floor even when `use_ema_source=0` (hard-fail under new validator).
2. `selective_gate => policy_loo` constraint still missing.
3. floor-feasibility check should be step-aware for `s3a_dino_alpha_floor_steps==0`.

### Round-1 follow-up fixes (implemented)
- Added `selective_gate => policy_loo` validator.
- Made floor-feasibility check step-aware.
- Added launcher-side `reopen_probe_alpha_effective` auto-zero logic for `no-ema-source` runs.

## Round 2 (after follow-up patch)
Same 3 agents re-audited.

### Round-2 outputs
- Mechanism agent verdict: **effective engineering S3A**.
- Engineering agent verdict: **Pass** on legacy safety/default safety/contract clarity/resume fail-closed.
- AC-style agent verdict: **Accept for controlled rollout**, no blocking findings.

## 4. Final Convergence Status
- Blocking findings: **None**.
- Decision: **REVISE complete, ACCEPT for engineering rollout**.
- Positioning: engineering contract is closed for controlled experiments; paper-claim boundary still should remain conservative (`guarded dual-source auxiliary alignment`).

## 5. Verification Executed
- `python -m py_compile train_s3a_multisource_dinov2.py` ✅
- `bash -n run_s3a_multisource_dinov2.sh` ✅
- `bash -n scripts/run_e0_e7_single_seed.sh` ✅
- Additional local semantic checks for:
  - resume missing-key fail-closed behavior
  - validation failures for invalid mode/floor combinations
  completed successfully.


## 6. DDP Unused-Parameter Fix Loop (Auto, 2026-04-15)

### 6.1 Problem observed
During dual-source canary runs, training failed at step 2 with DDP error:
`Expected to have finished reduction in the prior iteration...`
Missing-grad parameter indices were `56..111`, which map exactly to `ema_adapters.*`.

### 6.2 Root cause
1. `ema_adapters` trainable path was still detached before fusion, so gradients never reached `ema_adapters`.
2. In non-trainable mode, `ema_adapters` freezing was applied after DDP wrap, which is too late for reducer membership.

### 6.3 Code fixes
1. Keep gradient path in trainable self-source branch:
- file: `/home/liuchunfa/2026qjx/2026test/train_s3a_multisource_dinov2.py`
- area: `compute_s3a_alignment_loss()` around lines ~986-996.
- change: do not `.detach()` `ema_proj` when `use_trainable_ema_adapters=True`; keep detached behavior only for frozen fallback path.

2. Freeze `ema_adapters` before DDP construction when non-trainable:
- file: `/home/liuchunfa/2026qjx/2026test/train_s3a_multisource_dinov2.py`
- area: model setup around lines ~1956+.
- change: move `ema_adapters` requires_grad false logic to pre-DDP stage; remove post-DDP mutation branch.

### 6.4 Multi-agent review (research-review, xhigh)
Agents:
- Mechanism: `019d8d62-3e02-71d0-9f15-d4dfef00e734`
- Engineering: `019d8d62-3e44-73f3-9598-a21a2273e52f`
- AC-style: `019d8d62-3e96-7e00-a7fb-b6cb571532d3`

Converged result:
- No `P0/P1` blockers.
- Accepted for controlled rollout.
- One `P2` note: abnormal partial tap-capture could still create DDP brittleness, but not expected in normal configured runs.

### 6.5 Runtime verification
1. Dual-source canary (after fix) completed successfully:
- run dir:
`/tmp/s3a_effective_dualsrc_fix_20260415_030431/DiT-S-8-seed0-20260415-030439-51bbfc-s3a-dinov2-lam0.1-traincosine_decay-diffcosine`
- reached `max_steps=6` and exited normally.
- checkpoints created:
  - `checkpoints/0000003.pt` (+ sha256 sidecar)
  - `checkpoints/0000006.pt` (+ sha256 sidecar)

2. Metrics/logs confirmed:
- `utility_probe_mode=policy_loo`
- dual-source active (`alpha_self` non-zero, `gate_self=1.0` in run)
- no DDP reduction error after fix.

### 6.6 Status
DDP unused-parameter issue is closed for normal dual-source training path.

## 7. Final Post-Review Verification (2026-04-15)

### 7.1 Final review summary
A second targeted 3-agent review on the DDP fix converged to:
- no `P0/P1` blockers,
- accepted for controlled dual-source rollout,
- one `P2` residual note only for abnormal partial tap-capture edge cases.

### 7.2 Post-review run (required final check)
Executed dual-source S3A short run **after review**:
- run dir:
`/tmp/s3a_postreview_dualsrc_20260415_160008/DiT-S-8-seed0-20260415-160016-e36d5a-s3a-dinov2-lam0.1-traincosine_decay-diffcosine`
- config highlights:
  - `--s3a-trainable-ema-adapters`
  - `--s3a-self-warmup-steps 0 --s3a-allow-unsafe-zero-warmup`
  - `--max-steps 3`
- result:
  - completed to max_steps without DDP reduction error,
  - checkpoint written with sha256 manifest:
    - `checkpoints/0000003.pt`
    - `checkpoints/0000003.pt.sha256.json`

Conclusion: the specific DDP unused-parameter failure is resolved in practical dual-source execution.
