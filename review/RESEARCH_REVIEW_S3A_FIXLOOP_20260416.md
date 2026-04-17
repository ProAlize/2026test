# Research Review Loop: S3A Fix-and-Launch (2026-04-16)

## Goal
Close audit blockers, re-run `$research-review` loop, then launch training.

## Patch Scope
Files patched in this loop:
- `/home/liuchunfa/2026qjx/2026test/train_s3a_multisource_dinov2.py`
- `/home/liuchunfa/2026qjx/2026test/run_s3a_multisource_dinov2.sh`
- `/home/liuchunfa/2026qjx/2026test/scripts/monitor_training_20m.sh`

### Key fixes
1. Launcher contract hardening:
- Added managed knob `S3A_ADAPTER_HIDDEN_DIM` and pass-through arg.
- Hardened managed-arg guard to block `--key`, `--key=*`, `--s3a-*`, `--no-s3a-*` override forms.
- Unified `DINOV2_WEIGHT_PATH` default with trainer parser.

2. Parser/contract hardening:
- Set `allow_abbrev=False` in argparse entrypoint.
- Updated hidden-dim validation to `0 -> None`, `<0 -> ValueError`.

3. Mechanism observability:
- Added and emitted canonical fields:
  - `alpha_dino_min_layer_above_floor`
  - `any_layer_dino_starved`
- Kept legacy aliases in text logs for transition compatibility.

4. Monitor correctness:
- Removed recoverable collapse-warning string from fatal regex.
- Error scan restricted to recent log tail.
- Parse canonical per-layer fields with fallback aliases.
- Parse numeric `dino_starved_alarm` and `collapse_alarm` explicitly.

## Research-Review Loop

### Round A (multi-agent)
- Mechanism reviewer (`019d9310-5114-71c1-8c9d-282eb3ccece2`): `CONDITIONAL`
- Claim reviewer (`019d9310-51a2-7171-a716-1168cc98cae3`): claim ceiling `C0` (engineering guardrail) before new evidence.
- Engineering reviewer (`019d9310-50c2-7291-9e66-51210add0873`): `FAIL`
  - Remaining HIGH then: default `DINOV2_WEIGHT_PATH` mismatch.

### Round B (after final patch)
- Engineering gate reviewer (`019d931a-34bb-73b2-8a1e-6614d8886877`): `PASS`
- Statement: no remaining HIGH blockers in static launch/repro/monitor gate.

## Training Launch

### First launch attempt
- Session: `s3a_fixloop_canary10k`
- Result: failed early with CUDA device-side assert.
- Root cause: class count mismatch (`num_classes=1000` default vs dataset has 1282 classes).

### Restart (active)
- Session: `s3a_fixloop_canary10k_r2`
- Monitor session: `s3a_fixloop_monitor20m_r2`
- Command profile: fresh no-resume 10k canary, explicit `--num-classes 1282`, warmup 200, KL 0.1, probe every 10.
- Launch log:
  - `/home/liuchunfa/2026qjx/2026test/monitor_logs/s3a_fixloop_canary10k_r2.log`
- Monitor logs:
  - `/home/liuchunfa/2026qjx/2026test/monitor_logs/s3a_fixloop_canary10k_r2_20m_analysis.log`
  - `/home/liuchunfa/2026qjx/2026test/monitor_logs/s3a_fixloop_canary10k_r2_20m_runtime.log`

### Runtime sanity observed
- DDP ranks started.
- S3A setup logged successfully.
- Step-1 metrics emitted; warmup behavior is consistent (`alpha_dino≈1`, `alpha_self≈0`).
- No immediate crash after restart.

## Current status
- Engineering launch contract: closed to `PASS` (static gate).
- Mechanism status: still `CONDITIONAL` pending new canary evidence.
- Training: running (`r2` session active).
