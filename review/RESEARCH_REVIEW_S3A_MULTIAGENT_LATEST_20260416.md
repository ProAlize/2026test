# Research Review: Multi-Agent Audit on Latest S3A (2026-04-16)

## Scope
- User request: `$research-review` multi-agent协同审查 latest S3A design.
- Code baseline reviewed:
  - Commit `67ef188`: critical fixes (independent ema_adapters, spatial normalization, DINO variants, adapter default)
  - Commit `be3e3e7`: codex-audit follow-up (resume compat, any-layer starvation, launcher pins)
- Primary files:
  - `/home/liuchunfa/2026qjx/2026test/train_s3a_multisource_dinov2.py`
  - `/home/liuchunfa/2026qjx/2026test/run_s3a_multisource_dinov2.sh`
  - `/home/liuchunfa/2026qjx/2026test/scripts/monitor_training_20m.sh`

## Reviewers
- Mechanism reviewer: `gpt-5.4 xhigh` (agent `019d9304-6f42-7702-87b5-96b647577570`)
- Engineering/interface reviewer: `gpt-5.4 xhigh` (agent `019d9304-6f99-7bc2-8961-222fc3bb08d3`)
- Results-to-claim reviewer: `gpt-5.4 xhigh` (agent `019d9304-7071-7b33-b7cd-bc9dd866cfdd`)

## Round Results

### Mechanism line
- Verdict: `CONDITIONAL`
- Consensus point:
  - Hard collapse (`source0 -> exact zero`) is much better controlled.
  - Long-tail marginalization risk remains: DINO can still be alive mostly via floor support rather than stable above-floor contribution.
- Key mechanism concerns:
  1. DINO utility has diagnosis power but weak direct control power in reopen/steady-state controller dynamics.
  2. Self source remains an endogenous shadow target (even after independent ema_adapters), still structurally favorable.
  3. Current alarm thresholds are tuned for catastrophic collapse, less sensitive to chronic weak collaboration regimes.

### Engineering/interface line
- Verdict: `FAIL`
- High-severity findings:
  1. Legacy resume path around `s3a_adapter_hidden_dim` is still system-fragile (trainer/launcher/contract interaction).
  2. Launcher trailing-arg guard is incomplete (`--key=value`, `--no-s3a-*` style can bypass intent and alter managed behavior).
  3. Monitor can permanently poison status after one historical collapse warning due to whole-log regex match.
- Medium findings:
  1. Telemetry contract drift: trainer text uses `mask_self`, monitor parses `gate_self` from text.
  2. `s3a_adapter_hidden_dim` normalization currently accepts any `<=0` as legacy fallback; should fail fast for `<0`.

### Results-to-claim line
- Verdict: `partial`
- Allowed now:
  - Engineering anti-collapse/guardrail claim (contract-level).
  - Short-run anti-starvation improvement claim.
- Not allowed now:
  - Mechanism solved.
  - Source-reliability routing learned.
  - Sustained above-floor DINO contribution and positive dual-source synergy.
- Provided evidence ladder `C0 -> C6` and minimal experiment package to promote claims.

## Converged Final Verdict

### Overall ship verdict (latest design as a complete system)
- `FAIL` for release-level confidence (because interface/monitor/launcher contract issues can produce wrong conclusions or non-reproducible runs).

### Mechanism maturity (assuming interface bugs are fixed)
- `CONDITIONAL`.
- Reason: anti-hard-collapse is stronger, but sustained collaboration beyond floor support is not yet closed by mechanism evidence.

### Claim status (without fresh successful 10k evidence)
- `partial`.
- Safest statement remains: engineering guardrail baseline, not mechanism effectiveness baseline.

## What Was Closed vs Still Open

### Closed compared with previous review rounds
1. Any-layer starvation signal is now included in starvation/collapse triggering logic.
2. Launcher now explicitly pins `--s3a-use-ema-source` and `--s3a-enable-selective-gate`.
3. `dinov2_model_variant` is integrated into parser/launcher/resume contract path.
4. Legacy missing-key compat for `s3a_adapter_hidden_dim` was partially addressed.

### Still open (blocking)
1. System-level resume usability for legacy hidden-dim contract is not robust enough.
2. Launcher managed-arg guard remains bypassable in important forms.
3. Monitor semantics can falsely latch `CRITICAL` after recoverable warning.
4. Text-log field vs monitor parser field mismatch persists.
5. Mechanism still lacks a robust DINO-recovery control contract (beyond floor-preservation).

## Prioritized Action List (engineering-first)
1. Harden launcher guard:
- Reject `--key`, `--key=*`, and `--no-s3a-*` override forms for managed args.
- Keep all managed behavior only via environment variables.

2. Finish adapter-hidden-dim legacy contract:
- Add launcher-managed `S3A_ADAPTER_HIDDEN_DIM`.
- Validate as `0 -> None`, `<0 -> ValueError`.
- Ensure legacy resume works without forcing `ALLOW_LEGACY_RESUME_ARGS=1`.

3. Fix monitor false-critical logic:
- Remove `S3A collapse alarm triggered` from fatal keyword path.
- Prefer latest `metrics.jsonl` numeric alarm fields for status.

4. Align telemetry contract:
- Either unify on one field naming (`mask_self` vs `gate_self`) or make monitor parse both explicitly.
- Gate monitor parser by `metrics_schema_version`.

5. Add one minimal mechanism patch:
- Introduce reopen hysteresis with DINO-recovery conditions (above-floor + raw-alpha margin + min-layer margin windows), not only self-side inactive utility.

## Minimal Evidence Package (to move claim upward)
- Fresh no-resume `10k` run (`Run M`) under current r6 contract.
- Two cheap ablations (`A1`: no router KL, `A2`: remove persistent source0 floor with unsafe flag).
- Conditional second seed repeat (`Run R`) only if `Run M` passes.
- Estimated cost from reviewer: ~13.6 GPU-hours total for core package.

## Claim Matrix (compressed)
- `C0`: no fresh 10k evidence -> engineering guardrail only.
- `C1`: no alarms but no dual_alive/synergy -> coexistence retention only.
- `C2`: sustained dual_alive + above-floor + positive synergy (single seed) -> narrow mechanism-effective claim.
- `C3/C4`: add causal ablation failures + second-seed repeat -> reproducible mechanism claim.
- `C5+`: add external endpoint advantage -> downstream benefit claim.

## Notes
- This audit intentionally prioritizes engineering落地: small contractual fixes first, then mechanism evidence.
- Large refactor is still optional and should be deferred until contract correctness is stable.
