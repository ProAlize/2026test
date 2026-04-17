# S3A monitoring delta audit final (2026-04-15)

Scope: validate `scripts/monitor_training_20m.sh` for numeric safety and behavior on both current new logs and legacy logs.

Method:
- Read current monitor script and trainer log/metric emitters.
- Replay synthetic new-log and legacy-log snippets through the script.
- Probe shell and awk edge cases, including missing fields, scientific notation, and keyword-based error detection.

## Findings

1. Medium: `error_hit` can false-trigger `CRITICAL` on innocent text containing `inf`.
- Code: `scripts/monitor_training_20m.sh:140`
- Current pattern is `Traceback|RuntimeError|Non-finite|nan|NaN|inf|Inf|S3A collapse alarm triggered`.
- Because `rg` matches substrings, a benign line such as `inference warmup note` is treated as an error and forces `status=CRITICAL`.
- Local repro on a synthetic log confirmed `latest_error_line=... inference warmup note` and `reason=error_keyword_detected`.
- Impact: operational false alarms; the monitor is not production-ready as a generic log watcher until this is narrowed with word boundaries or more specific patterns.

2. Low: GPU snapshot fallback records command failure text as if it were a snapshot.
- Code: `scripts/monitor_training_20m.sh:146`
- When `nvidia-smi` fails, the shell pipeline still returns a non-empty string and the script writes it under `gpu_snapshot=`.
- This does not change status, but it can mislead downstream readers into treating an environment failure as a valid hardware snapshot.

3. Low: numeric parsing is only safe for the current fixed-decimal trainer format.
- Code: `scripts/monitor_training_20m.sh:36`
- `extract_field()` strips everything except digits, dot, and minus. If a future log ever prints scientific notation like `1e-4`, it becomes `1-4` and changes numeric meaning.
- Current trainer human logs use fixed formatting (`.2f`, `.3f`, `.4f`, `.6f`), so this is not a blocker for the present contract.

## Checks

1. New logs: pass within current contract.
- The script correctly consumed `a_dino_above_floor`, `dino_starved`, and `dual_alive` from current trainer log lines.
- Stateful counters behaved as intended across rounds in synthetic replay.

2. Legacy logs: pass within legacy heuristic scope.
- Missing `a_dino_above_floor`, `dino_starved`, and `dual_alive` no longer crash awk comparisons.
- The fallback path correctly uses safe defaults and continues to produce status lines.
- Output fields remain blank for unavailable legacy keys, which is acceptable for a text monitor.

3. No current shell crash found on empty-field paths.
- Under `set -euo pipefail`, the current `awk` comparisons no longer abort the monitor when new fields are missing.

## Production readiness scope

Acceptable now for:
- current S3A trainer human logs emitted by `train_s3a_multisource_dinov2.py`
- legacy S3A logs that follow the same `Loss=... (step=...)` line pattern
- engineering monitoring where occasional formatting limitations are acceptable

Not yet acceptable for:
- generic production log monitoring across arbitrary auxiliary log text
- any pipeline that treats `CRITICAL` as a pager-worthy signal without human review

Blocking reason:
- the `inf|Inf` substring matcher is too broad and can raise false `CRITICAL` alerts.

Minimal next fix:
- replace bare `inf|Inf` with a more exact non-finite detector, for example bounded tokens around `=inf`, `=nan`, or explicit `Non-finite` / traceback lines.
