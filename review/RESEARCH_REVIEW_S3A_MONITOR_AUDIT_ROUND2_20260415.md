# Research Review: S3A Monitor Audit Round 2

Date: 2026-04-15
Mode: local fallback review for `$research-review` (no external `spawn_agent` interface exposed in this session)
Scope: `scripts/monitor_training_20m.sh` runtime monitoring logic for `a_dino_above_floor` / `dino_starved` / `dual_alive`

## Verdict

Not ready to ship as the canonical monitoring path.

The current script can correctly read the new trainer log fields on the current log format, and the `dino_starved` path does produce stable warn/critical escalation through persisted `dino_starve_rounds`. But the monitoring contract is still incomplete:

1. only `dino_starved` drives alert state,
2. `a_dino_above_floor` and `dual_alive` are logged but not used for alert semantics,
3. empty legacy fields can still trigger `awk` syntax errors in the fallback path.

If the intended release bar is only "current trainer logs can expose explicit starvation and the monitor will escalate after repeated windows", this is usable. If the release bar is "the new monitor correctly consumes the new dual-source health signals and is robust on old logs", this should not go live yet.

## Findings

### 1. High: only `dino_starved` is connected to alert state; `a_dino_above_floor` and `dual_alive` are observability-only

Relevant code:
- `scripts/monitor_training_20m.sh:77`
- `scripts/monitor_training_20m.sh:78`
- `scripts/monitor_training_20m.sh:81`
- `scripts/monitor_training_20m.sh:82`
- `scripts/monitor_training_20m.sh:94`
- `scripts/monitor_training_20m.sh:143`
- `train_s3a_multisource_dinov2.py:2713`
- `train_s3a_multisource_dinov2.py:2730`
- `train_s3a_multisource_dinov2.py:2731`

What is true now:
- the script successfully extracts `a_dino_above_floor`, `dino_starved`, and `dual_alive` from the current trainer log line,
- the alert state machine increments `dino_starve_rounds` only from explicit `dino_starved` or legacy fallback,
- `dual_alive` never affects `status` or `reason`,
- `a_dino_above_floor` never affects `status` or `reason`.

Why this matters:
- a run can stay in `dual_alive=0` for a long time without ever producing a monitor alert, as long as the trainer-side `dino_starved` threshold is not crossed,
- this means the monitor does not yet operationalize the full "dual-source alive" contract; it only operationalizes the stricter starvation endpoint.

Assessment:
- field ingestion is correct,
- alert semantics are incomplete.

### 2. Medium: legacy fallback is only partially robust; missing values can produce `awk` syntax errors and silent non-alerts

Relevant code:
- `scripts/monitor_training_20m.sh:34`
- `scripts/monitor_training_20m.sh:74`
- `scripts/monitor_training_20m.sh:79`
- `scripts/monitor_training_20m.sh:80`
- `scripts/monitor_training_20m.sh:96`

Observed repro:
- with a partial old-style metric line that contains `a_dino` but lacks `a_self` and `gate_self`, `extract_field` returns empty strings,
- the fallback condition expands into an invalid `awk` program like `BEGIN {exit !(0.000 <= 0.001 &&  >= 0.999 &&  >= 0.999)}`,
- `awk` prints a syntax error to stderr,
- the monitor then falls through to `dino_starve_rounds=0`.

Why this matters:
- current "old log compatibility" is good for older S3A logs that still contain `a_dino`, `a_self`, and `gate_self`,
- it is not robust for partial or drifted logs,
- failure mode is noisy stderr plus silent miss on starvation classification.

Assessment:
- compatibility is partial, not hard-safe.

### 3. Medium: the monitor still scrapes the human log instead of the canonical `metrics.jsonl` schema

Relevant code:
- `scripts/monitor_training_20m.sh:63`
- `scripts/monitor_training_20m.sh:102`
- `train_s3a_multisource_dinov2.py:2738`
- `train_s3a_multisource_dinov2.py:2759`
- `train_s3a_multisource_dinov2.py:2760`
- `train_s3a_multisource_dinov2.py:2761`
- `train_s3a_multisource_dinov2.py:2250`

What this means:
- the trainer already writes canonical metric rows with `alpha_dino_above_floor`, `dino_starved`, and `dual_source_alive`,
- the monitor ignores that stable machine-readable path and instead parses the human log alias `dual_alive`,
- this keeps the monitor coupled to string formatting and alias spelling in the `logger.info(...)` line.

Why this matters:
- current format works,
- future log-format drift can break the monitor even while `metrics.jsonl` remains correct,
- the monitor cannot currently leverage `contract_row` thresholds directly.

Assessment:
- not a correctness failure on today's logs,
- still the weaker engineering contract.

## Checks

### A. Can it read the new fields from the current trainer logs?

Yes.

The trainer log line includes:
- `a_dino_above_floor=...`
- `dino_starved=...`
- `dual_alive=...`

The monitor extracts those exact keys from the text log and writes them back out to the analysis log.

### B. Does it form stable alerts for explicit starvation?

Yes, for the explicit trainer-side starvation signal.

Behavior:
- `dino_starved=1` increments persisted `dino_starve_rounds`,
- `>=3` rounds yields `WARN`,
- `>=6` rounds yields `CRITICAL`.

That persistence makes the alert stable rather than flappy.

### C. Is old-log compatibility actually preserved?

Partially.

Works for:
- old S3A logs that still contain `a_dino`, `a_self`, `gate_self`.

Not robust for:
- partial metric lines,
- drifted old formats where one or more fallback fields are absent.

### D. Do shell/awk boundaries look clean?

No, not fully.

Main shell boundary issue:
- empty extracted fields are interpolated directly into `awk` numeric expressions without a default value,
- this can produce syntax errors and silently disable the fallback starvation detector.

## Repro Summary

I ran three local smoke cases against `scripts/monitor_training_20m.sh`:

1. Current-format log line with `dino_starved=1`
- result: new fields were read correctly,
- persisted starvation reached `WARN` once the state file was preloaded to two prior rounds.

2. Older-format log line with only `a_dino/a_self/gate_self`
- result: fallback compatibility worked,
- no stderr issues.

3. Partial old-style log line missing `a_self/gate_self`
- result: stderr showed `awk` syntax errors,
- starvation fallback silently failed.

## Ship Decision

Can it be directly shipped?

No, not as the final monitoring contract.

Reason:
- it is operational for current explicit `dino_starved` logs,
- but it does not yet turn `a_dino_above_floor` / `dual_alive` into alert semantics,
- and its legacy fallback still has empty-value shell fragility.

## Minimal Fix Direction

Keep this small and engineering-focused:

1. Guard numeric fallbacks with defaults before `awk` comparison.
2. Promote `dual_alive` into the status machine, at least as a persistent `WARN` when it stays `0` after warmup/unlock.
3. Prefer `metrics.jsonl` as primary input and keep text-log grep only as backward-compatible fallback.
