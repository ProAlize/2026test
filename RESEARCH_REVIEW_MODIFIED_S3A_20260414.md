# Research Review: Modified S3A Design Audit (2026-04-14)

## Reviewer Setup
- Reviewer M (method/narrative): `019d8c1f-fe83-7a70-a5b8-45d0a19119c5`
- Reviewer E (engineering/interface): `019d8c20-5389-7490-9689-a28f038fc4bb`
- Model/reasoning: `gpt-5.4`, `xhigh`

## Scope
Audit target: latest code modifications for S3A contract hardening.

Modified files:
- `train_s3a_multisource_dinov2.py`
- `scripts/run_e0_e7_single_seed.sh`
- `run_s3a_multisource_dinov2.sh`

## What Changed (reviewed)
1. Unsafe dual-source fail-fast:
- reject `use_ema_source=True && self_warmup<=0` unless explicit override.
2. Self shortcut mitigation:
- `--s3a-trainable-ema-adapters` (default off), default path avoids trainable self-target adapter.
3. Early DINO retention:
- `--s3a-dino-alpha-floor`, `--s3a-dino-alpha-floor-steps`.
4. Expanded observability:
- raw/effective alpha, router entropy, source-only losses, per-layer alpha, collapse alarms.
5. Launcher defaults hardened:
- E5/E6/E7 use warmup>0 + dino floor + non-trainable EMA adapter path.

## Round 1 Findings (method)
Verdict: improved, but still not fully claim-safe.

Major points:
- Upgraded from pure patch to a "guarded contract", but not yet principled reliability routing.
- Router still lacks source-specific reliability evidence input.
- DINO floor is a safety constraint, not proof router itself solved collapse.
- New launcher bundles multiple fixes together, weakening causal attribution.

Method score:
- Contract defensibility: 6/10
- Current paper posture: REVISE

## Round 1 Findings (engineering)
Verdict: safety improved, interface still has compatibility and maintainability risks.

Critical risks:
1. Checkpoint compatibility regression:
- state_dict shape may differ when `ema_adapters` are absent by default.
2. Monitoring overhead too high:
- source-only losses computed every step/layer in graph path.
3. Default contract split across parser vs launcher:
- behavior differs by entrypoint.

Engineering score:
- Maintainability: 6/10
- Testability: 5/10
- Reproducibility: 7/10
- Overall: 6.0/10

## Consensus
- This revision is a meaningful improvement and prevents the most obvious previous failure mode.
- It is still a REVISE-state design, not yet submission-grade as-is.
- Proceed only in revise-and-verify mode with strict causal experiments.

## Claims Matrix (post-modification)
Allowed now:
- "The revised implementation introduces explicit anti-collapse safeguards and observability."
- "Unsafe dual-source config is no longer silently accepted by default."

Not allowed now:
- "S3A has already validated principled reliability routing."
- "DINO injection effectiveness is solved" (without new causal runs).

## Priority TODO
### P0.5 (immediate)
1. Preserve checkpoint compatibility for `ema_adapters` state shape.
2. Move source-only probes to `no_grad` and/or `probe_every` cadence.
3. Unify safe defaults in parser and launcher to one canonical contract.

### P1
1. Gate update from utility instead of raw alpha confidence.
2. Persist collapse-alarm state across resume.
3. Refactor S3A into modular components (router/fusion/gate/monitor).

## Minimal Evidence Package Required Next
- `NoAlign-match`
- `DINO-only-match`
- `DualSrc-warmup5k`
- `PlaceboDINO-warmup5k`
- Optional follow-up: one 40k promotion run if above supports continue.

## Final Recommendation
- Continue mainline only as REVISE-track.
- Do not resume large-scale long runs for claim-building until P0.5 + minimal causal package complete.

## Addendum (post-review patch, same day)
After receiving reviewer feedback, two issues were immediately mitigated:
1. **Checkpoint compatibility**: `ema_adapters` structure remains stable when `use_ema_source=True`; trainability is now controlled by flag/optimizer path rather than module deletion.
2. **Default contract consistency**: parser defaults now align with launcher safety defaults (`dino_alpha_floor=0.1`, `floor_steps=5000`).

Additional engineering hardening:
- Added `--s3a-probe-every` and moved source-only probes to `no_grad` to reduce monitoring overhead.

Updated status:
- Still `REVISE`, but risk posture improved from “fragile safety patch” to “guarded, executable contract pending causal evidence”.
