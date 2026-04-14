# Utility-Gate Research Review (2026-04-14)

## Verdict

Utility-driven gating is a real conceptual improvement over raw-alpha gating, because it attempts to gate by contribution rather than by router preference. However, the current implementation has a blocking logical flaw: once self is gated off, the current utility definition makes reopening effectively impossible under the default positive reopen threshold.

## Bottom Line

- conceptual improvement: yes
- implementation ready for short validation: not yet
- recommendation: fix one blocking issue, then proceed

## Blocking Issue

When self is gated off:
- `source_mask[1] = 0`
- fused target becomes effectively DINO-only
- `utility_self = loss_dino_only - loss_fused`
- therefore `utility_self ~= 0`

With default `--s3a-gate-utility-on-threshold > 0`, reopen can never accumulate enough positive EMA to reopen self.

This turns the gate into a mostly one-way switch rather than a hysteretic utility gate.

## Additional Risks

1. Probe metrics are averaged over `log_steps`, not probe count, so `probe_every > 1` changes the scale of logged probe metrics and alarm behavior.
2. Utility thresholds are absolute and unnormalized, making them fragile across schedules, layer sets, and loss-weight settings.
3. The gate is still asymmetric in practice because DINO is protected and only self is truly gated.
