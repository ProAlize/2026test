# Research Review: S3A DINO Deactivation Risk Audit (2026-04-15)

## Scope
- Target code: `/home/liuchunfa/2026qjx/2026test/train_s3a_multisource_dinov2.py`
- Launcher context:
  - `/home/liuchunfa/2026qjx/2026test/run_s3a_multisource_dinov2.sh`
  - `/home/liuchunfa/2026qjx/2026test/scripts/run_e0_e7_single_seed.sh`
- Objective: deep mechanism review on possible DINO deactivation / DINO underuse blind spots.

## Reviewer Setup (`research-review`, `gpt-5.4`, `xhigh`)
- Mechanism: `019d902a-e6ec-7da1-8d33-1b4f64c6e34a`
- Engineering: `019d902a-e73b-7850-9469-f7b3016d6378`
- AC-style: `019d902a-e7c6-7242-8b6d-286aa9c82b44`

## Round 1 Findings (before hotfix)
Converged high-risk point:
1. **P0** collapse alarm dead-zone at shipped defaults:
- alarm condition used strict `<` for `alpha_dino < s3a_collapse_alpha_threshold`;
- defaults were `s3a_protect_source0_min_alpha=0.05` and `s3a_collapse_alpha_threshold=0.05`;
- with persistent floor, floor-level DINO underuse could not satisfy strict `<`.

Other consistent observations:
- source0 protection ensures availability, not guaranteed effective contribution;
- `raw_alpha_dino` can collapse while `alpha_dino` is clamped by floor;
- current alarm is a harmful-collapse detector, not a general DINO-underuse detector.

## Applied Minimal Hotfix
To avoid over-patching, only P0-level contract fix was applied:

1. Collapse predicates switched to inclusive boundary:
- `alpha_dino < threshold` -> `alpha_dino <= threshold`
- in both per-probe and window-level checks.
- refs:
  - `/home/liuchunfa/2026qjx/2026test/train_s3a_multisource_dinov2.py:1311`
  - `/home/liuchunfa/2026qjx/2026test/train_s3a_multisource_dinov2.py:2520`

2. Added fail-fast validation for unreachable threshold below persistent DINO floor:
- reject when `s3a_collapse_alpha_threshold < s3a_protect_source0_min_alpha`
- ref:
  - `/home/liuchunfa/2026qjx/2026test/train_s3a_multisource_dinov2.py:3223`

No redesign of controller/gating logic was introduced.

## Round 2 Follow-up Audit (after hotfix)
Same three reviewers re-audited with patch context.

Converged status:
- Previous P0 dead-zone is resolved.
- No new P0/P1 blockers introduced by this hotfix.
- Remaining risks are downgraded to **P1/P2 mechanism scope**:
  - still possible soft underuse path where alarm may stay silent if `utility_dino <= 0`;
  - current alarm remains “harmful collapse” rather than generic underuse alert.

## Final Technical Position
1. **Closed**
- Floor-equality blind spot in collapse trigger under shipped defaults.

2. **Still Open (non-blocking for current engineering run contract)**
- Soft DINO underuse observability is limited when underuse is not measured as beneficial (`utility_dino <= 0`).
- Optional future improvement: add separate warning diagnostic based on sustained low `raw_alpha_dino` / floor-clamp stats.

## Updated Claim Boundary
- Allowed:
  - DINO cannot be hard-gated off in dual-source selective-gate mode.
  - Floor-level collapse condition is now reachable and no longer dead by strict inequality.
- Not allowed:
  - Claiming guaranteed early detection of all gradual DINO underuse modes.
  - Claiming guaranteed preservation of strong DINO anchor effectiveness in all regimes.

## Runtime Verification After Hotfix
A post-hotfix dual-source smoke run completed successfully:
- run dir:
`/tmp/s3a_dino_review_hotfix_20260415_162625/DiT-S-8-seed0-20260415-162634-8a2af4-s3a-dinov2-lam0.1-traincosine_decay-diffcosine`
- command characteristics:
  - dual-source S3A
  - `--s3a-trainable-ema-adapters`
  - `--s3a-self-warmup-steps 0 --s3a-allow-unsafe-zero-warmup`
  - `max_steps=1` smoke
- outputs:
  - `checkpoints/0000001.pt`
  - `checkpoints/0000001.pt.sha256.json`
- exit: normal (`Done!`).

## Decision
- Status: **ACCEPT (engineering)** for current controlled training path.
- Recommendation: keep current hotfix; optionally add lightweight DINO-underuse warning telemetry in a separate small patch.
