# Deep Review: t-aware / SASA / S3A

Date: 2026-04-13
Mode: local code review + `research-review` 3-agent external audit (`gpt-5.4`, `xhigh`)

> Errata (2026-04-14): later confirmed that canonical `t-aware` implementation is in branch
> `exp_taware_adm_eval_20260410` rather than current-branch `train_2.py`.  
> The `t-aware` sections below reflect the 2026-04-13 assumption and should be interpreted accordingly.

## 1) Scope
Reviewed code and launchers:
- `train_2.py` (t-aware legacy path)
- `train_sasa.py`
- `train_sasa_dinov2.py`
- `train_s3a_multisource_dinov2.py`
- `run_dit_xl_repa_linear_80k.sh`
- `scripts_archive_20260410/run_dit_xl_repa_taware_b24_160k.sh`
- `run_sitxl_repa_dinov2_400k.sh`
- `run_sasa.sh`
- `run_sasa_dinov2.sh`
- `run_s3a_multisource_dinov2.sh`
- `diffusion/gaussian_diffusion.py`

External reviewers:
- Reviewer A (method/novelty): `019d8783-75a0-78e2-b880-71b1012fde0d`
- Reviewer B (code reliability): `019d8783-75e5-7ee1-90eb-f6bcf0e05733`
- Reviewer C (causal experiment design): `019d8783-765e-7bf3-a041-bd20c229f557`

## 2) Findings (ordered by severity)

### Critical
1. DINOv2 teacher loading is fail-open in SASA-DINOv2 and SiT-REPA variants.
- Evidence: `strict=False` + only print missing/unexpected keys:
  - `train_sasa_dinov2.py:311`
  - `train_sitxl_repa_dinov2_400k.py:253`
- Risk: malformed teacher checkpoints can silently pass and contaminate conclusions.
- Reference good pattern: strict loading and checkpoint-type guard in `train_s3a_multisource_dinov2.py:429`, `train_s3a_multisource_dinov2.py:452`.

2. `t-aware` in current code is not truly timestep-aware and is method-name overclaim.
- Evidence:
  - only train-step scalar schedule: `train_2.py:245`
  - align loss is plain cosine: `train_2.py:582`
- Risk: claims about diffusion-timestep-aware mechanism are unsupported.

### High
3. `t-aware` (`train_2.py`) uses two different noisy latents for diffusion loss and alignment loss.
- Evidence:
  - diffusion loss forward samples internal noise when `noise=None`: `diffusion/gaussian_diffusion.py:729`
  - training call does not pass noise: `train_2.py:543`
  - alignment separately samples new noise and new `x_t`: `train_2.py:556`
- Risk: objective coupling is inconsistent; baseline can be unfairly weakened/noisy.

4. Archived t-aware launcher is interface-incompatible with current trainer.
- Evidence:
  - archived script passes unsupported args (`--repa-diff-schedule`, `--repa-diff-threshold`, `--resume`):
    `scripts_archive_20260410/run_dit_xl_repa_taware_b24_160k.sh:271`, `:273`, `:215`
  - current parser lacks these args: `train_2.py:684`
- Risk: archived recipe is not reproducible as committed.

5. Active SiT launcher has broken filename contract.
- Evidence:
  - checks nonexistent `train_sitxl_repa_dinov2_400k` (without `.py`): `run_sitxl_repa_dinov2_400k.sh:56`
  - executes nonexistent `train_sit_repa_dinov2.py`: `run_sitxl_repa_dinov2_400k.sh:118`
  - actual file is `train_sitxl_repa_dinov2_400k.py`
- Risk: launcher cannot run as-is.

6. `train_sasa.py` keeps a meaningful CLI flag that has no runtime effect.
- Evidence:
  - deprecated but accepted flag: `train_sasa.py:649`
  - runtime says diffusion weighting disabled: `train_sasa.py:391`
  - loss path has no diff-timestep weighting: `train_sasa.py:486`
- Risk: ablation tables varying this flag are invalid.

7. S3A resume lacks semantic-args compatibility checks.
- Evidence:
  - saves args: `train_s3a_multisource_dinov2.py:986`
  - loads states but does not compare current CLI vs checkpoint CLI: `train_s3a_multisource_dinov2.py:999`
- Risk: silent resume into semantically different experiment settings.

### Medium
8. Pre-S3A scripts lack audit-grade checkpoint/restart support.
- Evidence:
  - no `format_version/train_steps/batches_seen/rng_state/manifest` in:
    `train_2.py:641`, `train_sasa.py:194`, `train_sasa_dinov2.py:402`, `train_sitxl_repa_dinov2_400k.py:335`
  - S3A has stronger stack at `train_s3a_multisource_dinov2.py:968+`
- Risk: weak forensic reproducibility.

9. S3A mechanism-vs-claim mismatch risk (not runtime bug, claim bug).
- Evidence:
  - optional EMA source uses trainable `ema_adapters`: `train_s3a_multisource_dinov2.py:611`, `:1200`
  - gate signal is router self-confidence threshold, not external utility metric: `train_s3a_multisource_dinov2.py:642`, `:867`
  - layer weights are static modes/custom constants: `train_s3a_multisource_dinov2.py:262`, `:1161`
- Risk: overclaiming “principled reliability / dynamic layer selection / fully fixed teacher target”.

10. Cross-architecture comparison is currently confounded.
- Evidence:
  - teacher family differs (`DINOv3` vs `DINOv2`), loss family differs (cosine-only vs holistic), parameter counts differ substantially.
- Measured param gap (DiT-XL/2 defaults):
  - REPA projector (SASA-DINOv2): `2,213,760`
  - S3A head (`use_ema_source=True`): `35,972,610`
  - S3A head (`use_ema_source=False`): `18,200,833`
- Risk: full-vs-full gains are not causal evidence for routing/gating.

## 3) Architecture rationality (current code reality)

### t-aware (`train_2.py`)
- Reality: single-layer, single-source, step-decay cosine alignment baseline.
- Score (consensus):
  - soundness: 2/10
  - novelty: 1/10
  - code reliability: 2/10
- Defensible claim: “minimal REPA-like regularizer baseline”.
- Not defensible: “timestep-aware / stage-aware” claims.

### SASA (`train_sasa.py` + `train_sasa_dinov2.py`)
- Reality:
  - v1: same-forward hook optimization + single-layer cosine + step schedule.
  - dinov2: adds true diff-timestep weighting, still single-layer.
- Score (consensus):
  - soundness: 4-5/10
  - novelty: 2/10
  - code reliability: 4/10
- Defensible claim: “hook-based single-layer REPA-style implementation variant; dinov2 version supports timestep-weighted alignment”.
- Not defensible: “new architecture beyond REPA class”.

### S3A (`train_s3a_multisource_dinov2.py`)
- Reality: multi-layer + optional EMA source + router + holistic losses + stronger checkpoint/resume stack.
- Score (consensus):
  - soundness: 5/10
  - novelty: 5/10
  - code reliability: 8/10
- Defensible claim: “heuristic multi-component auxiliary alignment branch with reproducibility-hardening”.
- Not defensible today: “principled reliability estimator”, “fully causal proof of routing superiority” without matched controls.

## 4) Claim boundaries (for paper safety)

Allowed now:
- Bundled S3A branch can improve quality under fixed teacher + fixed eval protocol.
- SASA-dinov2 can test whether diff-timestep weighting matters in single-layer setting.

Not allowed now:
- `t-aware`/`SASA` as true timestep-aware methods.
- S3A routing/gate/layer causal claims from full-method one-shot wins.
- Fairness claims across scripts with different teacher stack/eval stack/capacity.

## 5) Minimal high-ROI evidence package (from Reviewer C)
1. `E0`: SASA-dinov2, single layer, `uniform`
2. `E1`: E0 + `cosine`
3. `E1b`: E0 + `linear_low`
4. `E2`: degenerate S3A (1 layer, DINO-only, gate off, feat-only, uniform)
5. `E3`: E2 with late 4 layers
6. `E4`: E2 with spread 4 layers
7. `E5`: E4 + EMA source on, gate off
8. `E6`: E5 + gate on
9. `E7`: E6 + attn/spatial losses + cosine schedule

Needed additional control for routing claim:
- add `routing_mode={dynamic,fixed_half}` and compare under identical settings.

## 6) Priority patch list

P0 (must fix before new claim round):
1. Backport strict DINOv2 loading from S3A into SASA-DINOv2 and SiT trainer.
2. Fix broken SiT launcher filenames and target script path.
3. Remove/forbid no-op `--repa-diff-schedule` in `train_sasa.py`.
4. Add semantic-args compatibility checks in S3A resume.

P1:
5. Add resume/checksum/RNG checkpoint stack to non-S3A scripts.
6. Retire or rewrite archived t-aware launcher to match current parser.
7. Add static-fusion control in S3A for routing causality.

## 7) Overall verdict
- `t-aware`: baseline only, not paper-grade method.
- `SASA`: engineering variant, novelty ceiling low.
- `S3A`: only viable mainline, but currently suitable for conservative claims unless causal ablations close.
- Current submission posture: **journal-conservative / evidence-building**, not strong NeurIPS claim yet.
