# Research Review: Latest S3A (0b466ad) Dual-Source Failure Audit

## Scope
- User request: `$research-review` 审查最新 `s3a` 版本是否已解决双源失效。
- Target code revision:
  - `0b466ad` (`config: self_warmup=25k, dino_alpha_floor_steps=25k`)
  - plus immediate predecessor `0316bd5` (`prevent router softmax saturation during self_warmup`)
- Main files reviewed:
  - `/home/liuchunfa/2026qjx/2026test/train_s3a_multisource_dinov2.py`
  - `/home/liuchunfa/2026qjx/2026test/run_s3a_multisource_dinov2.sh`

## Reviewer Setup
- External reviewer agent: `019d95f6-d7d2-75d1-aa64-0b7adb25755b`
- Model: `gpt-5.4`
- Reasoning: `xhigh`

## Round-1 Verdict
- Verdict: `NOT_SOLVED`
- Confidence: `~0.80` (medium-high)

Core judgment:
1. `0316bd5` fixed a real warmup bug (masked-phase router gradient + warmup-exit router-head reset).
2. But this does not prove the historical post-unlock self-dominance failure is solved.
3. `0b466ad` mostly extends warmup/floor-protection horizon to `25k`; it can postpone visibility of failure in short canaries rather than demonstrate resolution.

## Key Technical Findings
1. Real mechanism improvement was made:
- warmup detachment of router output when source1 is masked
- one-time router output layer reset at warmup end

2. Remaining mechanism risk is still the same failure family:
- historical failure is not "self never opens" but "self opens and dominates"
- old evidence showed floor-hugging DINO + negative synergy

3. Current defaults changed evaluation regime:
- `s3a_self_warmup_steps=25000`
- `s3a_dino_alpha_floor_steps=25000`
- Therefore, `10k` canary cannot validate post-unlock dual-source collaboration anymore.

4. Control logic gap remains:
- collaboration indicators (`dual_alive`, `synergy_margin`) are mainly diagnostics
- strongest mitigation is still keyed by harder collapse/starvation predicates

5. Evidence gap is decisive:
- No fresh exact-commit run (`0316bd5/0b466ad`) beyond warmup unlock (`>25k`) was provided in this review.

## Converged Conclusion
- For the question "latest version solved dual-source failure?": **No, not yet closed**.
- Current status should be interpreted as: **partial mechanism progress + insufficient evidence + unresolved post-unlock risk**.

## Minimal Evidence Package Required to Close
1. Fresh no-resume run on exact `0b466ad` to at least `40k`.
2. Same setup second seed.
3. Resume-across-unlock run (`24k` resume to `40k`).
4. Optional causal ablation: old warmup/floor (`5k/8k`) contrast.

Required artifacts per run:
- `resolved_args.json`
- `metrics.jsonl`
- trainer log
- git revision

## Strict Pass/Fail Gate (reviewer suggested)
- Define `eval_start = s3a_self_warmup_steps + 500` (current defaults => `25500`).
- Require at least `10k` evaluated steps after `eval_start`.

Per-run PASS requires all:
1. self reopen within 500 post-warmup steps (`gate_self_state > 0.5`)
2. `dual_alive=1` in >=90% eval windows and all final 5 windows
3. median `a_dino_above_floor >= 0.02`
4. `a_dino_above_floor > 0.005` for final 5 windows
5. `alpha_dino_min_layer_above_floor > 0.005` for final 5 windows
6. median `synergy_margin > 0.01`
7. `synergy_margin > 0` in final 5 windows
8. no `dino_starved_alarm`, no `collapse_alarm`, no `mitigate` after `eval_start`

Immediate FAIL if any:
1. no self reopen within 500 post-warmup steps
2. `dual_alive=0` for 3 consecutive eval windows
3. `a_dino_above_floor <= 0.005` for 3 consecutive eval windows
4. `synergy_margin <= 0` for 2 consecutive eval windows
5. any starvation/collapse/mitigation alarm after `eval_start`

