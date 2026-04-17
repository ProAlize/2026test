# S3A Training Note (bz=128, 400k) - 2026-04-17

## 0) 中文结论摘要
- 本次 `bz=128 / 400k` 训练在工程层面稳定运行（多卡负载正常、无 NaN/崩溃）。
- 但机制层面未建立稳定“双源协同”，主要呈现：
  - 短暂双源打开 -> 自源主导（DINO 降至 floor）-> 触发缓解 -> 回到 DINO 单源。
- 因此该 run 暂不支持“已解决双源协同问题”的结论，当前更接近“单源切换 + 报警重置”的控制行为。

## 1) Experiment identity
- Branch: `s3a`
- Code revision used by this run: `0b466adaa5d325c571ccef830e0f384ede7c6475`
- Launch date: `2026-04-16 19:31:17` (Asia/Shanghai)
- Run dir:
  - `/data/liuchunfa/2026qjx/s3a_runs/dit_xl_s3a_dinov2_400k_bz128_20260416/DiT-XL-2-seed0-20260416-193137-945ecf-s3a-dinov2-lam0.5-trainpiecewise_cosine-diffcosine`

## 2) Key training config (resolved)
- `global_batch_size=128`
- `max_steps=400000`
- `log_every=500`
- `ckpt_every=20000`
- `s3a_lambda=0.5`
- `s3a_train_schedule=piecewise_cosine`
- `s3a_self_warmup_steps=25000`
- `s3a_dino_alpha_floor=0.1`
- `s3a_dino_alpha_floor_steps=25000`
- `s3a_protect_source0_min_alpha=0.05`
- `s3a_gate_utility_off_threshold=0.002`
- `s3a_gate_utility_on_threshold=0.005`
- `s3a_gate_patience=500`
- `s3a_gate_reopen_patience=200`
- `s3a_probe_every=10`
- `s3a_collapse_auto_mitigate=true`
- `s3a_collapse_mitigate_windows=3`
- `s3a_collapse_mitigate_cooldown_windows=6`
- `s3a_router_policy_kl_lambda=0.1`

## 3) Current status (as of 2026-04-17 11:41)
- Training is running normally on 4 GPUs (high utilization, no NaN/Inf crash).
- Latest observed step: `46500`.
- Latest metrics:
  - `loss=0.4037`
  - `loss_diff=0.1545`
  - `loss_align=0.4983`
  - `alpha_dino=1.0000`
  - `alpha_self≈0.0`
  - `alpha_dino_above_floor=0.9500`
  - `dual_source_alive=0`
  - `dual_synergy_margin=-0.4868`

## 4) Problem statement
The run is compute-stable but **does not establish sustained dual-source collaboration**.
Observed mode is a loop:
- brief dual activation,
- then self-dominant collapse (DINO to floor),
- then mitigation forces return to DINO-only,
- then repeat.

This means current evidence does **not** support a “dual-source effective collaboration” claim.

## 5) Evidence timeline (post unlock)
Unlock point is `step=25000`.

Representative events:
- `26000`: dual briefly opens (`a_dino=0.136`, `a_self=0.864`, `dual_alive=1`, but `synergy<0`).
- `26500~27500`: self-dominant collapse (`a_dino=0.05`, `a_self=0.95`, `a_dino_above_floor=0`), starvation alarm appears.
- `27500`: `dino_starved_alarm=1`, mitigation triggered.
- `28000+`: forced return to DINO-only (`a_dino=1.0`, `a_self=0.0`).
- Similar pattern repeats around `37000~39000`.

Aggregated post-unlock (`step>=25000`, 44 windows):
- `dino_only`: 36 windows (`81.82%`)
- `self_only`: 6 windows (`13.64%`)
- `dual_alive`: 2 windows (`4.55%`)
- `avg_dual_synergy_margin=-0.3993`
- `max_dino_starved_alarm=1`

## 6) Why this happens (engineering diagnosis)
The utility/optimization geometry currently favors single-source domination instead of cooperation:
- In post-unlock windows, `loss_self_only` is consistently far below `loss_dino_only`.
- Router/gate therefore repeatedly shifts to self-heavy mode.
- Anti-collapse mitigation can recover from catastrophic starvation, but recovery lands at DINO-only and does not maintain dual-source synergy.

So the controller behaves as a **single-source selector with alarm-based reset**, not a stable dual-source coordinator.

## 7) Recommended fix direction
Priority should be controller-policy fixes, not longer training time alone:
1. Add reopen constraints for self-source:
- require `alpha_dino_above_floor` and `dual_synergy_margin` both pass minimum thresholds for consecutive windows before full reopen.
2. Increase mitigation hysteresis:
- longer hold and cooldown to prevent immediate relapse.
3. Re-tune reopen thresholds:
- raise self reopen utility threshold and reduce optimistic off-probe floor to avoid premature self re-entry.
4. Keep strict acceptance gate:
- require sustained `dual_alive=1` and `synergy_margin>0` over consecutive post-unlock windows.

## 8) Attached logs
- Raw training log:
  - `reports/s3a_bz128_400k_20260417/logs/s3a_bz128_400k_v4.log`
- 20-minute monitor analysis log:
  - `reports/s3a_bz128_400k_20260417/logs/s3a_bz128_400k_v4_20m_analysis.log`
