# FID 低表现根因调查报告（2026-04-13）

## 1. 结论先行

结论：**`projector` 不是当前 FID 低表现的主因**。  
从代码证据看，`projector` 在各训练脚本中结构一致、梯度链路正常、参数也进入优化器；更可能拉低 FID 的是：

1. **评估口径与评估实现不一致**（`pytorch_fid` vs ADM evaluator；样本数口径、目录复用等）。
2. **训练/采样配置偏离基线**（例如 `batch=4`、`repa_lambda=1.0`、80k 步等）。
3. **与原 REPA 关键设定存在系统性漂移**（目标函数、teacher、层位、权重调度、采样协议）。
4. **采样实现细节风险**（CFG 仅作用 3 通道、VAE 假设固定 0.18215 等）。

---

## 2. 调查范围与方法

本次分析同时覆盖以下维度并交叉验证：

- 评估链路：`run_eval_fid_offline.sh`、`eval_fid_ddp.py`、`run_fid_eval.py`、`evaluator.py`
- 训练链路：`train_2.py`、`train_sasa.py`、`train_sasa_dinov2.py`、`train_sitxl_repa_dinov2_400k.py`
- 配置与脚本：`run_*.sh`、`scripts_archive_20260410/*`
- 与原 REPA 对照：`project/REPA/*`

---

## 3. `projector` 是否主因

### 3.1 证据：`projector` 本身“可工作”

- 结构在主训练脚本中一致：**REPA-faithful** `Linear -> SiLU -> Linear -> SiLU -> Linear`（默认 `projector_dim=2048`）  
  - `train_2.py:121`
  - `train_sasa.py:103`
  - `train_sasa_dinov2.py:396`
  - `train_sitxl_repa_dinov2_400k.py:333`
- 前向与反向链路正常，且参数明确加入优化器  
  - 前向调用：`train_sasa_dinov2.py:735`、`train_sasa.py:466`、`train_2.py:564`
  - 优化器纳入：`train_sasa_dinov2.py:576`、`train_sasa.py:337`、`train_2.py:472`

### 3.2 判断

`projector` 当然会影响上限，但当前“FID 很低”的主要解释更像是**系统性实验链路问题**，不是单点 `projector` 架构问题。

---

## 4. 高优先级根因（按影响排序）

## 4.1 评估口径与实现风险（最高优先）

1. **评估口径混用，FID 不可直接比较**  
   - `eval_fid_ddp.py` 使用 `pytorch_fid`（目录图像）  
   - `run_fid_eval.py`/`evaluator.py` 走 ADM 风格（npz）  
   - 证据：`eval_fid_ddp.py:170`、`run_fid_eval.py:133`、`run_fid_eval.py:295`

2. **请求样本数与实际评估样本数不一致**（向上取整后全目录参与 FID）  
   - 证据：`eval_fid_ddp.py:121`、`eval_fid_ddp.py:125`、`eval_fid_ddp.py:170`

3. **样本目录复用可能污染 FID**（未 overwrite 时历史样本可能混入）  
   - 证据：`eval_fid_ddp.py:105`、`run_eval_fid_offline.sh:67`

4. **`run_fid_eval.py` 脚本存在路径/行为风险**（`generate.py`、`evaluations/evaluator.py` 不在当前仓库）  
   - 证据：`run_fid_eval.py:63`、`run_fid_eval.py:137`

## 4.2 训练配置偏离基线

1. **默认超参明显偏离**（某些脚本默认 `batch=4`、`repa_lambda=1.0`）  
   - 证据：`run_dit_xl_repa_linear_80k.sh:17`、`run_dit_xl_repa_linear_80k.sh:23`  
   - 对照默认：`train_2.py:693`（batch 默认 256）、`train_2.py:731`（lambda 默认 0.1）

2. **80k 步 + 小 batch + 强对齐权重** 的组合本身就很容易把 FID 拉差。

## 4.3 与原 REPA 的系统漂移

1. **训练范式漂移**：REPA 原链路是 SiT + SILoss；当前多为 DiT + DDPM training loss  
   - 对照：`project/REPA/loss.py:52`、`project/REPA/models/sit.py:265` vs `train_2.py:425`、`train_2.py:543`

2. **对齐分支关键设定漂移**：
   - 原 REPA：3层 projector + `encoder_depth=8` + `proj_coeff=0.5`（常量）
   - 当前（2026-04-14后）：projector 已回到 3层 SiLU；主要漂移变为 token 层位、teacher、schedule 与训练范式差异
   - 对照：`project/REPA/models/sit.py:16`、`project/REPA/train.py:456`、`project/REPA/train.py:460` vs `train_2.py:121`、`train_2.py:731`、`train_sasa_dinov2.py:532`

3. **teacher 漂移**：部分脚本用 DINOv3，原 REPA 主线是 DINOv2

## 4.4 采样实现细节风险

1. **CFG 只作用前 3 通道，不是全 latent 通道**  
   - 证据：`models.py:262`、`models_2.py:297`、`model_sasa.py:301`

2. **VAE 缩放固定 0.18215**（若 VAE 不匹配会影响 FID）  
   - 证据：编码/解码路径均硬编码该比例（如 `train_2.py:538`、`train_2.py:327`）

---

## 5. 脚本层面直接问题（会影响你是否“测对了”）

1. `run_sitxl_repa_dinov2_400k.sh` 有明显文件名错误，默认无法按预期启动  
   - 证据：`run_sitxl_repa_dinov2_400k.sh:56`、`run_sitxl_repa_dinov2_400k.sh:118`

2. `scripts_archive_20260410` 下多脚本与当前根目录结构不一致，容易误跑旧链路。

---

## 6. 推荐的最小化排查实验（先排“评估假象”再排“训练真问题”）

## Phase A：先统一评估口径（1 天内）

1. 固定同一个 ckpt、同一批样本，**只用一套 evaluator**（建议先统一 `eval_fid_ddp.py + pytorch_fid`）。
2. 强制 `--overwrite`，避免样本复用。
3. 记录“请求样本数”和“实际文件数”，确保一致。
4. 统一最终报告样本数为 **50k**（不要 5k）。

## Phase B：恢复可信训练基线（1-2 天）

1. 先用保守基线：`global_batch_size=256`、`repa_lambda=0.1`、`max_steps>=80k`。
2. 避免混用归档脚本，统一使用根目录脚本。
3. 固定 `num_sampling_steps=250`、`cfg_scale=1.5` 做同口径对比。

## Phase C：做“是否 projector 主因”的定量验证（2-3 天）

固定其余超参，仅做：

- Exp-1: REPA-faithful projector（3层，SiLU，`projector_dim=2048`）
- Exp-2: 在同口径下替换为轻量变体（如 2层 GELU）做反向对照

如果 Exp-2 相比 Exp-1 只提升小幅（例如 <10-15% 相对改善），则进一步证明主因不在 projector。

## Phase D：再做高价值 ablation（2-3 天）

优先顺序：

1. `repa_lambda`（0.05 / 0.1 / 0.2）
2. hook 层位（中层 vs 最后一层）
3. diff timestep weighting（开/关）
4. CFG 全通道 vs 3通道

---

## 7. 建议的统一基线参数（用于后续对比）

- 训练：`image_size=256`, `global_batch_size=256`, `repa_lambda=0.1`, `repa_schedule_steps=40000`
- 采样：`num_sampling_steps=250`, `cfg_scale=1.5`
- 评估：`fid_num_samples=50000`, 固定同一个 reference（目录或 stats 二选一，不混用）
- 结果汇报：同时记录 `ckpt step`、`实际样本数`、`评估脚本 commit/路径`

---

## 8. 最终回答你的核心问题

你问“FID 很低是不是主要就是 projector？”  
**当前代码证据支持的答案：不是。**  
`projector` 是影响因素之一，但眼下最需要先修的是**评估链路一致性 + 训练配置基线 + 与 REPA 主链路漂移**，这些项的影响量级通常比 `projector` 更大。
