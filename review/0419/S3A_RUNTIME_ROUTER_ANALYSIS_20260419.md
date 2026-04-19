# S3A v3 续训状态分析（2026-04-19）

## 1. 背景与结论
- 运行任务：`0045000.pt -> 400000 step` 的续训任务。
- 当前进度：已推进至 `step=85000`（最新日志时间约 `2026-04-19 14:14 CST`）。
- 核心结论：
  - `synergy_margin` 当前确实稳定在约 `-0.112` 附近，短窗口波动很小。
  - 但这**不能直接推出 router 权重没变**。
  - 证据显示：router 参数与 raw 输出在持续变化；只是部署后的 `alpha_dino/alpha_self` 被 floor + source mask + 层间平均机制约束，表现为近似常数。

## 2. 运行健康状态
- GPU 使用：4 卡持续高负载（接近 `99%~100%`），显存约 `45.8/49.1 GB`。
- 训练速度：`steps_per_sec` 约 `0.64`（最近窗口稳定）。
- 稳定性告警：
  - `dino_starved = 0`
  - `collapse_alarm = 0`
  - `dual_source_alive = 1`

## 3. synergy_margin 与路由指标（最近窗口）
数据源：`metrics.jsonl`（本次 run）

### 最近 10 点（step 80500 -> 85000）
- `dual_synergy_margin`: mean `-0.111987`, std `0.000268`, range `[-0.112370, -0.111531]`
- `alpha_dino`: mean `0.287500`, std `0.000000`
- `alpha_self`: mean `0.712500`, std `0.000000`
- `raw_alpha_dino`: mean `0.009197`, std `0.000084`, delta `+0.000252`
- `router_entropy_norm`: mean `0.072172`, std `0.000421`, delta `+0.001301`
- `router_policy_kl`: mean `0.023021`, std `0.000030`, delta `-0.000087`

### 最近 80 点（step 45500 -> 85000）
- `dual_synergy_margin`: delta `+0.007534`（从更负到较少负）
- `raw_alpha_dino`: delta `+0.002244`
- `router_entropy_norm`: delta `+0.013029`
- `router_policy_kl`: delta `-0.000914`

## 4. 关键证据：router 参数是否在更新
比较 checkpoint：`0060000.pt` vs `0080000.pt`

- `s3a_head_state` 中 router 参数张量数：`13`
- 有变化张量：`13/13`
- router 全局参数变化：
  - `router_global_l2_diff = 4.7674194782`
  - 相对变化（对 0060000 的 L2）`= 0.1145533411`（约 11.46%）
  - 最大绝对差：`0.0822123513`（`router.head.1.weight`）

=> 说明 router 参数在持续更新，不是“完全不动”。

## 5. 为什么 alpha 看起来几乎不变
结合代码逻辑：
- source0 最小占比 floor：`source0_min_alpha_at_step(...)`（`s3a_protect_source0_min_alpha=0.05`）
- source mask 与 selective gate 对有效 source 做硬约束。
- 最后记录的 `alpha_dino` 是跨层平均值；本配置下层间贡献可形成稳定均值。

相关代码位置（文件：`train_s3a_multisource_dinov2.py`）：
- source0 floor：约 `469-480`
- source mask / warmup / gate：约 `986-1030`
- raw_alpha 与 alpha 构建：约 `1188-1286`
- synergy_margin 定义：约 `2917-2919`

## 6. 回答用户问题
问题：`synergy_margin=-0.1118` 长期变化不大，是否可说明 router 权重没变？

结论：**不能**。
- 可以说明：当前“融合优于单源”的度量尚未转正，且短期稳定。
- 不能说明：router 权重不更新。实证相反，router 权重与 raw 输出均在变化。

