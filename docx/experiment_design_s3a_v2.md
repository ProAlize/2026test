# S3A v2 实验设计文档

## 1. 方法概述

S3A v2 = REPA（外部对齐） + SRA 式自对齐（跨层跨时间步） + 动态融合 + 结构保真 loss

### 核心改动
- **Self-source 跨层**: student layer m → EMA teacher layer n (n = m + offset, 默认 offset=14)
- **Self-source 跨时间步**: EMA teacher 处理更低噪声 t' = max(0, t-k), k ~ Uniform(0, 200)
- **相同 noise 向量**: 保持扩散轨迹一致

### 层映射 (DiT-XL/2, depth=28, offset=14)
| Student Layer | EMA Teacher Layer | 语义 |
|--------------|------------------|------|
| 6 (25%) | 20 (75%) | 浅→深 |
| 13 (50%) | 27 (100%) | 中→最深 |
| 20 (75%) | 27 (100%) | 深→最深 (clamp) |
| 27 (100%) | = 27 (no offset) | 纯 DINO, 无 self-source |

## 2. 主实验 (400k steps, DiT-XL/2, ImageNet 256×256)

### E0: DINO-only 基线
```bash
S3A_ALLOW_UNSAFE_ZERO_WARMUP=1 S3A_SELF_WARMUP_STEPS=0 \
  ./run_s3a_multisource_dinov2.sh --no-s3a-use-ema-source
```
验证: S3A holistic loss + multi-layer + piecewise schedule 的独立贡献

### E1: S3A v2 Full (推荐配置)
```bash
./run_s3a_multisource_dinov2.sh
# 默认: offset=14, k_max=200, lambda=0.5, warmup=25k
```
验证: 完整双源协同（DINO + SRA 式 self-source）

### E2: REPA 对标
```bash
S3A_LAMBDA=0.5 S3A_LAYER_INDICES=6 \
S3A_FEAT_WEIGHT=1.0 S3A_ATTN_WEIGHT=0.0 S3A_SPATIAL_WEIGHT=0.0 \
S3A_TRAIN_SCHEDULE=linear_decay S3A_SCHEDULE_STEPS=40000 \
S3A_ALLOW_UNSAFE_ZERO_WARMUP=1 S3A_SELF_WARMUP_STEPS=0 \
  ./run_s3a_multisource_dinov2.sh --no-s3a-use-ema-source
```
验证: 框架内复现 REPA 行为

## 3. 消融实验 (80k steps 足够区分趋势)

| ID | 改动 | 对比 | 验证内容 |
|----|------|------|---------|
| A1 | self_layer_offset=0, timestep_offset_max=0 | E1 | 旧设计退化 (应看到坍塌) |
| A2 | self_layer_offset=14, timestep_offset_max=0 | E1 | 仅跨层的贡献 |
| A3 | self_layer_offset=0, timestep_offset_max=200 | E1 | 仅跨时间步的贡献 |
| A4 | self_layer_offset=7 | E1 | 偏移距离 |
| A5 | timestep_offset_max=100 / 300 | E1 | 时间偏移范围 |
| A6 | feat_weight=1, attn_weight=0, spatial_weight=0 | E1 | holistic loss 贡献 |
| A7 | layer_indices=6 (单层) | E1 | 多层注入贡献 |

## 4. 对比实验

| ID | 方法 | 步数 | 说明 |
|----|------|------|------|
| C1 | DiT-XL/2 vanilla (无对齐) | 400k | 绝对基线 |
| C2 | DiT-XL/2 + REPA (E2) | 400k | 外部对齐基线 |
| C3 | DiT-XL/2 + S3A v2 (E1) | 400k | Our method |

## 5. 评估指标

- FID-50K (ADM eval suite)
- sFID
- IS (Inception Score)
- Precision / Recall
- 中间监控: alpha_dino, alpha_self, dual_alive, synergy_margin, loss_diff, loss_align

## 6. 时间规划 (→ August 3, 2026 ARR deadline)

| 周 | 任务 |
|----|------|
| W1 (Apr 17-23) | 代码实现完成, 短测验证 |
| W2-3 (Apr 24 - May 7) | E0/E1/E2 主实验 |
| W4-5 (May 8-21) | A1-A7 消融 |
| W6-7 (May 22 - Jun 4) | C1-C3 对比 + FID 评估 |
| W8-9 (Jun 5-18) | 论文初稿 |
| W10-11 (Jun 19 - Jul 2) | 补充实验 + 可视化 |
| W12-13 (Jul 3-16) | 论文修订 + 内部 review |
| W14-15 (Jul 17 - Aug 2) | 最终修改 + 提交 |

## 7. 预期论文定位

**S3A 是第一个将外部表征对齐（REPA 路线）和自表征对齐（SRA 路线）统一到可学习动态融合框架中的方法。**

| vs REPA | vs SRA | vs DUPA |
|---------|--------|---------|
| +自对齐 | +外部 DINO | +不需 DDT |
| +holistic loss | +动态 routing | +multi-source |
| +multi-layer | +推理零开销 | +selective gate |

## 8. 关键超参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| s3a_lambda | 0.5 | 对齐预算匹配 REPA |
| s3a_self_layer_offset | 14 | SRA 实验最优 |
| s3a_self_timestep_offset_max | 200 | SRA: k_max/T ∈ [0.1, 0.2] |
| s3a_self_warmup_steps | 25000 | adapter 成熟后开 self-source |
| s3a_train_schedule | piecewise_cosine | 0-100k constant, 100k-300k decay, 300k-400k off |
| s3a_schedule_steps | 300000 | 衰减结束点 |
| s3a_schedule_warmup_steps | 100000 | constant 阶段长度 |
