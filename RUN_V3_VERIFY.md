# S3A v3 验证实验运行指令

## 目标
验证 v3 跨层跨时间步 self-source 是否打破了 v1 的恒等退化。

## 前置条件
```bash
cd /path/to/2026test
git pull origin s3a   # 确保是 commit 9dcf858+
git log --oneline -1  # 应显示: feat: S3A v2 — cross-layer cross-timestep...
```

## 实验 1: 短验证 (30k steps, 约 10h on 4×GPU)

关键：25k 步后观察 loss_self_only 和 dual_alive。

```bash
cd /path/to/2026test

MAX_STEPS=30000 \
CKPT_EVERY=10000 \
LOG_EVERY=500 \
GLOBAL_BATCH_SIZE=256 \
./run_s3a_multisource_dinov2.sh
```

默认参数 (无需手动设置):
- λ = 0.5
- schedule = piecewise_cosine (0-100k constant, 100k-300k decay)
- self_warmup = 25k
- self_layer_offset = 14 (student 6→EMA 20, 13→27, 20→27, 27→纯DINO)
- self_timestep_offset_max = 200

## 实验 2: 如果实验 1 成功, 跑完整 400k

```bash
MAX_STEPS=400000 \
CKPT_EVERY=20000 \
LOG_EVERY=500 \
GLOBAL_BATCH_SIZE=256 \
./run_s3a_multisource_dinov2.sh
```

## 实验 3: DINO-only 对照 (无 self-source)

```bash
MAX_STEPS=400000 \
CKPT_EVERY=20000 \
LOG_EVERY=500 \
S3A_ALLOW_UNSAFE_ZERO_WARMUP=1 \
S3A_SELF_WARMUP_STEPS=0 \
./run_s3a_multisource_dinov2.sh --no-s3a-use-ema-source
```

## 判定标准 (step 25000+ 之后)

### ✅ v3 成功 (恒等退化被打破):
- `loss_self_only` ≈ 0.3 ~ 1.0 (vs v1 的 0.03)
- `dual_alive` > 50% 的 log windows
- `synergy_margin` > 0 (融合比单源好)
- `a_dino` 和 `a_self` 都 > 0.1

### ❌ v3 失败 (仍然退化):
- `loss_self_only` < 0.1
- `a_self` > 0.9 且 `a_dino` 在 floor
- `dual_alive` < 10%
- 出现 dino_starved_alarm

如果失败 → 转实验 3 (DINO-only), 放弃 self-source

## 监控命令

实时查看训练 log:
```bash
tail -f /path/to/results/*/log.txt | grep -E "step=|dual_alive|a_dino=|a_self=|Lself="
```

查看关键指标趋势:
```bash
grep "step=" /path/to/results/*/log.txt | awk '{for(i=1;i<=NF;i++) if($i ~ /^(a_dino|a_self|Lself|dual_alive|synergy)/) print $i}' | tail -20
```

## 当前版本信息
- Branch: s3a
- Commit: 9dcf858
- 核心改动: cross-layer (offset=14) + cross-timestep (k_max=200) self-source
- 层映射: student [6,13,20,27] → EMA teacher [20,27,27,—]
