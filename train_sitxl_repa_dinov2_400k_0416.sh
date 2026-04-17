#!/usr/bin/env bash
# run_sit_repa_dinov2_400k.sh
# SiT-XL/2 + REPA (DINOv2 ViT-B/14 本地权重) 400k 步训练
# 二维对齐调度：
#   轴1 --repa-train-schedule=three_phase : 0~100k满权重，100k~300k线性衰减，300k后归零
#   轴2 --repa-diff-schedule=cosine       : 干净端(t≈0)权重高
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ── 训练脚本路径（集中管理）─────────────────────────────────────────────
TRAIN_SCRIPT="/mnt/tidal-alsh01/dataset/redaigc/yuantianshuo/2026/2026test/train_sitxl_repa_dinov2_400k_0416.py"

# ── 环境配置 ────────────────────────────────────────────────────────────
ENV_PREFIX="${ENV_PREFIX:-/mnt/tidal-alsh01/dataset/redaigc/yuantianshuo/tool/miniconda3/envs/dit}"
TORCHRUN_BIN="${TORCHRUN_BIN:-$ENV_PREFIX/bin/torchrun}"

# ── 数据 / 模型路径 ─────────────────────────────────────────────────────
DATA_PATH="/data/temp/ILSVRC/Data/CLS-LOC/train"
RESULTS_DIR="${RESULTS_DIR:-/mnt/tidal-alsh01/dataset/redaigc/yuantianshuo/2026/results/sit_xl_repa_dinov2_400k_0416}"
VAE_MODEL_DIR="${VAE_MODEL_DIR:-/mnt/tidal-alsh01/dataset/redaigc/yuantianshuo/2026/models/DiT-XL-2-256/vae}"

# ── DINOv2 路径 ─────────────────────────────────────────────────────────
DINOV2_REPO_DIR="${DINOV2_REPO_DIR:-/mnt/tidal-alsh01/dataset/redaigc/yuantianshuo/tmp/dinov2}"
DINOV2_WEIGHT_PATH="${DINOV2_WEIGHT_PATH:-/mnt/tidal-alsh01/dataset/redaigc/yuantianshuo/tmp/dinov2_weights/dinov2_vitb14_pretrain.pth}"

# ── GPU / 训练超参 ──────────────────────────────────────────────────────
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
NPROC_PER_NODE="${NPROC_PER_NODE:-4}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-256}"
NUM_WORKERS="${NUM_WORKERS:-4}"
EPOCHS="${EPOCHS:-1400}"
MAX_STEPS="${MAX_STEPS:-400000}"
CKPT_EVERY="${CKPT_EVERY:-10000}"
LOG_EVERY="${LOG_EVERY:-500}"
LR="${LR:-1e-4}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.0}"
MAX_GRAD_NORM="${MAX_GRAD_NORM:-1.0}"
MIXED_PRECISION="${MIXED_PRECISION:-fp16}"

# ── REPA 超参 ───────────────────────────────────────────────────────────
REPA_LAMBDA="${REPA_LAMBDA:-0.5}"
REPA_ENCODER_DEPTH="${REPA_ENCODER_DEPTH:-8}"
REPA_HIDDEN_DIM="${REPA_HIDDEN_DIM:-2048}"
Z_DIM="${Z_DIM:-768}"

# 投影器类型：repa（3层MLP，内置）或 irepa（Conv2d k=3，外置）
PROJECTOR_TYPE="${PROJECTOR_TYPE:-repa}"

# 轴1：训练步数权重调度（three_phase 推荐配置）
#   [0,               REPA_WARMUP_STEPS)  : w = 1.0  （满权重对齐）
#   [REPA_WARMUP_STEPS, REPA_SCHEDULE_STEPS) : w 线性衰减 1→0
#   [REPA_SCHEDULE_STEPS, ∞)              : w = 0.0  （纯生成训练）
#
# 注意：Python 脚本用 --repa-warmup-steps 对应 Shell 原来的 REPA_SCHEDULE_START
#       Python 脚本用 --repa-schedule-steps 表示衰减截止的【绝对步数】
REPA_TRAIN_SCHEDULE="${REPA_TRAIN_SCHEDULE:-three_phase}"
REPA_WARMUP_STEPS="${REPA_WARMUP_STEPS:-100000}"    # 开始衰减
REPA_SCHEDULE_STEPS="${REPA_SCHEDULE_STEPS:-400000}" # 停止衰减

# 轴2：扩散时间步权重调度
REPA_DIFF_SCHEDULE="${REPA_DIFF_SCHEDULE:-cosine}"

# ── 衰减区间信息（仅用于打印，方便核对）────────────────────────────────
# three_phase 衰减区间: [REPA_WARMUP_STEPS, REPA_SCHEDULE_STEPS)
DECAY_START_STEP="$REPA_WARMUP_STEPS"
DECAY_END_STEP="$REPA_SCHEDULE_STEPS"

# ── Sanity checks ────────────────────────────────────────────────────────
if [[ ! -x "$TORCHRUN_BIN" ]]; then
    echo "[ERROR] torchrun not found or not executable: $TORCHRUN_BIN" >&2
    exit 1
fi

for required_path in \
    "$TRAIN_SCRIPT" \
    "$DATA_PATH" \
    "$VAE_MODEL_DIR" \
    "$DINOV2_REPO_DIR" \
    "$DINOV2_REPO_DIR/hubconf.py" \
    "$DINOV2_WEIGHT_PATH"; do
    if [[ ! -e "$required_path" ]]; then
        echo "[ERROR] Required path not found: $required_path" >&2
        exit 1
    fi
done

# three_phase 参数合理性检查
if [[ "$REPA_TRAIN_SCHEDULE" == "three_phase" ]]; then
    if [[ "$REPA_WARMUP_STEPS" -ge "$REPA_SCHEDULE_STEPS" ]]; then
        echo "[ERROR] three_phase: REPA_WARMUP_STEPS ($REPA_WARMUP_STEPS)" \
             ">= REPA_SCHEDULE_STEPS ($REPA_SCHEDULE_STEPS)，衰减区间为空！" >&2
        exit 1
    fi
    if [[ "$REPA_SCHEDULE_STEPS" -gt "$MAX_STEPS" ]]; then
        echo "[WARNING] three_phase: REPA_SCHEDULE_STEPS ($REPA_SCHEDULE_STEPS)" \
             "> MAX_STEPS ($MAX_STEPS)，权重将在训练结束前未能完全归零。" >&2
    fi
fi

# 投影器类型校验
if [[ "$PROJECTOR_TYPE" != "repa" && "$PROJECTOR_TYPE" != "irepa" ]]; then
    echo "[ERROR] PROJECTOR_TYPE must be 'repa' or 'irepa', got: $PROJECTOR_TYPE" >&2
    exit 1
fi

mkdir -p "$RESULTS_DIR"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
LAUNCH_LOG="${LAUNCH_LOG:-$RESULTS_DIR/launch_${TIMESTAMP}.log}"

# ── 启动信息 ─────────────────────────────────────────────────────────────
cd "$ROOT_DIR"
echo "====== SiT-XL/2 + REPA (DINOv2 ViT-B/14) — 400k steps ======"
echo "Train script     : $TRAIN_SCRIPT"
echo "Torchrun         : $TORCHRUN_BIN"
echo "GPUs             : $CUDA_VISIBLE_DEVICES  (nproc=$NPROC_PER_NODE)"
echo "Results dir      : $RESULTS_DIR"
echo "Launch log       : $LAUNCH_LOG"
echo "Max steps        : $MAX_STEPS"
echo "Global batch     : $GLOBAL_BATCH_SIZE"
echo "Ckpt every       : $CKPT_EVERY steps"
echo "Log every        : $LOG_EVERY steps"
echo "LR               : $LR"
echo "Weight decay     : $WEIGHT_DECAY"
echo "Max grad norm    : $MAX_GRAD_NORM"
echo "Mixed precision  : $MIXED_PRECISION"
echo "── 注意：interpolant 已内联（Linear path + v-prediction） ────"
echo "   --path-type / --prediction / --loss-weight 等参数已移除"
echo "── REPA 二维对齐调度 ───────────────────────────────────────────"
echo "Projector type   : $PROJECTOR_TYPE"
echo "REPA lambda      : $REPA_LAMBDA"
echo "Z dim            : $Z_DIM"
echo "Encoder depth    : $REPA_ENCODER_DEPTH"
echo "  hook block idx : $((REPA_ENCODER_DEPTH - 1))"
echo "Projector hidden : $REPA_HIDDEN_DIM"
echo "── 轴1 w_train(step) : $REPA_TRAIN_SCHEDULE ──────────────────"
if [[ "$REPA_TRAIN_SCHEDULE" == "three_phase" ]]; then
    echo "  phase1 [0,                   ${DECAY_START_STEP}) : w = 1.0 (满权重对齐)"
    echo "  phase2 [${DECAY_START_STEP}, ${DECAY_END_STEP})   : w 线性 1.0→0.0"
    echo "  phase3 [${DECAY_END_STEP},   ∞)                   : w = 0.0 (纯生成训练)"
    echo "  --repa-warmup-steps  : $REPA_WARMUP_STEPS"
    echo "  --repa-schedule-steps: $REPA_SCHEDULE_STEPS (截止绝对步数)"
else
    echo "  schedule       : $REPA_TRAIN_SCHEDULE"
    echo "  schedule_steps : $REPA_SCHEDULE_STEPS"
fi
echo "── 轴2 w_diff(t)     : $REPA_DIFF_SCHEDULE ───────────────────"
echo "  干净端(t≈0)权重高，噪声端(t≈1)权重趋近0"
echo "── eff_λ = $REPA_LAMBDA × w_train(step) × w_diff(t) ──────────"
echo "DINOv2 repo      : $DINOV2_REPO_DIR"
echo "DINOv2 weight    : $DINOV2_WEIGHT_PATH"
echo "VAE              : $VAE_MODEL_DIR"
echo "==============================================================="

# ── 启动训练 ─────────────────────────────────────────────────────────────
env CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" \
    "$TORCHRUN_BIN" \
        --standalone \
        --nnodes=1 \
        --nproc_per_node="$NPROC_PER_NODE" \
    "$TRAIN_SCRIPT" \
        --data-path              "$DATA_PATH"              \
        --results-dir            "$RESULTS_DIR"            \
        --model                  SiT-XL/2                  \
        --image-size             256                       \
        --global-batch-size      "$GLOBAL_BATCH_SIZE"      \
        --num-workers            "$NUM_WORKERS"            \
        --epochs                 "$EPOCHS"                 \
        --max-steps              "$MAX_STEPS"              \
        --ckpt-every             "$CKPT_EVERY"             \
        --log-every              "$LOG_EVERY"              \
        --lr                     "$LR"                     \
        --weight-decay           "$WEIGHT_DECAY"           \
        --max-grad-norm          "$MAX_GRAD_NORM"          \
        --mixed-precision        "$MIXED_PRECISION"        \
        --vae-model-dir          "$VAE_MODEL_DIR"          \
        --z-dim                  "$Z_DIM"                  \
        --projector-type         "$PROJECTOR_TYPE"         \
        --repa-lambda            "$REPA_LAMBDA"            \
        --repa-encoder-depth     "$REPA_ENCODER_DEPTH"     \
        --repa-hidden-dim        "$REPA_HIDDEN_DIM"        \
        --repa-train-schedule    "$REPA_TRAIN_SCHEDULE"    \
        --repa-warmup-steps      "$REPA_WARMUP_STEPS"      \
        --repa-schedule-steps    "$REPA_SCHEDULE_STEPS"    \
        --repa-diff-schedule     "$REPA_DIFF_SCHEDULE"     \
        --dinov2-repo-dir        "$DINOV2_REPO_DIR"        \
        --dinov2-weight-path     "$DINOV2_WEIGHT_PATH"     \
    2>&1 | stdbuf -oL tee "$LAUNCH_LOG"
