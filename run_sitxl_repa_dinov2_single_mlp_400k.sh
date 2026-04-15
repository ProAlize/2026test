#!/usr/bin/env bash
# run_sit_repa_dinov2_400k.sh
# SiT-XL/2 + REPA (DINOv2 ViT-B/14 本地权重) 400k 步训练
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ── 环境配置 ────────────────────────────────────────────────────────────
ENV_PREFIX="${ENV_PREFIX:-/mnt/tidal-alsh01/dataset/redaigc/yuantianshuo/tool/miniconda3/envs/dit}"
TORCHRUN_BIN="${TORCHRUN_BIN:-$ENV_PREFIX/bin/torchrun}"

# ── 数据 / 模型路径 ─────────────────────────────────────────────────────
DATA_PATH="/data/temp/ILSVRC/Data/CLS-LOC/train"
RESULTS_DIR="${RESULTS_DIR:-/mnt/tidal-alsh01/dataset/redaigc/yuantianshuo/2026/results/sit_xl_repa_dinov2_400k}"
# SiT 与 DiT 共用同一个 VAE（sd-vae-ft-ema）
VAE_MODEL_DIR="${VAE_MODEL_DIR:-/mnt/tidal-alsh01/dataset/redaigc/yuantianshuo/2026/models/DiT-XL-2-256/vae}"

# # ── SiT-XL 预训练权重 ────────────────────────────────────────────────────
# SIT_CKPT="${SIT_CKPT:-/mnt/tidal-alsh01/dataset/redaigc/yuantianshuo/2026/models/SiT-XL-2-256/SiT-XL-2-256.pt}"

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
CKPT_EVERY="${CKPT_EVERY:-10000}"      # ← 每 10k 步保存一次
LOG_EVERY="${LOG_EVERY:-500}"

# ── SiT Transport 参数 ──────────────────────────────────────────────────
PATH_TYPE="${PATH_TYPE:-Linear}"
PREDICTION="${PREDICTION:-velocity}"
LOSS_WEIGHT="${LOSS_WEIGHT:-}"
TRAIN_EPS="${TRAIN_EPS:-}"
SAMPLE_EPS="${SAMPLE_EPS:-}"

# ── REPA 超参 ───────────────────────────────────────────────────────────
REPA_LAMBDA="${REPA_LAMBDA:-0.5}"
REPA_ENCODER_DEPTH="${REPA_ENCODER_DEPTH:-8}"
REPA_HIDDEN_DIM="${REPA_HIDDEN_DIM:-}"
REPA_TRAIN_SCHEDULE="${REPA_TRAIN_SCHEDULE:-linear_decay}"
REPA_SCHEDULE_STEPS="${REPA_SCHEDULE_STEPS:-40000}"
REPA_DIFF_SCHEDULE="${REPA_DIFF_SCHEDULE:-cosine}"

# ── Sanity checks ────────────────────────────────────────────────────────
if [[ ! -x "$TORCHRUN_BIN" ]]; then
    echo "[ERROR] torchrun not found: $TORCHRUN_BIN" >&2; exit 1
fi

for required_path in \
    "/mnt/tidal-alsh01/dataset/redaigc/yuantianshuo/2026/2026test/train_sitxl_repa_dinov2_single_mlp_400k.py" \
    "$DATA_PATH" \
    "$VAE_MODEL_DIR" \
    "$DINOV2_REPO_DIR" \
    "$DINOV2_REPO_DIR/hubconf.py" \
    "$DINOV2_WEIGHT_PATH"; do
    if [[ ! -e "$required_path" ]]; then
        echo "[ERROR] Required path not found: $required_path" >&2; exit 1
    fi
done

mkdir -p "$RESULTS_DIR"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
LAUNCH_LOG="${LAUNCH_LOG:-$RESULTS_DIR/launch_${TIMESTAMP}.log}"

# ── 可选参数 ─────────────────────────────────────────────────────────────
OPTIONAL_ARGS=()
if [[ -n "$REPA_HIDDEN_DIM" ]]; then
    OPTIONAL_ARGS+=(--repa-hidden-dim "$REPA_HIDDEN_DIM")
fi
if [[ -n "$LOSS_WEIGHT" ]]; then
    OPTIONAL_ARGS+=(--loss-weight "$LOSS_WEIGHT")
fi
if [[ -n "$TRAIN_EPS" ]]; then
    OPTIONAL_ARGS+=(--train-eps "$TRAIN_EPS")
fi
if [[ -n "$SAMPLE_EPS" ]]; then
    OPTIONAL_ARGS+=(--sample-eps "$SAMPLE_EPS")
fi

# ── 启动信息 ─────────────────────────────────────────────────────────────
cd "$ROOT_DIR"
echo "====== SiT-XL/2 + REPA (DINOv2 ViT-B/14) — 400k steps ======"
echo "Torchrun         : $TORCHRUN_BIN"
echo "GPUs             : $CUDA_VISIBLE_DEVICES  (nproc=$NPROC_PER_NODE)"
echo "Results dir      : $RESULTS_DIR"
echo "Launch log       : $LAUNCH_LOG"
echo "Max steps        : $MAX_STEPS"
echo "Global batch     : $GLOBAL_BATCH_SIZE"
echo "Ckpt every       : $CKPT_EVERY steps"
echo "Log every        : $LOG_EVERY steps"
echo "── SiT Transport ──────────────────────────────────────────────"
echo "Path type        : $PATH_TYPE"
echo "Prediction       : $PREDICTION"
echo "Loss weight      : ${LOSS_WEIGHT:-None (uniform)}"
echo "── REPA ────────────────────────────────────────────────────────"
echo "REPA lambda      : $REPA_LAMBDA"
echo "Encoder depth    : $REPA_ENCODER_DEPTH  (hook block idx=$((REPA_ENCODER_DEPTH-1)))"
echo "Train schedule   : $REPA_TRAIN_SCHEDULE (decay over $REPA_SCHEDULE_STEPS steps)"
echo "Diff schedule    : $REPA_DIFF_SCHEDULE"
echo "DINOv2 repo      : $DINOV2_REPO_DIR"
echo "DINOv2 weight    : $DINOV2_WEIGHT_PATH"
echo "VAE              : $VAE_MODEL_DIR"
[[ -n "$REPA_HIDDEN_DIM" ]] && echo "Projector hidden : $REPA_HIDDEN_DIM"
echo "==============================================================="

# ── 启动训练 ─────────────────────────────────────────────────────────────
env CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" \
    "$TORCHRUN_BIN" \
        --standalone \
        --nnodes=1 \
        --nproc_per_node="$NPROC_PER_NODE" \
    "/mnt/tidal-alsh01/dataset/redaigc/yuantianshuo/2026/2026test/train_sitxl_repa_dinov2_400k.py" \
        --data-path              "$DATA_PATH"             \
        --results-dir            "$RESULTS_DIR"           \
        --model                  SiT-XL/2                 \
        --image-size             256                      \
        --global-batch-size      "$GLOBAL_BATCH_SIZE"     \
        --num-workers            "$NUM_WORKERS"           \
        --epochs                 "$EPOCHS"                \
        --max-steps              "$MAX_STEPS"             \
        --ckpt-every             "$CKPT_EVERY"            \
        --log-every              "$LOG_EVERY"             \
        --vae-model-dir          "$VAE_MODEL_DIR"         \
        --path-type              "$PATH_TYPE"             \
        --prediction             "$PREDICTION"            \
        --repa                                            \
        --repa-lambda            "$REPA_LAMBDA"           \
        --repa-encoder-depth     "$REPA_ENCODER_DEPTH"    \
        --repa-train-schedule    "$REPA_TRAIN_SCHEDULE"   \
        --repa-schedule-steps    "$REPA_SCHEDULE_STEPS"   \
        --repa-diff-schedule     "$REPA_DIFF_SCHEDULE"    \
        --dinov2-repo-dir        "$DINOV2_REPO_DIR"       \
        --dinov2-weight-path     "$DINOV2_WEIGHT_PATH"    \
        "${OPTIONAL_ARGS[@]}"                             \
    2>&1 | tee "$LAUNCH_LOG"
