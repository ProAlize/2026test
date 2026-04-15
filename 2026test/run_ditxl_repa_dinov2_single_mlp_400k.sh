#!/usr/bin/env bash
# run_dinov2_repa_400k.sh
# DiT-XL/2 + REPA (DINOv2 ViT-B/14 本地权重) 400k 步训练
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ── 环境配置 ────────────────────────────────────────────────────────────
ENV_PREFIX="${ENV_PREFIX:-/mnt/tidal-alsh01/dataset/redaigc/yuantianshuo/tool/miniconda3/envs/dit}"
TORCHRUN_BIN="${TORCHRUN_BIN:-$ENV_PREFIX/bin/torchrun}"

# ── 数据 / 模型路径 ─────────────────────────────────────────────────────
DATA_PATH="/data/temp/ILSVRC/Data/CLS-LOC/train"
RESULTS_DIR="${RESULTS_DIR:-/mnt/tidal-alsh01/dataset/redaigc/yuantianshuo/2026/results/dit_xl_repa_dinov2_400k}"
VAE_MODEL_DIR="${VAE_MODEL_DIR:-/mnt/tidal-alsh01/dataset/redaigc/yuantianshuo/2026/models/DiT-XL-2-256/vae}"

# ── DINOv2 路径（对应图中目录结构） ─────────────────────────────────────
# 源码目录：含 hubconf.py
DINOV2_REPO_DIR="${DINOV2_REPO_DIR:-/mnt/tidal-alsh01/dataset/redaigc/yuantianshuo/tmp/dinov2}"
# 权重文件
DINOV2_WEIGHT_PATH="${DINOV2_WEIGHT_PATH:-/mnt/tidal-alsh01/dataset/redaigc/yuantianshuo/tmp/dinov2_weights/dinov2_vitb14_pretrain.pth}"

# ── GPU / 训练超参 ──────────────────────────────────────────────────────
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
NPROC_PER_NODE="${NPROC_PER_NODE:-4}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-256}"
NUM_WORKERS="${NUM_WORKERS:-4}"
EPOCHS="${EPOCHS:-1400}"
MAX_STEPS="${MAX_STEPS:-400000}"
CKPT_EVERY="${CKPT_EVERY:-1000}"
LOG_EVERY="${LOG_EVERY:-500}"

# ── REPA 超参 ───────────────────────────────────────────────────────────
REPA_LAMBDA="${REPA_LAMBDA:-0.1}"
REPA_TOKEN_LAYER="${REPA_TOKEN_LAYER:-}"       # 空 = 默认最后一层
REPA_HIDDEN_DIM="${REPA_HIDDEN_DIM:-}"         # 空 = DiT hidden_size
REPA_TRAIN_SCHEDULE="${REPA_TRAIN_SCHEDULE:-linear_decay}"
REPA_SCHEDULE_STEPS="${REPA_SCHEDULE_STEPS:-40000}"
REPA_DIFF_SCHEDULE="${REPA_DIFF_SCHEDULE:-cosine}"   # REPA原版默认

# ── Sanity checks ────────────────────────────────────────────────────────
if [[ ! -x "$TORCHRUN_BIN" ]]; then
    echo "[ERROR] torchrun not found: $TORCHRUN_BIN" >&2; exit 1
fi

for required_path in \
    "/mnt/tidal-alsh01/dataset/redaigc/yuantianshuo/2026/2026test/train_ditxl_repa_dinov2_single_mlp_400k.py" \
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
if [[ -n "$REPA_TOKEN_LAYER" ]]; then
    OPTIONAL_ARGS+=(--repa-token-layer "$REPA_TOKEN_LAYER")
fi
if [[ -n "$REPA_HIDDEN_DIM" ]]; then
    OPTIONAL_ARGS+=(--repa-hidden-dim "$REPA_HIDDEN_DIM")
fi

# ── 启动信息 ─────────────────────────────────────────────────────────────
cd "$ROOT_DIR"
echo "====== DiT-XL/2 + REPA (DINOv2 ViT-B/14) — 400k steps ======"
echo "Torchrun         : $TORCHRUN_BIN"
echo "GPUs             : $CUDA_VISIBLE_DEVICES  (nproc=$NPROC_PER_NODE)"
echo "Results dir      : $RESULTS_DIR"
echo "Launch log       : $LAUNCH_LOG"
echo "Max steps        : $MAX_STEPS"
echo "Global batch     : $GLOBAL_BATCH_SIZE"
echo "Ckpt every       : $CKPT_EVERY steps"
echo "Log every        : $LOG_EVERY steps"
echo "REPA lambda      : $REPA_LAMBDA"
echo "Train schedule   : $REPA_TRAIN_SCHEDULE (decay over $REPA_SCHEDULE_STEPS steps)"
echo "Diff schedule    : $REPA_DIFF_SCHEDULE  (REPA-style timestep weighting)"
echo "DINOv2 repo      : $DINOV2_REPO_DIR"
echo "DINOv2 weight    : $DINOV2_WEIGHT_PATH"
echo "VAE              : $VAE_MODEL_DIR"
[[ -n "$REPA_TOKEN_LAYER" ]] && echo "Token layer      : $REPA_TOKEN_LAYER"
[[ -n "$REPA_HIDDEN_DIM"  ]] && echo "Projector hidden : $REPA_HIDDEN_DIM"
echo "=============================================================="

# ── 启动训练 ─────────────────────────────────────────────────────────────
env CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" \
    "$TORCHRUN_BIN" \
        --standalone \
        --nnodes=1 \
        --nproc_per_node="$NPROC_PER_NODE" \
    "/mnt/tidal-alsh01/dataset/redaigc/yuantianshuo/2026/2026test/train_ditxl_repa_dinov2_single_mlp_400k.py" \
        --data-path              "$DATA_PATH"             \
        --results-dir            "$RESULTS_DIR"           \
        --model                  DiT-XL/2                 \
        --image-size             256                      \
        --global-batch-size      "$GLOBAL_BATCH_SIZE"     \
        --num-workers            "$NUM_WORKERS"           \
        --epochs                 "$EPOCHS"                \
        --max-steps              "$MAX_STEPS"             \
        --ckpt-every             "$CKPT_EVERY"            \
        --log-every              "$LOG_EVERY"             \
        --vae-model-dir          "$VAE_MODEL_DIR"         \
        --repa                                            \
        --repa-lambda            "$REPA_LAMBDA"           \
        --repa-train-schedule    "$REPA_TRAIN_SCHEDULE"   \
        --repa-schedule-steps    "$REPA_SCHEDULE_STEPS"   \
        --repa-diff-schedule     "$REPA_DIFF_SCHEDULE"    \
        --dinov2-repo-dir        "$DINOV2_REPO_DIR"       \
        --dinov2-weight-path     "$DINOV2_WEIGHT_PATH"    \
        "${OPTIONAL_ARGS[@]}"                             \
    2>&1 | tee "$LAUNCH_LOG"
