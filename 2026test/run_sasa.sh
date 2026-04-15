#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

ENV_PREFIX="${ENV_PREFIX:-/mnt/tidal-alsh01/dataset/redaigc/yuantianshuo/tool/miniconda3/envs/dit}"
TORCHRUN_BIN="${TORCHRUN_BIN:-$ENV_PREFIX/bin/torchrun}"

DATA_PATH="/data/temp/ILSVRC/Data/CLS-LOC/train"
RESULTS_DIR="${RESULTS_DIR:-/mnt/tidal-alsh01/dataset/redaigc/yuantianshuo/2026/results/dit_xl_sasa_stepdecay_80k}"
DINO_MODEL_DIR="${DINO_MODEL_DIR:-/mnt/tidal-alsh01/dataset/redaigc/yuantianshuo/tmp/DiT/dinov3-vitb16-pretrain-lvd1689m}"
VAE_MODEL_DIR="${VAE_MODEL_DIR:-/mnt/tidal-alsh01/dataset/redaigc/yuantianshuo/2026/models/DiT-XL-2-256/vae}"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
NPROC_PER_NODE="${NPROC_PER_NODE:-4}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-256}"
NUM_WORKERS="${NUM_WORKERS:-4}"
EPOCHS="${EPOCHS:-1400}"
MAX_STEPS="${MAX_STEPS:-80000}"
CKPT_EVERY="${CKPT_EVERY:-1000}"
LOG_EVERY="${LOG_EVERY:-500}"

# SASA / REPA args
REPA_LAMBDA="${REPA_LAMBDA:-0.1}"
REPA_TOKEN_LAYER="${REPA_TOKEN_LAYER:-}"
REPA_HIDDEN_DIM="${REPA_HIDDEN_DIM:-}"
REPA_TRAIN_SCHEDULE="${REPA_TRAIN_SCHEDULE:-linear_decay}"
REPA_SCHEDULE_STEPS="${REPA_SCHEDULE_STEPS:-40000}"
# REPA_DIFF_SCHEDULE 仍传入以保持参数兼容，但训练代码中已忽略
REPA_DIFF_SCHEDULE="${REPA_DIFF_SCHEDULE:-cosine}"

# ---------------------------------------------------------------
# Sanity checks
# ---------------------------------------------------------------
if [[ ! -x "$TORCHRUN_BIN" ]]; then
    echo "Missing torchrun binary: $TORCHRUN_BIN" >&2
    exit 1
fi

for required_path in \
    "$ROOT_DIR/train_sasa.py" \
    "$DATA_PATH" \
    "$DINO_MODEL_DIR" \
    "$VAE_MODEL_DIR"; do
    if [[ ! -e "$required_path" ]]; then
        echo "Required path not found: $required_path" >&2
        exit 1
    fi
done

mkdir -p "$RESULTS_DIR"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
LAUNCH_LOG="${LAUNCH_LOG:-$RESULTS_DIR/launch_${TIMESTAMP}.log}"

# ---------------------------------------------------------------
# Optional args
# ---------------------------------------------------------------
OPTIONAL_ARGS=()

if [[ -n "$REPA_TOKEN_LAYER" ]]; then
    OPTIONAL_ARGS+=(--repa-token-layer "$REPA_TOKEN_LAYER")
fi

if [[ -n "$REPA_HIDDEN_DIM" ]]; then
    OPTIONAL_ARGS+=(--repa-hidden-dim "$REPA_HIDDEN_DIM")
fi

# ---------------------------------------------------------------
# Summary
# ---------------------------------------------------------------
cd "$ROOT_DIR"

echo "===== DiT-XL/2 SASA  step-decay (no diff-timestep weighting) ====="
echo "Environment prefix : $ENV_PREFIX"
echo "Torchrun           : $TORCHRUN_BIN"
echo "GPUs               : $CUDA_VISIBLE_DEVICES"
echo "Results dir        : $RESULTS_DIR"
echo "Launch log         : $LAUNCH_LOG"
echo "Max steps          : $MAX_STEPS"
echo "Epochs (max bound) : $EPOCHS"
echo "Global batch size  : $GLOBAL_BATCH_SIZE"
echo "Ckpt every         : $CKPT_EVERY steps"
echo "Log every          : $LOG_EVERY steps"
echo "REPA lambda        : $REPA_LAMBDA"
echo "Train schedule     : $REPA_TRAIN_SCHEDULE  (decay over $REPA_SCHEDULE_STEPS steps)"
echo "Diff  schedule     : DISABLED (uniform across all timesteps)"
echo "VAE model dir      : $VAE_MODEL_DIR"
[[ -n "$REPA_TOKEN_LAYER" ]] && echo "Token layer        : $REPA_TOKEN_LAYER"
[[ -n "$REPA_HIDDEN_DIM"  ]] && echo "Projector hidden   : $REPA_HIDDEN_DIM"
echo "=================================================================="

# ---------------------------------------------------------------
# Launch
# ---------------------------------------------------------------
env CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" \
    "$TORCHRUN_BIN" --standalone --nnodes=1 --nproc_per_node="$NPROC_PER_NODE" \
    "$ROOT_DIR/train_sasa.py" \
    --data-path              "$DATA_PATH" \
    --results-dir            "$RESULTS_DIR" \
    --model                  DiT-XL/2 \
    --image-size             256 \
    --global-batch-size      "$GLOBAL_BATCH_SIZE" \
    --num-workers            "$NUM_WORKERS" \
    --epochs                 "$EPOCHS" \
    --max-steps              "$MAX_STEPS" \
    --ckpt-every             "$CKPT_EVERY" \
    --log-every              "$LOG_EVERY" \
    --repa \
    --repa-lambda            "$REPA_LAMBDA" \
    --repa-train-schedule    "$REPA_TRAIN_SCHEDULE" \
    --repa-schedule-steps    "$REPA_SCHEDULE_STEPS" \
    --repa-diff-schedule     "$REPA_DIFF_SCHEDULE" \
    --dino-model-dir         "$DINO_MODEL_DIR" \
    --vae-model-dir          "$VAE_MODEL_DIR" \
    "${OPTIONAL_ARGS[@]}" \
    2>&1 | tee "$LAUNCH_LOG"
