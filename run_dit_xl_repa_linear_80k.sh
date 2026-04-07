#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

ENV_PREFIX="${ENV_PREFIX:-/home/liuchunfa/anaconda3/envs/DiT}"
TORCHRUN_BIN="${TORCHRUN_BIN:-$ENV_PREFIX/bin/torchrun}"

DATA_PATH="${DATA_PATH:-/data/liuchunfa/2026qjx/ILSVRC/Data/CLS-LOC/train}"
FID_REF_DIR="${FID_REF_DIR:-/data/liuchunfa/2026qjx/ILSVRC/Data/CLS-LOC/val}"
RESULTS_DIR="${RESULTS_DIR:-/data/liuchunfa/2026qjx/dit_xl_repa_linear_80k}"
DINO_MODEL_DIR="${DINO_MODEL_DIR:-/home/liuchunfa/.cache/modelscope/hub/models/facebook/dinov3-vitb16-pretrain-lvd1689m}"
VAE_MODEL_DIR="${VAE_MODEL_DIR:-/home/liuchunfa/.cache/modelscope/hub/models/facebook/DiT-XL-2-256/vae}"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
NPROC_PER_NODE="${NPROC_PER_NODE:-4}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-4}"
NUM_WORKERS="${NUM_WORKERS:-4}"
MAX_STEPS="${MAX_STEPS:-80000}"
CKPT_EVERY="${CKPT_EVERY:-10000}"
FID_EVERY="${FID_EVERY:-0}"
FID_NUM_SAMPLES="${FID_NUM_SAMPLES:-5000}"
REPA_LAMBDA="${REPA_LAMBDA:-1.0}"
REPA_SCHEDULE_STEPS="${REPA_SCHEDULE_STEPS:-40000}"
KEEP_FID_SAMPLES="${KEEP_FID_SAMPLES:-1}"

if [[ ! -x "$TORCHRUN_BIN" ]]; then
    echo "Missing torchrun binary: $TORCHRUN_BIN" >&2
    exit 1
fi

for required_path in \
    "$ROOT_DIR/train_2.py" \
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

KEEP_FID_ARGS=()
if [[ "$KEEP_FID_SAMPLES" != "0" ]]; then
    KEEP_FID_ARGS+=(--keep-fid-samples)
fi

cd "$ROOT_DIR"

echo "Starting DiT-XL/2 REPA run"
echo "Environment prefix: $ENV_PREFIX"
echo "Torchrun: $TORCHRUN_BIN"
echo "GPUs: $CUDA_VISIBLE_DEVICES"
echo "Results dir: $RESULTS_DIR"
echo "Launch log: $LAUNCH_LOG"
if [[ "$FID_EVERY" -le 0 ]]; then
    echo "Inline FID: disabled"
else
    echo "Inline FID every: $FID_EVERY"
    echo "FID samples kept: $KEEP_FID_SAMPLES"
fi

env CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" \
    "$TORCHRUN_BIN" --standalone --nnodes=1 --nproc_per_node="$NPROC_PER_NODE" \
    "$ROOT_DIR/train_2.py" \
    --data-path "$DATA_PATH" \
    --results-dir "$RESULTS_DIR" \
    --model DiT-XL/2 \
    --image-size 256 \
    --global-batch-size "$GLOBAL_BATCH_SIZE" \
    --num-workers "$NUM_WORKERS" \
    --max-steps "$MAX_STEPS" \
    --ckpt-every "$CKPT_EVERY" \
    --fid-every "$FID_EVERY" \
    --fid-num-samples "$FID_NUM_SAMPLES" \
    --fid-ref-dir "$FID_REF_DIR" \
    --repa \
    --repa-lambda "$REPA_LAMBDA" \
    --repa-schedule linear \
    --repa-schedule-steps "$REPA_SCHEDULE_STEPS" \
    --dino-model-dir "$DINO_MODEL_DIR" \
    --vae-model-dir "$VAE_MODEL_DIR" \
    "${KEEP_FID_ARGS[@]}" \
    2>&1 | tee "$LAUNCH_LOG"
