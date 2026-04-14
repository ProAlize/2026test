#!/usr/bin/env bash
# Matched t-aware baseline run (XL/2, single seed, no resume)
set -euo pipefail

WORKTREE_DIR="${WORKTREE_DIR:-/home/liuchunfa/2026qjx/worktrees/exp_taware_adm_eval_20260410}"
ENV_PREFIX="${ENV_PREFIX:-/home/liuchunfa/anaconda3/envs/DiT}"
TORCHRUN_BIN="${TORCHRUN_BIN:-$ENV_PREFIX/bin/torchrun}"
PYTHON_BIN="${PYTHON_BIN:-$ENV_PREFIX/bin/python}"

DATA_PATH="${DATA_PATH:-/data/liuchunfa/2026qjx/repa_imagenet256_official/images}"
VAE_MODEL_DIR="${VAE_MODEL_DIR:-/home/liuchunfa/.cache/modelscope/hub/models/facebook/DiT-XL-2-256/vae}"
DINO_MODEL_DIR="${DINO_MODEL_DIR:-/home/liuchunfa/.cache/modelscope/hub/models/facebook/dinov3-vitb16-pretrain-lvd1689m}"

RESULTS_ROOT="${RESULTS_ROOT:-/data/liuchunfa/2026qjx/2026test_runs/taware_xl_compare}"
RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
NPROC_PER_NODE="${NPROC_PER_NODE:-4}"

MODEL="${MODEL:-DiT-XL/2}"
IMAGE_SIZE="${IMAGE_SIZE:-256}"
NUM_CLASSES="${NUM_CLASSES:-auto}"
GLOBAL_SEED="${GLOBAL_SEED:-0}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-128}"
NUM_WORKERS="${NUM_WORKERS:-8}"
EPOCHS="${EPOCHS:-1400}"
MAX_STEPS="${MAX_STEPS:-40000}"
CKPT_EVERY="${CKPT_EVERY:-2000}"
LOG_EVERY="${LOG_EVERY:-200}"

REPA_LAMBDA="${REPA_LAMBDA:-0.1}"
REPA_DIFF_SCHEDULE="${REPA_DIFF_SCHEDULE:-linear_high_noise}"
REPA_DIFF_THRESHOLD="${REPA_DIFF_THRESHOLD:-0.5}"

FORBID_RESUME="${FORBID_RESUME:-1}"
if [[ "$FORBID_RESUME" == "1" ]]; then
    if [[ -n "${RESUME_CKPT:-}" || -n "${RESUME:-}" ]]; then
        echo "[ERROR] This launcher forbids resume. Unset RESUME/RESUME_CKPT." >&2
        exit 1
    fi
fi

for p in \
    "$WORKTREE_DIR/train_2.py" \
    "$TORCHRUN_BIN" \
    "$PYTHON_BIN" \
    "$DATA_PATH" \
    "$VAE_MODEL_DIR" \
    "$DINO_MODEL_DIR"; do
    [[ -e "$p" ]] || { echo "[ERROR] Missing path: $p" >&2; exit 1; }
done

GPU_COUNT=$(echo "$CUDA_VISIBLE_DEVICES" | tr ',' '\n' | wc -l | tr -d ' ')
if [[ "$GPU_COUNT" -lt "$NPROC_PER_NODE" ]]; then
    echo "[ERROR] NPROC_PER_NODE=$NPROC_PER_NODE exceeds visible GPU count=$GPU_COUNT" >&2
    exit 1
fi

if [[ "$NUM_CLASSES" == "auto" ]]; then
    NUM_CLASSES="$($PYTHON_BIN - <<PY
import os
print(sum(1 for x in os.scandir('$DATA_PATH') if x.is_dir()))
PY
)"
fi

RUN_DIR="$RESULTS_ROOT/$RUN_TAG"
mkdir -p "$RUN_DIR"
LAUNCH_LOG="$RUN_DIR/launch_taware_xl.log"

echo "===== t-aware matched baseline (XL/2, no resume) ====="
echo "worktree        : $WORKTREE_DIR"
echo "results         : $RUN_DIR"
echo "model           : $MODEL"
echo "num_classes     : $NUM_CLASSES"
echo "global_batch    : $GLOBAL_BATCH_SIZE"
echo "max_steps       : $MAX_STEPS"
echo "repa_lambda     : $REPA_LAMBDA"
echo "repa_schedule   : $REPA_DIFF_SCHEDULE"
echo "repa_threshold  : $REPA_DIFF_THRESHOLD"
echo "======================================================"

cd "$WORKTREE_DIR"
env CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" \
    "$TORCHRUN_BIN" --standalone --nnodes=1 --nproc_per_node="$NPROC_PER_NODE" \
    "$WORKTREE_DIR/train_2.py" \
    --data-path "$DATA_PATH" \
    --results-dir "$RUN_DIR" \
    --model "$MODEL" \
    --image-size "$IMAGE_SIZE" \
    --num-classes "$NUM_CLASSES" \
    --global-batch-size "$GLOBAL_BATCH_SIZE" \
    --global-seed "$GLOBAL_SEED" \
    --num-workers "$NUM_WORKERS" \
    --epochs "$EPOCHS" \
    --max-steps "$MAX_STEPS" \
    --ckpt-every "$CKPT_EVERY" \
    --log-every "$LOG_EVERY" \
    --vae-model-dir "$VAE_MODEL_DIR" \
    --repa \
    --repa-lambda "$REPA_LAMBDA" \
    --repa-diff-schedule "$REPA_DIFF_SCHEDULE" \
    --repa-diff-threshold "$REPA_DIFF_THRESHOLD" \
    --dino-model-dir "$DINO_MODEL_DIR" \
    2>&1 | tee "$LAUNCH_LOG"
