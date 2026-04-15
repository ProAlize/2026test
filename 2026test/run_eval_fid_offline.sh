#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ── 改动1：换成你自己的环境路径 ──────────────────────────────────
ENV_PREFIX="${ENV_PREFIX:-/mnt/tidal-alsh01/dataset/redaigc/yuantianshuo/tool/miniconda3/envs/dit}"
TORCHRUN_BIN="${TORCHRUN_BIN:-$ENV_PREFIX/bin/torchrun}"

CKPT_PATH="${1:-${CKPT_PATH:-}}"
if [[ -z "$CKPT_PATH" ]]; then
    echo "Usage: $0 /path/to/checkpoint.pt" >&2
    exit 1
fi

# ── 改动2：换成你自己的参考数据集和VAE路径 ───────────────────────
FID_REF_DIR="${FID_REF_DIR:-/data/temp/ILSVRC/Data/CLS-LOC/val}"
VAE_MODEL_DIR="${VAE_MODEL_DIR:-/mnt/tidal-alsh01/dataset/redaigc/yuantianshuo/2026/models/DiT-XL-2-256/vae}"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-4,5,6,7}"
NPROC_PER_NODE="${NPROC_PER_NODE:-4}"
# ── 改动3：batch size调大，加快生成速度 ───────────────────────────
PER_PROC_BATCH_SIZE="${PER_PROC_BATCH_SIZE:-64}"
# ── 改动4：标准FID用50000张 ──────────────────────────────────────
FID_NUM_SAMPLES="${FID_NUM_SAMPLES:-50000}"
FID_BATCH_SIZE="${FID_BATCH_SIZE:-32}"
NUM_SAMPLING_STEPS="${NUM_SAMPLING_STEPS:-250}"
CFG_SCALE="${CFG_SCALE:-1.5}"
KEEP_SAMPLES="${KEEP_SAMPLES:-0}"
OVERWRITE_EVAL="${OVERWRITE_EVAL:-1}"

if [[ ! -x "$TORCHRUN_BIN" ]]; then
    echo "Missing torchrun binary: $TORCHRUN_BIN" >&2
    exit 1
fi

if [[ ! -f "$CKPT_PATH" ]]; then
    echo "Checkpoint not found: $CKPT_PATH" >&2
    exit 1
fi

if [[ ! -d "$FID_REF_DIR" ]]; then
    echo "Reference directory not found: $FID_REF_DIR" >&2
    exit 1
fi

if [[ ! -e "$VAE_MODEL_DIR" ]]; then
    echo "VAE directory not found: $VAE_MODEL_DIR" >&2
    exit 1
fi

CKPT_DIR="$(cd "$(dirname "$CKPT_PATH")" && pwd)"
CKPT_STEM="$(basename "${CKPT_PATH%.pt}")"
EVAL_ROOT="${EVAL_ROOT:-$CKPT_DIR/offline_eval/${CKPT_STEM}_fid}"

mkdir -p "$EVAL_ROOT"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
LAUNCH_LOG="${LAUNCH_LOG:-$EVAL_ROOT/launch_${TIMESTAMP}.log}"

KEEP_ARGS=()
if [[ "$KEEP_SAMPLES" != "0" ]]; then
    KEEP_ARGS+=(--keep-samples)
fi

OVERWRITE_ARGS=()
if [[ "$OVERWRITE_EVAL" != "0" ]]; then
    OVERWRITE_ARGS+=(--overwrite)
fi

cd "$ROOT_DIR"

echo "========================================================"
echo "Starting offline FID evaluation"
echo "Checkpoint       : $CKPT_PATH"
echo "Eval root        : $EVAL_ROOT"
echo "Reference dir    : $FID_REF_DIR"
echo "VAE model dir    : $VAE_MODEL_DIR"
echo "Torchrun         : $TORCHRUN_BIN"
echo "GPUs             : $CUDA_VISIBLE_DEVICES"
echo "Num GPUs         : $NPROC_PER_NODE"
echo "Per-proc batch   : $PER_PROC_BATCH_SIZE"
echo "FID num samples  : $FID_NUM_SAMPLES"
echo "Sampling steps   : $NUM_SAMPLING_STEPS"
echo "CFG scale        : $CFG_SCALE"
echo "Launch log       : $LAUNCH_LOG"
echo "========================================================"

env CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" \
    "$TORCHRUN_BIN" --standalone --nnodes=1 --nproc_per_node="$NPROC_PER_NODE" \
    "$ROOT_DIR/eval_fid_ddp.py" \
    --ckpt "$CKPT_PATH" \
    --eval-root "$EVAL_ROOT" \
    --fid-ref-dir "$FID_REF_DIR" \
    --vae-model-dir "$VAE_MODEL_DIR" \
    --per-proc-batch-size "$PER_PROC_BATCH_SIZE" \
    --fid-num-samples "$FID_NUM_SAMPLES" \
    --fid-batch-size "$FID_BATCH_SIZE" \
    --num-sampling-steps "$NUM_SAMPLING_STEPS" \
    --cfg-scale "$CFG_SCALE" \
    "${KEEP_ARGS[@]}" \
    "${OVERWRITE_ARGS[@]}" \
    2>&1 | tee "$LAUNCH_LOG"
