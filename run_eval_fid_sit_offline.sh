#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

ENV_PREFIX="${ENV_PREFIX:-/home/liuchunfa/anaconda3/envs/DiT}"
PYTHON_BIN="${PYTHON_BIN:-$ENV_PREFIX/bin/python}"
TORCHRUN_BIN="${TORCHRUN_BIN:-$ENV_PREFIX/bin/torchrun}"
REPA_DIR="${REPA_DIR:-$ROOT_DIR/project/REPA}"

CKPT_PATH="${1:-${CKPT_PATH:-}}"
if [[ -z "$CKPT_PATH" ]]; then
    echo "Usage: $0 /path/to/sit_checkpoint.pt" >&2
    exit 1
fi

FID_REF_DIR="${FID_REF_DIR:-/data/liuchunfa/2026qjx/ILSVRC/Data/CLS-LOC/val}"
VAE_MODEL_DIR="${VAE_MODEL_DIR:-/home/liuchunfa/.cache/modelscope/hub/models/facebook/DiT-XL-2-256/vae}"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
NPROC_PER_NODE="${NPROC_PER_NODE:-1}"
PER_PROC_BATCH_SIZE="${PER_PROC_BATCH_SIZE:-8}"
FID_NUM_SAMPLES="${FID_NUM_SAMPLES:-5000}"
FID_BATCH_SIZE="${FID_BATCH_SIZE:-32}"
FID_NUM_WORKERS="${FID_NUM_WORKERS:-4}"
REF_STATS_PATH="${REF_STATS_PATH:-/data/liuchunfa/2026qjx/fid_stats/imagenet_val_256_center_crop_dims2048.npz}"
NUM_SAMPLING_STEPS="${NUM_SAMPLING_STEPS:-250}"
CFG_SCALE="${CFG_SCALE:-1.5}"
GLOBAL_SEED="${GLOBAL_SEED:-0}"

MODEL="${MODEL:-SiT-XL/2}"
ENCODER_DEPTH="${ENCODER_DEPTH:-8}"
PROJECTOR_EMBED_DIMS="${PROJECTOR_EMBED_DIMS:-768}"
RESOLUTION="${RESOLUTION:-256}"
PATH_TYPE="${PATH_TYPE:-linear}"
MODE="${MODE:-ode}"
GUIDANCE_LOW="${GUIDANCE_LOW:-0.0}"
GUIDANCE_HIGH="${GUIDANCE_HIGH:-1.0}"
VAE_NAME="${VAE_NAME:-ema}"
FUSED_ATTN="${FUSED_ATTN:-1}"
QK_NORM="${QK_NORM:-0}"
HEUN="${HEUN:-0}"

KEEP_SAMPLES="${KEEP_SAMPLES:-0}"
OVERWRITE_EVAL="${OVERWRITE_EVAL:-1}"

if [[ ! -x "$PYTHON_BIN" ]]; then
    echo "Missing python binary: $PYTHON_BIN" >&2
    exit 1
fi
if [[ ! -x "$TORCHRUN_BIN" ]]; then
    echo "Missing torchrun binary: $TORCHRUN_BIN" >&2
    exit 1
fi
if [[ ! -f "$CKPT_PATH" ]]; then
    echo "Checkpoint not found: $CKPT_PATH" >&2
    exit 1
fi
if [[ ! -d "$REPA_DIR" ]]; then
    echo "REPA directory not found: $REPA_DIR" >&2
    exit 1
fi
if [[ ! -d "$FID_REF_DIR" ]]; then
    echo "Reference directory not found: $FID_REF_DIR" >&2
    exit 1
fi
if [[ ! -e "$VAE_MODEL_DIR" ]]; then
    echo "VAE model directory not found: $VAE_MODEL_DIR" >&2
    exit 1
fi

CKPT_DIR="$(cd "$(dirname "$CKPT_PATH")" && pwd)"
CKPT_STEM="$(basename "${CKPT_PATH%.pt}")"
EVAL_ROOT="${EVAL_ROOT:-$CKPT_DIR/offline_eval/${CKPT_STEM}_fid}"
SAMPLE_ROOT="$EVAL_ROOT/generated"

MODEL_STRING="${MODEL//\//-}"
CKPT_STRING="$CKPT_STEM"
FOLDER_NAME="${MODEL_STRING}-${CKPT_STRING}-size-${RESOLUTION}-vae-${VAE_NAME}-cfg-${CFG_SCALE}-seed-${GLOBAL_SEED}-${MODE}"
SAMPLE_DIR="$SAMPLE_ROOT/$FOLDER_NAME"
SAMPLE_NPZ="${SAMPLE_DIR}.npz"

mkdir -p "$EVAL_ROOT" "$SAMPLE_ROOT"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
LAUNCH_LOG="${LAUNCH_LOG:-$EVAL_ROOT/launch_sit_${TIMESTAMP}.log}"
FID_JSON="$EVAL_ROOT/fid_result.json"
FID_TXT="$EVAL_ROOT/fid_result.txt"

{
    echo "Starting SiT offline FID evaluation"
    echo "Checkpoint: $CKPT_PATH"
    echo "Eval root: $EVAL_ROOT"
    echo "Sample dir: $SAMPLE_DIR"
    echo "Reference dir: $FID_REF_DIR"
    echo "GPUs: $CUDA_VISIBLE_DEVICES"
    echo "NPROC_PER_NODE: $NPROC_PER_NODE"
    echo "FID_NUM_SAMPLES: $FID_NUM_SAMPLES"
    echo "NUM_SAMPLING_STEPS: $NUM_SAMPLING_STEPS"
    echo "CFG_SCALE: $CFG_SCALE"
    echo "REF_STATS_PATH: $REF_STATS_PATH"
    echo "Launch log: $LAUNCH_LOG"

    if [[ "$OVERWRITE_EVAL" != "0" ]]; then
        rm -rf "$SAMPLE_DIR"
        rm -f "$SAMPLE_NPZ" "$FID_JSON" "$FID_TXT"
    fi

    GEN_ARGS=(
        --ckpt "$CKPT_PATH"
        --sample-dir "$SAMPLE_ROOT"
        --model "$MODEL"
        --encoder-depth "$ENCODER_DEPTH"
        --projector-embed-dims "$PROJECTOR_EMBED_DIMS"
        --resolution "$RESOLUTION"
        --vae "$VAE_NAME"
        --vae-model-dir "$VAE_MODEL_DIR"
        --per-proc-batch-size "$PER_PROC_BATCH_SIZE"
        --num-fid-samples "$FID_NUM_SAMPLES"
        --mode "$MODE"
        --cfg-scale "$CFG_SCALE"
        --path-type "$PATH_TYPE"
        --num-steps "$NUM_SAMPLING_STEPS"
        --guidance-low "$GUIDANCE_LOW"
        --guidance-high "$GUIDANCE_HIGH"
        --global-seed "$GLOBAL_SEED"
    )

    if [[ "$FUSED_ATTN" != "0" ]]; then
        GEN_ARGS+=(--fused-attn)
    else
        GEN_ARGS+=(--no-fused-attn)
    fi
    if [[ "$QK_NORM" != "0" ]]; then
        GEN_ARGS+=(--qk-norm)
    else
        GEN_ARGS+=(--no-qk-norm)
    fi
    if [[ "$HEUN" != "0" ]]; then
        GEN_ARGS+=(--heun)
    fi

    cd "$REPA_DIR"
    env CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" \
        "$TORCHRUN_BIN" --standalone --nnodes=1 --nproc_per_node="$NPROC_PER_NODE" \
        "$REPA_DIR/generate.py" "${GEN_ARGS[@]}"

    if [[ ! -d "$SAMPLE_DIR" ]]; then
        echo "Sample directory not found after generation: $SAMPLE_DIR" >&2
        exit 1
    fi

    cd "$ROOT_DIR"
    env CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" \
        "$PYTHON_BIN" "$ROOT_DIR/compute_fid_recursive.py" \
        --sample-dir "$SAMPLE_DIR" \
        --ref-dir "$FID_REF_DIR" \
        --ref-stats "$REF_STATS_PATH" \
        --device "cuda:0" \
        --image-size "$RESOLUTION" \
        --batch-size "$FID_BATCH_SIZE" \
        --num-workers "$FID_NUM_WORKERS" \
        --out-json "$FID_JSON"

    FID_VALUE="$("$PYTHON_BIN" - <<PY
import json
from pathlib import Path
obj = json.loads(Path("$FID_JSON").read_text())
print(obj["fid"])
PY
)"

    echo "FID: $FID_VALUE" | tee "$FID_TXT"

    if [[ "$KEEP_SAMPLES" == "0" ]]; then
        rm -rf "$SAMPLE_DIR"
        rm -f "$SAMPLE_NPZ"
        echo "Removed generated samples to save disk."
    fi

    echo "SiT FID evaluation done."
} 2>&1 | tee "$LAUNCH_LOG"
