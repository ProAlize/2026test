#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

ENV_PREFIX="${ENV_PREFIX:-/home/liuchunfa/anaconda3/envs/DiT}"
PYTHON_BIN="${PYTHON_BIN:-$ENV_PREFIX/bin/python}"
TORCHRUN_BIN="${TORCHRUN_BIN:-$ENV_PREFIX/bin/torchrun}"
REPA_DIR="${REPA_DIR:-$ROOT_DIR/project/REPA}"

ADM_PYTHON_BIN="${ADM_PYTHON_BIN:-/data/liuchunfa/2026qjx/venvs/adm_eval/bin/python}"
ADM_EVALUATOR_PY="${ADM_EVALUATOR_PY:-$ROOT_DIR/project/guided-diffusion/evaluations/evaluator.py}"
ADM_WORKDIR="${ADM_WORKDIR:-/data/liuchunfa/2026qjx/adm_eval_cache}"
ADM_CUDA_VISIBLE_DEVICES="${ADM_CUDA_VISIBLE_DEVICES:-}"

CKPT_PATH="${1:-${CKPT_PATH:-}}"
if [[ -z "$CKPT_PATH" ]]; then
    echo "Usage: $0 /path/to/sit_checkpoint.pt" >&2
    exit 1
fi
if [[ ! -f "$CKPT_PATH" ]]; then
    echo "Checkpoint not found: $CKPT_PATH" >&2
    exit 1
fi

MODEL="${MODEL:-SiT-XL/2}"
ENCODER_DEPTH="${ENCODER_DEPTH:-8}"
PROJECTOR_EMBED_DIMS="${PROJECTOR_EMBED_DIMS:-768}"
RESOLUTION="${RESOLUTION:-256}"
PATH_TYPE="${PATH_TYPE:-linear}"
MODE="${MODE:-sde}"
VAE_NAME="${VAE_NAME:-ema}"
VAE_MODEL_DIR="${VAE_MODEL_DIR:-/home/liuchunfa/.cache/modelscope/hub/models/facebook/DiT-XL-2-256/vae}"
NUM_CLASSES="${NUM_CLASSES:-1000}"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
NPROC_PER_NODE="${NPROC_PER_NODE:-4}"
PER_PROC_BATCH_SIZE="${PER_PROC_BATCH_SIZE:-16}"
FID_NUM_SAMPLES="${FID_NUM_SAMPLES:-50000}"
NUM_SAMPLING_STEPS="${NUM_SAMPLING_STEPS:-250}"
CFG_SCALE="${CFG_SCALE:-1.8}"
GUIDANCE_LOW="${GUIDANCE_LOW:-0.0}"
GUIDANCE_HIGH="${GUIDANCE_HIGH:-0.7}"
GLOBAL_SEED="${GLOBAL_SEED:-0}"
FUSED_ATTN="${FUSED_ATTN:-1}"
QK_NORM="${QK_NORM:-0}"
HEUN="${HEUN:-0}"

REF_BATCH_NPZ="${REF_BATCH_NPZ:-/data/liuchunfa/2026qjx/adm_ref/VIRTUAL_imagenet256_labeled.npz}"
STRICT_REF_NAME="${STRICT_REF_NAME:-1}"
STRICT_PAPER_CONFIG="${STRICT_PAPER_CONFIG:-1}"

OVERWRITE_EVAL="${OVERWRITE_EVAL:-1}"
SKIP_GENERATE_IF_NPZ_EXISTS="${SKIP_GENERATE_IF_NPZ_EXISTS:-1}"
KEEP_PNG_SAMPLES="${KEEP_PNG_SAMPLES:-0}"
KEEP_SAMPLE_NPZ="${KEEP_SAMPLE_NPZ:-1}"

if [[ "$FID_NUM_SAMPLES" != "50000" ]]; then
    echo "Strict mode expects FID_NUM_SAMPLES=50000, got $FID_NUM_SAMPLES" >&2
    exit 1
fi
if [[ "$NUM_SAMPLING_STEPS" != "250" ]]; then
    echo "Strict mode expects NUM_SAMPLING_STEPS=250, got $NUM_SAMPLING_STEPS" >&2
    exit 1
fi
if [[ "$STRICT_PAPER_CONFIG" != "0" ]]; then
    if [[ "$MODE" != "sde" ]]; then
        echo "Strict paper config expects MODE=sde, got $MODE" >&2
        exit 1
    fi
    if [[ "$CFG_SCALE" != "1.8" ]]; then
        echo "Strict paper config expects CFG_SCALE=1.8, got $CFG_SCALE" >&2
        exit 1
    fi
    if [[ "$GUIDANCE_LOW" != "0.0" || "$GUIDANCE_HIGH" != "0.7" ]]; then
        echo "Strict paper config expects GUIDANCE_LOW/HIGH = 0.0/0.7, got $GUIDANCE_LOW/$GUIDANCE_HIGH" >&2
        exit 1
    fi
fi

if [[ ! -x "$PYTHON_BIN" ]]; then
    echo "Missing python binary: $PYTHON_BIN" >&2
    exit 1
fi
if [[ ! -x "$TORCHRUN_BIN" ]]; then
    echo "Missing torchrun binary: $TORCHRUN_BIN" >&2
    exit 1
fi
if [[ ! -x "$ADM_PYTHON_BIN" ]]; then
    echo "Missing ADM python binary: $ADM_PYTHON_BIN" >&2
    exit 1
fi
if [[ ! -d "$REPA_DIR" ]]; then
    echo "REPA directory not found: $REPA_DIR" >&2
    exit 1
fi
if [[ ! -f "$ADM_EVALUATOR_PY" ]]; then
    echo "ADM evaluator script not found: $ADM_EVALUATOR_PY" >&2
    echo "Please set ADM_EVALUATOR_PY to guided-diffusion/evaluations/evaluator.py" >&2
    exit 1
fi
if [[ ! -d "$VAE_MODEL_DIR" ]]; then
    echo "VAE model directory not found: $VAE_MODEL_DIR" >&2
    exit 1
fi

if [[ ! -f "$REF_BATCH_NPZ" ]]; then
    echo "Reference batch npz not found: $REF_BATCH_NPZ" >&2
    echo "Hard strict mode requires official ADM reference batch npz." >&2
    echo "Expected file name: VIRTUAL_imagenet256_labeled.npz" >&2
    exit 1
fi
if [[ "$STRICT_REF_NAME" != "0" ]]; then
    if [[ "$(basename "$REF_BATCH_NPZ")" != "VIRTUAL_imagenet256_labeled.npz" ]]; then
        echo "Hard strict mode requires official reference filename VIRTUAL_imagenet256_labeled.npz" >&2
        echo "Current REF_BATCH_NPZ: $REF_BATCH_NPZ" >&2
        exit 1
    fi
fi

"$ADM_PYTHON_BIN" - <<'PY'
import tensorflow.compat.v1 as tf
print("TensorFlow OK:", tf.__version__)
PY

CKPT_DIR="$(cd "$(dirname "$CKPT_PATH")" && pwd)"
CKPT_STEM="$(basename "${CKPT_PATH%.pt}")"
EVAL_ROOT="${EVAL_ROOT:-$CKPT_DIR/offline_eval_adm/${CKPT_STEM}_adm50k}"
SAMPLE_ROOT="$EVAL_ROOT/generated"
mkdir -p "$EVAL_ROOT" "$SAMPLE_ROOT" "$ADM_WORKDIR"

MODEL_STRING="${MODEL//\//-}"
FOLDER_NAME="${MODEL_STRING}-${CKPT_STEM}-size-${RESOLUTION}-vae-${VAE_NAME}-cfg-${CFG_SCALE}-seed-${GLOBAL_SEED}-${MODE}"
SAMPLE_DIR="$SAMPLE_ROOT/$FOLDER_NAME"
SAMPLE_NPZ="${SAMPLE_DIR}.npz"

ADM_RAW_TXT="$EVAL_ROOT/adm_evaluator_stdout.txt"
ADM_METRICS_JSON="$EVAL_ROOT/adm_metrics.json"
LAUNCH_LOG="$EVAL_ROOT/launch_adm_$(date +%Y%m%d_%H%M%S).log"

{
    echo "Starting strict ADM evaluation"
    echo "Checkpoint: $CKPT_PATH"
    echo "Eval root: $EVAL_ROOT"
    echo "Sample dir: $SAMPLE_DIR"
    echo "Sample npz: $SAMPLE_NPZ"
    echo "Reference npz: $REF_BATCH_NPZ"
    echo "ADM evaluator: $ADM_EVALUATOR_PY"
    echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
    echo "NPROC_PER_NODE: $NPROC_PER_NODE"
    echo "PER_PROC_BATCH_SIZE: $PER_PROC_BATCH_SIZE"
    echo "FID_NUM_SAMPLES: $FID_NUM_SAMPLES"
    echo "NUM_SAMPLING_STEPS: $NUM_SAMPLING_STEPS"
    echo "CFG_SCALE: $CFG_SCALE"
    echo "MODE: $MODE"
    echo "GUIDANCE_LOW/HIGH: $GUIDANCE_LOW / $GUIDANCE_HIGH"
    echo "Launch log: $LAUNCH_LOG"

    if [[ "$OVERWRITE_EVAL" != "0" ]]; then
        rm -rf "$SAMPLE_DIR"
        rm -f "$SAMPLE_NPZ" "$ADM_RAW_TXT" "$ADM_METRICS_JSON"
    fi

    NEED_GENERATE=1
    if [[ "$SKIP_GENERATE_IF_NPZ_EXISTS" == "1" && -f "$SAMPLE_NPZ" ]]; then
        NEED_GENERATE=0
        echo "Found existing sample npz; skip generation."
    fi

    if [[ "$NEED_GENERATE" == "1" ]]; then
        GEN_ARGS=(
            --ckpt "$CKPT_PATH"
            --sample-dir "$SAMPLE_ROOT"
            --model "$MODEL"
            --num-classes "$NUM_CLASSES"
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
    fi

    if [[ ! -f "$SAMPLE_NPZ" ]]; then
        echo "Sample npz not found: $SAMPLE_NPZ" >&2
        exit 1
    fi

    cd "$ADM_WORKDIR"
    env CUDA_VISIBLE_DEVICES="$ADM_CUDA_VISIBLE_DEVICES" "$ADM_PYTHON_BIN" "$ADM_EVALUATOR_PY" "$REF_BATCH_NPZ" "$SAMPLE_NPZ" \
        2>&1 | tee "$ADM_RAW_TXT"

    "$PYTHON_BIN" - <<PY
import json
import pathlib
import re

txt_path = pathlib.Path("$ADM_RAW_TXT")
text = txt_path.read_text(encoding="utf-8", errors="ignore")

def extract_float(label):
    m = re.search(rf"{re.escape(label)}\\s*[:=]\\s*([-+]?\\d*\\.?\\d+(?:[eE][-+]?\\d+)?)", text)
    return float(m.group(1)) if m else None

result = {
    "checkpoint": "$CKPT_PATH",
    "sample_npz": "$SAMPLE_NPZ",
    "ref_npz": "$REF_BATCH_NPZ",
    "fid": extract_float("FID"),
    "sFID": extract_float("sFID"),
    "inception_score": extract_float("Inception Score"),
    "precision": extract_float("Precision"),
    "recall": extract_float("Recall"),
}

out_path = pathlib.Path("$ADM_METRICS_JSON")
out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\\n", encoding="utf-8")
print(json.dumps(result, ensure_ascii=False, indent=2))
PY

    if [[ "$KEEP_PNG_SAMPLES" == "0" ]]; then
        rm -rf "$SAMPLE_DIR"
        echo "Removed png sample folder: $SAMPLE_DIR"
    fi
    if [[ "$KEEP_SAMPLE_NPZ" == "0" ]]; then
        rm -f "$SAMPLE_NPZ"
        echo "Removed sample npz: $SAMPLE_NPZ"
    fi

    echo "Strict ADM evaluation finished."
    echo "Metrics json: $ADM_METRICS_JSON"
} 2>&1 | tee "$LAUNCH_LOG"
