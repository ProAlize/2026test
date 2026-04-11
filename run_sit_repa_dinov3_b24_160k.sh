#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="${REPO_DIR:-$ROOT_DIR/project/REPA}"

ENV_PREFIX="${ENV_PREFIX:-/home/liuchunfa/anaconda3/envs/DiT}"
PYTHON_BIN="${PYTHON_BIN:-$ENV_PREFIX/bin/python}"
ACCELERATE_BIN="${ACCELERATE_BIN:-$ENV_PREFIX/bin/accelerate}"

DATA_DIR="${DATA_DIR:-/data/liuchunfa/2026qjx/repa_imagenet256_ditvae}"
OUTPUT_DIR="${OUTPUT_DIR:-/data/liuchunfa/2026qjx/repa_sit_dinov3_b36_160k}"
EXP_NAME="${EXP_NAME:-sit_xl2_dinov3_b_enc8_b36_160k}"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
NUM_PROCESSES="${NUM_PROCESSES:-4}"
MAIN_PROCESS_PORT="${MAIN_PROCESS_PORT:-29521}"

BATCH_SIZE="${BATCH_SIZE:-36}"
MAX_TRAIN_STEPS="${MAX_TRAIN_STEPS:-160000}"
CHECKPOINTING_STEPS="${CHECKPOINTING_STEPS:-20000}"
SAMPLING_STEPS="${SAMPLING_STEPS:-20000}"
NUM_WORKERS="${NUM_WORKERS:-4}"
SEED="${SEED:-0}"

MODEL="${MODEL:-SiT-XL/2}"
ENC_TYPE="${ENC_TYPE:-dinov3-vit-b}"
DINO_MODEL_DIR="${DINO_MODEL_DIR:-/home/liuchunfa/.cache/modelscope/hub/models/facebook/dinov3-vitb16-pretrain-lvd1689m}"
VAE_MODEL_DIR="${VAE_MODEL_DIR:-/home/liuchunfa/.cache/modelscope/hub/models/facebook/DiT-XL-2-256/vae}"
PROJ_COEFF="${PROJ_COEFF:-0.5}"
PROJ_DIFF_SCHEDULE="${PROJ_DIFF_SCHEDULE:-linear_high_noise}"
ENCODER_DEPTH="${ENCODER_DEPTH:-8}"

REPORT_TO="${REPORT_TO:-wandb}"
WANDB_MODE="${WANDB_MODE:-offline}"
WANDB_DIR="${WANDB_DIR:-$OUTPUT_DIR/wandb}"
MIXED_PRECISION="${MIXED_PRECISION:-fp16}"

HF_HOME="${HF_HOME:-/data/liuchunfa/2026qjx/hf_cache}"
TORCH_HOME="${TORCH_HOME:-/data/liuchunfa/2026qjx/torch_cache}"

AUTO_RESUME="${AUTO_RESUME:-1}"
RESUME_STEP="${RESUME_STEP:-}"

if [[ ! -x "$ACCELERATE_BIN" ]]; then
    echo "Missing accelerate binary: $ACCELERATE_BIN" >&2
    exit 1
fi
if [[ ! -x "$PYTHON_BIN" ]]; then
    echo "Missing python binary: $PYTHON_BIN" >&2
    exit 1
fi
if [[ ! -d "$REPO_DIR" ]]; then
    echo "Missing REPA repo: $REPO_DIR" >&2
    exit 1
fi
if [[ ! -d "$DATA_DIR/images" || ! -d "$DATA_DIR/vae-sd" ]]; then
    echo "Missing official REPA dataset layout under: $DATA_DIR" >&2
    echo "Expected directories: $DATA_DIR/images and $DATA_DIR/vae-sd" >&2
    exit 1
fi
if [[ ! -d "$DINO_MODEL_DIR" ]]; then
    echo "Missing local DINOv3 directory: $DINO_MODEL_DIR" >&2
    exit 1
fi
if [[ ! -d "$VAE_MODEL_DIR" ]]; then
    echo "Missing local VAE directory: $VAE_MODEL_DIR" >&2
    exit 1
fi

mkdir -p "$OUTPUT_DIR" "$WANDB_DIR" "$HF_HOME" "$TORCH_HOME"
export CUDA_VISIBLE_DEVICES HF_HOME TORCH_HOME WANDB_MODE WANDB_DIR

if [[ -z "$RESUME_STEP" && "$AUTO_RESUME" != "0" ]]; then
    CHECKPOINT_GLOB="$OUTPUT_DIR/$EXP_NAME/checkpoints/"*.pt
    shopt -s nullglob
    checkpoints=($CHECKPOINT_GLOB)
    shopt -u nullglob
    if (( ${#checkpoints[@]} > 0 )); then
        latest_ckpt="$(printf '%s\n' "${checkpoints[@]}" | sort | tail -n 1)"
        latest_name="$(basename "$latest_ckpt" .pt)"
        if [[ "$latest_name" =~ ^[0-9]+$ ]]; then
            RESUME_STEP="$((10#$latest_name))"
        fi
    fi
fi

echo "REPA repo: $REPO_DIR"
echo "Dataset dir: $DATA_DIR"
echo "Output dir: $OUTPUT_DIR"
echo "Experiment name: $EXP_NAME"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "Num processes: $NUM_PROCESSES"
echo "Global batch size: $BATCH_SIZE"
echo "Max train steps: $MAX_TRAIN_STEPS"
echo "Teacher: $ENC_TYPE"
echo "DINO model dir: $DINO_MODEL_DIR"
echo "VAE model dir: $VAE_MODEL_DIR"
echo "Proj coeff: $PROJ_COEFF"
echo "Proj diff schedule: $PROJ_DIFF_SCHEDULE"
echo "Encoder depth: $ENCODER_DEPTH"
echo "WANDB_MODE: $WANDB_MODE"
if [[ -n "$RESUME_STEP" ]]; then
    echo "Resuming from step: $RESUME_STEP"
fi

echo "Prefetching local REPA VAE..."
"$PYTHON_BIN" -c "from diffusers.models import AutoencoderKL; AutoencoderKL.from_pretrained('$VAE_MODEL_DIR', local_files_only=True); print('vae_cached')"

cd "$REPO_DIR"

CMD=(
    "$ACCELERATE_BIN" launch
    --multi_gpu
    --num_processes "$NUM_PROCESSES"
    --num_machines 1
    --mixed_precision "$MIXED_PRECISION"
    --main_process_port "$MAIN_PROCESS_PORT"
    train.py
    --report-to="$REPORT_TO"
    --allow-tf32
    --mixed-precision="$MIXED_PRECISION"
    --seed="$SEED"
    --path-type="linear"
    --prediction="v"
    --weighting="uniform"
    --model="$MODEL"
    --enc-type="$ENC_TYPE"
    --dino-model-dir="$DINO_MODEL_DIR"
    --vae-model-dir="$VAE_MODEL_DIR"
    --proj-coeff="$PROJ_COEFF"
    --proj-diff-schedule="$PROJ_DIFF_SCHEDULE"
    --encoder-depth="$ENCODER_DEPTH"
    --output-dir="$OUTPUT_DIR"
    --exp-name="$EXP_NAME"
    --data-dir="$DATA_DIR"
    --batch-size="$BATCH_SIZE"
    --max-train-steps="$MAX_TRAIN_STEPS"
    --checkpointing-steps="$CHECKPOINTING_STEPS"
    --sampling-steps="$SAMPLING_STEPS"
    --num-workers="$NUM_WORKERS"
)

if [[ -n "$RESUME_STEP" ]]; then
    CMD+=(--resume-step="$RESUME_STEP")
fi

exec "${CMD[@]}"
