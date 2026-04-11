#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="${REPO_DIR:-$ROOT_DIR/project/REPA}"
PREPROCESS_DIR="$REPO_DIR/preprocessing"

ENV_PREFIX="${ENV_PREFIX:-/home/liuchunfa/anaconda3/envs/DiT}"
PYTHON_BIN="${PYTHON_BIN:-$ENV_PREFIX/bin/python}"

RAW_IMAGENET_DIR="${RAW_IMAGENET_DIR:-/data/liuchunfa/2026qjx/ILSVRC/Data/CLS-LOC/train}"
TARGET_ROOT="${TARGET_ROOT:-/data/liuchunfa/2026qjx/repa_imagenet256_official}"
IMAGES_DIR="${IMAGES_DIR:-$TARGET_ROOT/images}"
LATENTS_DIR="${LATENTS_DIR:-$TARGET_ROOT/vae-sd}"
RESOLUTION="${RESOLUTION:-256x256}"
MODEL_URL="${MODEL_URL:-/home/liuchunfa/.cache/modelscope/hub/models/facebook/DiT-XL-2-256/vae}"
MAX_IMAGES="${MAX_IMAGES:-}"

HF_HOME="${HF_HOME:-/data/liuchunfa/2026qjx/hf_cache}"
TORCH_HOME="${TORCH_HOME:-/data/liuchunfa/2026qjx/torch_cache}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

if [[ ! -d "$PREPROCESS_DIR" ]]; then
    echo "Missing preprocessing directory: $PREPROCESS_DIR" >&2
    exit 1
fi
if [[ ! -d "$RAW_IMAGENET_DIR" ]]; then
    echo "Missing raw ImageNet directory: $RAW_IMAGENET_DIR" >&2
    exit 1
fi

mkdir -p "$TARGET_ROOT" "$HF_HOME" "$TORCH_HOME"
export HF_HOME TORCH_HOME CUDA_VISIBLE_DEVICES

MAX_IMAGE_ARGS=()
if [[ -n "$MAX_IMAGES" ]]; then
    MAX_IMAGE_ARGS+=(--max-images "$MAX_IMAGES")
fi

cd "$PREPROCESS_DIR"

echo "REPA preprocessing source: $RAW_IMAGENET_DIR"
echo "REPA preprocessing target: $TARGET_ROOT"
echo "REPA preprocessing images dir: $IMAGES_DIR"
echo "REPA preprocessing latents dir: $LATENTS_DIR"
echo "HF cache: $HF_HOME"
echo "Torch cache: $TORCH_HOME"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"

if [[ ! -f "$IMAGES_DIR/dataset.json" ]]; then
    echo "Running official REPA image conversion..."
    "$PYTHON_BIN" dataset_tools.py convert \
        --source="$RAW_IMAGENET_DIR" \
        --dest="$IMAGES_DIR" \
        --resolution="$RESOLUTION" \
        --transform=center-crop-dhariwal \
        "${MAX_IMAGE_ARGS[@]}"
else
    echo "Skipping conversion: found $IMAGES_DIR/dataset.json"
fi

if [[ ! -f "$LATENTS_DIR/dataset.json" ]]; then
    echo "Running official REPA VAE encoding..."
    "$PYTHON_BIN" dataset_tools.py encode \
        --model-url="$MODEL_URL" \
        --source="$IMAGES_DIR" \
        --dest="$LATENTS_DIR" \
        "${MAX_IMAGE_ARGS[@]}"
else
    echo "Skipping encoding: found $LATENTS_DIR/dataset.json"
fi

echo "Official REPA preprocessing complete."
