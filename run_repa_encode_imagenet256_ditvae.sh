#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="${REPO_DIR:-$ROOT_DIR/project/REPA}"
PREPROCESS_DIR="$REPO_DIR/preprocessing"

ENV_PREFIX="${ENV_PREFIX:-/home/liuchunfa/anaconda3/envs/DiT}"
PYTHON_BIN="${PYTHON_BIN:-$ENV_PREFIX/bin/python}"

SOURCE_IMAGES_DIR="${SOURCE_IMAGES_DIR:-/data/liuchunfa/2026qjx/repa_imagenet256_official/images}"
TARGET_ROOT="${TARGET_ROOT:-/data/liuchunfa/2026qjx/repa_imagenet256_ditvae}"
TARGET_IMAGES_DIR="${TARGET_IMAGES_DIR:-$TARGET_ROOT/images}"
TARGET_LATENTS_DIR="${TARGET_LATENTS_DIR:-$TARGET_ROOT/vae-sd}"
MODEL_URL="${MODEL_URL:-/home/liuchunfa/.cache/modelscope/hub/models/facebook/DiT-XL-2-256/vae}"

HF_HOME="${HF_HOME:-/data/liuchunfa/2026qjx/hf_cache}"
TORCH_HOME="${TORCH_HOME:-/data/liuchunfa/2026qjx/torch_cache}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

if [[ ! -d "$PREPROCESS_DIR" ]]; then
    echo "Missing preprocessing directory: $PREPROCESS_DIR" >&2
    exit 1
fi
if [[ ! -f "$SOURCE_IMAGES_DIR/dataset.json" ]]; then
    echo "Missing converted official REPA images dataset: $SOURCE_IMAGES_DIR" >&2
    exit 1
fi
if [[ ! -d "$MODEL_URL" ]]; then
    echo "Missing local VAE directory: $MODEL_URL" >&2
    exit 1
fi

mkdir -p "$TARGET_ROOT" "$HF_HOME" "$TORCH_HOME"
export HF_HOME TORCH_HOME CUDA_VISIBLE_DEVICES

if [[ ! -e "$TARGET_IMAGES_DIR" ]]; then
    ln -s "$SOURCE_IMAGES_DIR" "$TARGET_IMAGES_DIR"
fi

cd "$PREPROCESS_DIR"

echo "Encoding official REPA images with local DiT VAE..."
echo "Source images: $SOURCE_IMAGES_DIR"
echo "Target root: $TARGET_ROOT"
echo "Target latents: $TARGET_LATENTS_DIR"
echo "Model URL: $MODEL_URL"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"

"$PYTHON_BIN" dataset_tools.py encode \
    --model-url="$MODEL_URL" \
    --source="$SOURCE_IMAGES_DIR" \
    --dest="$TARGET_LATENTS_DIR"

echo "Official REPA VAE encoding complete."
