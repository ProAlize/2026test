#!/usr/bin/env bash
# download_s3a_sit_deps.sh — Download all external dependencies for S3A-SiT.
# Usage: bash download_s3a_sit_deps.sh [BASE_DIR]
#
# Downloads: SiT repo, DINOv2 repo + weights, VAE model.
# Skips already-existing files.
set -euo pipefail

BASE_DIR="${1:-$(pwd)/deps}"
echo "Installing S3A-SiT dependencies to: $BASE_DIR"
mkdir -p "$BASE_DIR"

# ────────────────────────────────────────────────────────────────
# 1. SiT Repository (willisma/SiT)
# ────────────────────────────────────────────────────────────────
SIT_REPO_DIR="$BASE_DIR/SiT"
if [[ -f "$SIT_REPO_DIR/models.py" ]] && [[ -d "$SIT_REPO_DIR/transport" ]]; then
    echo "[SKIP] SiT repo already exists at $SIT_REPO_DIR"
else
    echo "[CLONE] SiT repo → $SIT_REPO_DIR"
    git clone --depth 1 https://github.com/willisma/SiT.git "$SIT_REPO_DIR"
fi

# ────────────────────────────────────────────────────────────────
# 2. DINOv2 Repository (facebookresearch/dinov2)
# ────────────────────────────────────────────────────────────────
DINOV2_REPO_DIR="$BASE_DIR/dinov2"
if [[ -f "$DINOV2_REPO_DIR/hubconf.py" ]]; then
    echo "[SKIP] DINOv2 repo already exists at $DINOV2_REPO_DIR"
else
    echo "[CLONE] DINOv2 repo → $DINOV2_REPO_DIR"
    git clone --depth 1 https://github.com/facebookresearch/dinov2.git "$DINOV2_REPO_DIR"
fi

# ────────────────────────────────────────────────────────────────
# 3. DINOv2 ViT-B/14 Weights (~330 MB)
# ────────────────────────────────────────────────────────────────
DINOV2_WEIGHT_DIR="$BASE_DIR/dinov2_weights"
DINOV2_WEIGHT_PATH="$DINOV2_WEIGHT_DIR/dinov2_vitb14_pretrain.pth"
mkdir -p "$DINOV2_WEIGHT_DIR"
if [[ -f "$DINOV2_WEIGHT_PATH" ]]; then
    echo "[SKIP] DINOv2 ViT-B/14 weights already exist"
else
    echo "[DOWNLOAD] DINOv2 ViT-B/14 weights → $DINOV2_WEIGHT_PATH"
    wget -q --show-progress -O "$DINOV2_WEIGHT_PATH" \
        https://dl.fbaipublicfiles.com/dinov2/dinov2_vitb14/dinov2_vitb14_pretrain.pth
fi

# ────────────────────────────────────────────────────────────────
# 4. Stable Diffusion VAE (sd-vae-ft-ema)
# ────────────────────────────────────────────────────────────────
VAE_MODEL_DIR="$BASE_DIR/vae"
if [[ -f "$VAE_MODEL_DIR/config.json" ]]; then
    echo "[SKIP] VAE model already exists at $VAE_MODEL_DIR"
else
    echo "[DOWNLOAD] Stable Diffusion VAE → $VAE_MODEL_DIR"
    python3 -c "
from diffusers.models import AutoencoderKL
vae = AutoencoderKL.from_pretrained('stabilityai/sd-vae-ft-ema')
vae.save_pretrained('$VAE_MODEL_DIR')
print('VAE saved.')
"
fi

# ────────────────────────────────────────────────────────────────
# Summary
# ────────────────────────────────────────────────────────────────
echo ""
echo "════════════════════════════════════════════════════════"
echo "  S3A-SiT dependencies ready!"
echo "════════════════════════════════════════════════════════"
echo ""
echo "Set these env vars before launching:"
echo ""
echo "  export SIT_REPO_DIR=$SIT_REPO_DIR"
echo "  export DINOV2_REPO_DIR=$DINOV2_REPO_DIR"
echo "  export DINOV2_WEIGHT_PATH=$DINOV2_WEIGHT_PATH"
echo "  export VAE_MODEL_DIR=$VAE_MODEL_DIR"
echo ""
echo "Then run:"
echo "  export DATA_PATH=/path/to/imagenet/train"
echo "  bash run_s3a_sit_multisource_dinov2.sh"
