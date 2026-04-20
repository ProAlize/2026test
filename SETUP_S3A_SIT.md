# S3A-SiT Setup Guide

S3A-SiT = SiT-XL/2 backbone + S3A alignment (DINOv2 + EMA self-source).

## Prerequisites

- 4× NVIDIA GPU (A100 40GB or better recommended)
- CUDA 11.8+ / 12.1+
- ImageNet-1K training data (ImageFolder format)

## 1. Create Conda Environment

```bash
conda env create -f environment_s3a_sit.yml
conda activate s3a-sit
```

Or install into an existing environment:

```bash
pip install torch>=2.1.0 torchvision>=0.16.0 diffusers>=0.24.0 \
    accelerate>=0.25.0 timm>=0.9.0 scipy scikit-learn tqdm
```

## 2. Clone SiT Repository

S3A-SiT depends on `models.py` and `transport/` from the SiT repo.

```bash
# Choose a directory (default used by the launcher script)
SIT_REPO_DIR=/path/to/SiT

git clone https://github.com/willisma/SiT.git "$SIT_REPO_DIR"
```

Verify the expected files exist:

```bash
ls "$SIT_REPO_DIR/models.py" "$SIT_REPO_DIR/transport/"
```

> **Important**: The workspace contains a `models.py` for DiT that will shadow SiT's
> if PYTHONPATH is wrong. The launcher script handles this via
> `export PYTHONPATH="$SIT_REPO_DIR:$PYTHONPATH"`.

## 3. Download DINOv2 Weights

```bash
DINOV2_REPO_DIR=/path/to/dinov2
DINOV2_WEIGHT_DIR=/path/to/dinov2_weights

# Clone the DINOv2 repo (needed for hubconf.py)
git clone https://github.com/facebookresearch/dinov2.git "$DINOV2_REPO_DIR"

# Download ViT-B/14 pretrained weights
mkdir -p "$DINOV2_WEIGHT_DIR"
wget -O "$DINOV2_WEIGHT_DIR/dinov2_vitb14_pretrain.pth" \
    https://dl.fbaipublicfiles.com/dinov2/dinov2_vitb14/dinov2_vitb14_pretrain.pth
```

Optional larger variants:

```bash
# ViT-L/14 (1024-d)
wget -O "$DINOV2_WEIGHT_DIR/dinov2_vitl14_pretrain.pth" \
    https://dl.fbaipublicfiles.com/dinov2/dinov2_vitl14/dinov2_vitl14_pretrain.pth

# ViT-g/14 (1536-d)
wget -O "$DINOV2_WEIGHT_DIR/dinov2_vitg14_pretrain.pth" \
    https://dl.fbaipublicfiles.com/dinov2/dinov2_vitg14/dinov2_vitg14_pretrain.pth
```

## 4. Download VAE

The Stable Diffusion VAE is used for latent encoding. Either:

**Option A**: Download from HuggingFace (automatic, needs internet):
```bash
# The script auto-downloads if --vae-model-dir is not set.
# Uses: stabilityai/sd-vae-ft-ema
```

**Option B**: Pre-download for offline use:
```bash
VAE_MODEL_DIR=/path/to/vae
python -c "
from diffusers.models import AutoencoderKL
vae = AutoencoderKL.from_pretrained('stabilityai/sd-vae-ft-ema')
vae.save_pretrained('$VAE_MODEL_DIR')
print('VAE saved to $VAE_MODEL_DIR')
"
```

## 5. Prepare ImageNet Data

Data must be in PyTorch ImageFolder format:

```
/path/to/imagenet/train/
    n01440764/
        ILSVRC2012_val_00000293.JPEG
        ...
    n01443537/
        ...
    ...
```

## 6. Configure and Launch

Edit env vars in the launcher or export them:

```bash
export SIT_REPO_DIR=/path/to/SiT
export DATA_PATH=/path/to/imagenet/train
export DINOV2_REPO_DIR=/path/to/dinov2
export DINOV2_WEIGHT_PATH=/path/to/dinov2_weights/dinov2_vitb14_pretrain.pth
export VAE_MODEL_DIR=/path/to/vae
export RESULTS_DIR=/path/to/results
export CUDA_VISIBLE_DEVICES=0,1,2,3

bash run_s3a_sit_multisource_dinov2.sh
```

## 7. Key Differences from S3A-DiT

| Aspect | S3A-DiT | S3A-SiT |
|--------|---------|---------|
| Backbone | DiT-XL/2 (DDPM) | SiT-XL/2 (Transport/Interpolant) |
| Time convention | Discrete t∈{0,...,999} | Continuous t∈[0,1] |
| Extra dependency | `diffusion/` (in-repo) | `transport/` (external SiT repo) |
| Loss computation | `diffusion.training_losses()` | `transport.training_losses()` |
| EMA cross-timestep | `t_ema = max(0, t-k)` | `t_ema = min(1, t+k)` |
| Constraint | None | `--path-type` must be `linear` when using EMA source |

## 8. Directory Structure (After Setup)

```
2026test/                          # This repo
├── train_s3a_sit_multisource_dinov2.py   # S3A-SiT training script
├── run_s3a_sit_multisource_dinov2.sh     # Launcher
├── environment_s3a_sit.yml               # Conda environment
├── SETUP_S3A_SIT.md                      # This file
├── download_s3a_sit_deps.sh              # Dependency download helper
├── model_sasa.py                         # DiT model (not used by SiT)
└── ...

/path/to/SiT/                     # External SiT repo (willisma/SiT)
├── models.py                     # SiT_models dict
├── transport/                    # Transport/interpolant framework
│   ├── __init__.py
│   ├── transport.py
│   └── ...
└── ...

/path/to/dinov2/                  # DINOv2 repo (facebookresearch/dinov2)
├── hubconf.py
└── ...

/path/to/dinov2_weights/
└── dinov2_vitb14_pretrain.pth    # ~330 MB

/path/to/vae/                     # Stable Diffusion VAE
├── config.json
├── diffusion_pytorch_model.safetensors
└── ...
```
