#!/usr/bin/env bash
# run_generate_repa.sh
# 完全对齐 REPA generate.py 的采样启动脚本
# 用法：
#   bash run_generate_repa.sh /path/to/checkpoint.pt
#   或：
#   CKPT_PATH=/path/to/checkpoint.pt bash run_generate_repa.sh
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ── 环境路径 ─────────────────────────────────────────────────────────────────
ENV_PREFIX="${ENV_PREFIX:-/mnt/tidal-alsh01/dataset/redaigc/yuantianshuo/tool/miniconda3/envs/dit}"
TORCHRUN_BIN="${TORCHRUN_BIN:-$ENV_PREFIX/bin/torchrun}"

# ── Checkpoint（位置参数优先，其次环境变量） ──────────────────────────────────
CKPT_PATH="${1:-${CKPT_PATH:-}}"
if [[ -z "$CKPT_PATH" ]]; then
    echo "Usage: $0 /path/to/checkpoint.pt" >&2
    exit 1
fi

# ── VAE 路径 ─────────────────────────────────────────────────────────────────
VAE_MODEL_DIR="${VAE_MODEL_DIR:-/mnt/tidal-alsh01/dataset/redaigc/yuantianshuo/2026/models/DiT-XL-2-256/vae}"

# ── GPU 配置 ─────────────────────────────────────────────────────────────────
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-4,5,6,7}"
NPROC_PER_NODE="${NPROC_PER_NODE:-4}"

# ── 采样超参（与 REPA 默认值对齐） ───────────────────────────────────────────
NUM_FID_SAMPLES="${NUM_FID_SAMPLES:-50000}"
PER_PROC_BATCH="${PER_PROC_BATCH:-64}"
NUM_STEPS="${NUM_STEPS:-250}"
CFG_SCALE="${CFG_SCALE:-1.5}"
GLOBAL_SEED="${GLOBAL_SEED:-0}"

# ── 模型超参（空 = 从 checkpoint 自动推断） ───────────────────────────────────
MODEL="${MODEL:-}"
IMAGE_SIZE="${IMAGE_SIZE:-}"
NUM_CLASSES="${NUM_CLASSES:-}"

# ── 输出目录（与 run_eval_fid_offline.sh 风格对齐：放在 ckpt 同级目录下） ────
CKPT_DIR="$(cd "$(dirname "$CKPT_PATH")" && pwd)"
CKPT_STEM="$(basename "${CKPT_PATH%.pt}")"
SAMPLE_DIR="${SAMPLE_DIR:-$CKPT_DIR/samples/${CKPT_STEM}}"

# ── Sanity checks ─────────────────────────────────────────────────────────────
if [[ ! -x "$TORCHRUN_BIN" ]]; then
    echo "[ERROR] torchrun not found: $TORCHRUN_BIN" >&2; exit 1
fi
if [[ ! -f "$CKPT_PATH" ]]; then
    echo "[ERROR] Checkpoint not found: $CKPT_PATH" >&2; exit 1
fi
if [[ ! -f "$ROOT_DIR/generate_repa.py" ]]; then
    echo "[ERROR] generate_repa.py not found in $ROOT_DIR" >&2; exit 1
fi
if [[ ! -d "$VAE_MODEL_DIR" ]]; then
    echo "[ERROR] VAE directory not found: $VAE_MODEL_DIR" >&2; exit 1
fi

mkdir -p "$SAMPLE_DIR"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
LAUNCH_LOG="${LAUNCH_LOG:-$SAMPLE_DIR/launch_${TIMESTAMP}.log}"

# ── 可选参数拼接 ──────────────────────────────────────────────────────────────
OPTIONAL_ARGS=()
[[ -n "$MODEL"       ]] && OPTIONAL_ARGS+=(--model       "$MODEL")
[[ -n "$IMAGE_SIZE"  ]] && OPTIONAL_ARGS+=(--image-size  "$IMAGE_SIZE")
[[ -n "$NUM_CLASSES" ]] && OPTIONAL_ARGS+=(--num-classes "$NUM_CLASSES")

# ── 启动信息 ──────────────────────────────────────────────────────────────────
cd "$ROOT_DIR"
echo "========================================================"
echo "Starting REPA-aligned sampling"
echo "Checkpoint       : $CKPT_PATH"
echo "Sample dir       : $SAMPLE_DIR"
echo "VAE model dir    : $VAE_MODEL_DIR"
echo "Torchrun         : $TORCHRUN_BIN"
echo "GPUs             : $CUDA_VISIBLE_DEVICES"
echo "Num GPUs         : $NPROC_PER_NODE"
echo "Num FID samples  : $NUM_FID_SAMPLES"
echo "Per-proc batch   : $PER_PROC_BATCH"
echo "Sampling steps   : $NUM_STEPS"
echo "CFG scale        : $CFG_SCALE"
echo "Global seed      : $GLOBAL_SEED"
echo "Launch log       : $LAUNCH_LOG"
[[ -n "$MODEL"       ]] && echo "Model            : $MODEL"      || echo "Model            : (from ckpt)"
[[ -n "$IMAGE_SIZE"  ]] && echo "Image size       : $IMAGE_SIZE" || echo "Image size       : (from ckpt)"
[[ -n "$NUM_CLASSES" ]] && echo "Num classes      : $NUM_CLASSES"|| echo "Num classes      : (from ckpt)"
echo "========================================================"

# ── 启动采样 ──────────────────────────────────────────────────────────────────
env CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" \
    "$TORCHRUN_BIN" \
        --standalone \
        --nnodes=1 \
        --nproc_per_node="$NPROC_PER_NODE" \
    "$ROOT_DIR/generate_repa.py" \
        --ckpt                "$CKPT_PATH"       \
        --sample-dir          "$SAMPLE_DIR"      \
        --num-fid-samples     "$NUM_FID_SAMPLES" \
        --per-proc-batch-size "$PER_PROC_BATCH"  \
        --num-sampling-steps  "$NUM_STEPS"       \
        --cfg-scale           "$CFG_SCALE"       \
        --global-seed         "$GLOBAL_SEED"     \
        --vae-model-dir       "$VAE_MODEL_DIR"   \
        "${OPTIONAL_ARGS[@]}"                    \
    2>&1 | tee "$LAUNCH_LOG"

# ── 示例调用 ──────────────────────────────────────────────────────────────────
# bash run_generate_repa.sh \
#   /mnt/tidal-alsh01/dataset/redaigc/yuantianshuo/2026/results/\
# dit_xl_sasa_stepdecay_80k/010-DiT-XL-2-sasa-lam0.1-trainlinear_decay-nodiffweight/\
# checkpoints/0080000.pt
