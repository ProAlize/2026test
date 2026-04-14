#!/usr/bin/env bash
# DiT-XL/2 + S3A (DINOv2 + EMA-self source)
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

ENV_PREFIX="${ENV_PREFIX:-/mnt/tidal-alsh01/dataset/redaigc/yuantianshuo/tool/miniconda3/envs/dit}"
TORCHRUN_BIN="${TORCHRUN_BIN:-$ENV_PREFIX/bin/torchrun}"
PYTHON_BIN="${PYTHON_BIN:-$ENV_PREFIX/bin/python}"

DATA_PATH="${DATA_PATH:-/data/temp/ILSVRC/Data/CLS-LOC/train}"
RESULTS_DIR="${RESULTS_DIR:-/mnt/tidal-alsh01/dataset/redaigc/yuantianshuo/2026/results/dit_xl_s3a_dinov2_400k}"
VAE_MODEL_DIR="${VAE_MODEL_DIR:-/mnt/tidal-alsh01/dataset/redaigc/yuantianshuo/2026/models/DiT-XL-2-256/vae}"

DINOV2_REPO_DIR="${DINOV2_REPO_DIR:-/mnt/tidal-alsh01/dataset/redaigc/yuantianshuo/tmp/dinov2}"
DINOV2_WEIGHT_PATH="${DINOV2_WEIGHT_PATH:-/mnt/tidal-alsh01/dataset/redaigc/yuantianshuo/tmp/dinov2_weights/dinov2_vitb14_pretrain.pth}"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
NPROC_PER_NODE="${NPROC_PER_NODE:-4}"
GLOBAL_SEED="${GLOBAL_SEED:-0}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-256}"
NUM_WORKERS="${NUM_WORKERS:-4}"
EPOCHS="${EPOCHS:-1400}"
MAX_STEPS="${MAX_STEPS:-400000}"
CKPT_EVERY="${CKPT_EVERY:-1000}"
LOG_EVERY="${LOG_EVERY:-500}"
DISK_BUDGET_GB="${DISK_BUDGET_GB:-4000}"

S3A_LAMBDA="${S3A_LAMBDA:-0.1}"
S3A_LAYER_INDICES="${S3A_LAYER_INDICES:-auto}"
S3A_LAYER_WEIGHT_MODE="${S3A_LAYER_WEIGHT_MODE:-mid_focus}"
S3A_LAYER_WEIGHTS="${S3A_LAYER_WEIGHTS:-}"
S3A_TRAIN_SCHEDULE="${S3A_TRAIN_SCHEDULE:-cosine_decay}"
S3A_SCHEDULE_STEPS="${S3A_SCHEDULE_STEPS:-40000}"
S3A_DIFF_SCHEDULE="${S3A_DIFF_SCHEDULE:-cosine}"
S3A_FEAT_WEIGHT="${S3A_FEAT_WEIGHT:-1.0}"
S3A_ATTN_WEIGHT="${S3A_ATTN_WEIGHT:-0.5}"
S3A_SPATIAL_WEIGHT="${S3A_SPATIAL_WEIGHT:-0.5}"
S3A_SELF_WARMUP_STEPS="${S3A_SELF_WARMUP_STEPS:-5000}"
S3A_DINO_ALPHA_FLOOR="${S3A_DINO_ALPHA_FLOOR:-0.1}"
S3A_DINO_ALPHA_FLOOR_STEPS="${S3A_DINO_ALPHA_FLOOR_STEPS:-8000}"
S3A_PROBE_EVERY="${S3A_PROBE_EVERY:-10}"
S3A_UTILITY_PROBE_MODE="${S3A_UTILITY_PROBE_MODE:-uniform}"
S3A_GATE_PATIENCE="${S3A_GATE_PATIENCE:-500}"
S3A_GATE_REOPEN_PATIENCE="${S3A_GATE_REOPEN_PATIENCE:-200}"
S3A_GATE_UTILITY_OFF_THRESHOLD="${S3A_GATE_UTILITY_OFF_THRESHOLD:-0.002}"
S3A_GATE_UTILITY_ON_THRESHOLD="${S3A_GATE_UTILITY_ON_THRESHOLD:-0.005}"
S3A_GATE_UTILITY_EMA_MOMENTUM="${S3A_GATE_UTILITY_EMA_MOMENTUM:-0.9}"
S3A_COLLAPSE_ALPHA_THRESHOLD="${S3A_COLLAPSE_ALPHA_THRESHOLD:-0.05}"
S3A_COLLAPSE_SELF_THRESHOLD="${S3A_COLLAPSE_SELF_THRESHOLD:-0.90}"
S3A_COLLAPSE_MARGIN="${S3A_COLLAPSE_MARGIN:-0.01}"
S3A_COLLAPSE_WINDOWS="${S3A_COLLAPSE_WINDOWS:-3}"
S3A_COLLAPSE_UTILITY_THRESHOLD="${S3A_COLLAPSE_UTILITY_THRESHOLD:-0.0}"
S3A_COLLAPSE_AUTO_MITIGATE="${S3A_COLLAPSE_AUTO_MITIGATE:-1}"
S3A_COLLAPSE_MITIGATE_WINDOWS="${S3A_COLLAPSE_MITIGATE_WINDOWS:-3}"
S3A_COLLAPSE_MITIGATE_COOLDOWN_WINDOWS="${S3A_COLLAPSE_MITIGATE_COOLDOWN_WINDOWS:-6}"
S3A_TRAINABLE_EMA_ADAPTERS="${S3A_TRAINABLE_EMA_ADAPTERS:-0}"
RESUME_CKPT="${RESUME_CKPT:-}"
ALLOW_MISSING_MANIFEST="${ALLOW_MISSING_MANIFEST:-0}"

if [[ ! -x "$TORCHRUN_BIN" ]]; then
    echo "[ERROR] torchrun not found: $TORCHRUN_BIN" >&2; exit 1
fi
if [[ ! -x "$PYTHON_BIN" ]]; then
    echo "[ERROR] python not found: $PYTHON_BIN" >&2; exit 1
fi

GPU_COUNT=$(echo "$CUDA_VISIBLE_DEVICES" | tr ',' '\n' | wc -l | tr -d ' ')
if [[ "$GPU_COUNT" -lt "$NPROC_PER_NODE" ]]; then
    echo "[ERROR] NPROC_PER_NODE=$NPROC_PER_NODE exceeds visible GPU count=$GPU_COUNT" >&2
    exit 1
fi

for required_path in \
    "$ROOT_DIR/train_s3a_multisource_dinov2.py" \
    "$DATA_PATH" \
    "$VAE_MODEL_DIR" \
    "$DINOV2_REPO_DIR" \
    "$DINOV2_REPO_DIR/hubconf.py" \
    "$DINOV2_WEIGHT_PATH"; do
    if [[ ! -e "$required_path" ]]; then
        echo "[ERROR] Required path not found: $required_path" >&2; exit 1
    fi
done

mkdir -p "$RESULTS_DIR"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
LAUNCH_LOG="${LAUNCH_LOG:-$RESULTS_DIR/launch_s3a_${TIMESTAMP}.log}"
cd "$ROOT_DIR"

PARAM_COUNT="$("$PYTHON_BIN" - <<'PY'
from model_sasa import DiT_models
m = DiT_models["DiT-XL/2"](input_size=32, num_classes=1000)
print(sum(p.numel() for p in m.parameters()))
PY
)"
EST_CKPT_GB="$("$PYTHON_BIN" - <<PY
params = int("$PARAM_COUNT")
print(f"{(params * 16) / (1024**3):.2f}")
PY
)"
EST_NUM_CKPTS="$("$PYTHON_BIN" - <<PY
import math
max_steps = int("$MAX_STEPS")
ckpt_every = int("$CKPT_EVERY")
print(max(1, math.ceil(max_steps / ckpt_every)))
PY
)"
EST_TOTAL_GB="$("$PYTHON_BIN" - <<PY
print(f"{float("$EST_CKPT_GB") * int("$EST_NUM_CKPTS"):.2f}")
PY
)"
"$PYTHON_BIN" - <<PY
budget = float("$DISK_BUDGET_GB")
proj = float("$EST_TOTAL_GB")
if proj > budget:
    raise SystemExit(
        f"[ERROR] Projected checkpoint footprint {proj:.2f} GiB exceeds DISK_BUDGET_GB={budget:.2f}. "
        "Increase CKPT_EVERY or retention policy before launch."
    )
PY

echo "====== DiT-XL/2 + S3A (DINOv2 + EMA-self) ======"
echo "Torchrun         : $TORCHRUN_BIN"
echo "Python           : $PYTHON_BIN"
echo "GPUs             : $CUDA_VISIBLE_DEVICES  (nproc=$NPROC_PER_NODE)"
echo "Global seed      : $GLOBAL_SEED"
echo "Results dir      : $RESULTS_DIR"
echo "Launch log       : $LAUNCH_LOG"
echo "Max steps        : $MAX_STEPS"
echo "Global batch     : $GLOBAL_BATCH_SIZE"
echo "Ckpt every       : $CKPT_EVERY steps"
echo "Log every        : $LOG_EVERY steps"
echo "Disk budget      : $DISK_BUDGET_GB GiB"
echo "Est ckpt size    : $EST_CKPT_GB GiB"
echo "Est #checkpoints : $EST_NUM_CKPTS"
echo "Est total ckpt   : $EST_TOTAL_GB GiB"
echo "S3A lambda       : $S3A_LAMBDA"
echo "S3A layer indices: $S3A_LAYER_INDICES"
echo "Train schedule   : $S3A_TRAIN_SCHEDULE (steps=$S3A_SCHEDULE_STEPS)"
echo "Diff schedule    : $S3A_DIFF_SCHEDULE"
echo "DINO alpha floor : $S3A_DINO_ALPHA_FLOOR (steps=$S3A_DINO_ALPHA_FLOOR_STEPS)"
echo "Probe every      : $S3A_PROBE_EVERY"
echo "Probe mode       : $S3A_UTILITY_PROBE_MODE"
echo "Gate patience    : off=$S3A_GATE_PATIENCE, reopen=$S3A_GATE_REOPEN_PATIENCE"
echo "Gate utility     : off=$S3A_GATE_UTILITY_OFF_THRESHOLD, on=$S3A_GATE_UTILITY_ON_THRESHOLD"
echo "Gate EMA mom     : $S3A_GATE_UTILITY_EMA_MOMENTUM"
echo "Collapse         : alpha<$S3A_COLLAPSE_ALPHA_THRESHOLD, self>$S3A_COLLAPSE_SELF_THRESHOLD, u>$S3A_COLLAPSE_UTILITY_THRESHOLD, margin=$S3A_COLLAPSE_MARGIN, windows=$S3A_COLLAPSE_WINDOWS"
echo "Mitigation       : auto=$S3A_COLLAPSE_AUTO_MITIGATE, hold=$S3A_COLLAPSE_MITIGATE_WINDOWS, cooldown=$S3A_COLLAPSE_MITIGATE_COOLDOWN_WINDOWS"
echo "Train EMA adapter: $S3A_TRAINABLE_EMA_ADAPTERS"
echo "DINOv2 repo      : $DINOV2_REPO_DIR"
echo "DINOv2 weight    : $DINOV2_WEIGHT_PATH"
echo "VAE              : $VAE_MODEL_DIR"
[[ -n "$S3A_LAYER_WEIGHTS" ]] && echo "S3A layer weights: $S3A_LAYER_WEIGHTS"
[[ -n "$RESUME_CKPT" ]] && echo "Resume checkpoint: $RESUME_CKPT"
echo "=================================================="

OPTIONAL_ARGS=()
if [[ -n "$S3A_LAYER_WEIGHTS" ]]; then
    OPTIONAL_ARGS+=(--s3a-layer-weights "$S3A_LAYER_WEIGHTS")
fi
if [[ -n "$RESUME_CKPT" ]]; then
    OPTIONAL_ARGS+=(--resume "$RESUME_CKPT")
fi
if [[ "$ALLOW_MISSING_MANIFEST" == "1" ]]; then
    OPTIONAL_ARGS+=(--allow-missing-manifest)
fi
if [[ "$S3A_TRAINABLE_EMA_ADAPTERS" == "1" ]]; then
    OPTIONAL_ARGS+=(--s3a-trainable-ema-adapters)
else
    OPTIONAL_ARGS+=(--no-s3a-trainable-ema-adapters)
fi
if [[ "$S3A_COLLAPSE_AUTO_MITIGATE" == "1" ]]; then
    OPTIONAL_ARGS+=(--s3a-collapse-auto-mitigate)
else
    OPTIONAL_ARGS+=(--no-s3a-collapse-auto-mitigate)
fi

env CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" \
    "$TORCHRUN_BIN" \
        --standalone \
        --nnodes=1 \
        --nproc_per_node="$NPROC_PER_NODE" \
    "$ROOT_DIR/train_s3a_multisource_dinov2.py" \
        --s3a \
        --data-path "$DATA_PATH" \
        --results-dir "$RESULTS_DIR" \
        --model DiT-XL/2 \
        --image-size 256 \
        --global-batch-size "$GLOBAL_BATCH_SIZE" \
        --global-seed "$GLOBAL_SEED" \
        --num-workers "$NUM_WORKERS" \
        --epochs "$EPOCHS" \
        --max-steps "$MAX_STEPS" \
        --ckpt-every "$CKPT_EVERY" \
        --log-every "$LOG_EVERY" \
        --vae-model-dir "$VAE_MODEL_DIR" \
        --s3a-lambda "$S3A_LAMBDA" \
        --s3a-layer-indices "$S3A_LAYER_INDICES" \
        --s3a-layer-weight-mode "$S3A_LAYER_WEIGHT_MODE" \
        --s3a-train-schedule "$S3A_TRAIN_SCHEDULE" \
        --s3a-schedule-steps "$S3A_SCHEDULE_STEPS" \
        --s3a-diff-schedule "$S3A_DIFF_SCHEDULE" \
        --s3a-feat-weight "$S3A_FEAT_WEIGHT" \
        --s3a-attn-weight "$S3A_ATTN_WEIGHT" \
        --s3a-spatial-weight "$S3A_SPATIAL_WEIGHT" \
        --s3a-self-warmup-steps "$S3A_SELF_WARMUP_STEPS" \
        --s3a-dino-alpha-floor "$S3A_DINO_ALPHA_FLOOR" \
        --s3a-dino-alpha-floor-steps "$S3A_DINO_ALPHA_FLOOR_STEPS" \
        --s3a-probe-every "$S3A_PROBE_EVERY" \
        --s3a-utility-probe-mode "$S3A_UTILITY_PROBE_MODE" \
        --s3a-gate-patience "$S3A_GATE_PATIENCE" \
        --s3a-gate-reopen-patience "$S3A_GATE_REOPEN_PATIENCE" \
        --s3a-gate-utility-off-threshold "$S3A_GATE_UTILITY_OFF_THRESHOLD" \
        --s3a-gate-utility-on-threshold "$S3A_GATE_UTILITY_ON_THRESHOLD" \
        --s3a-gate-utility-ema-momentum "$S3A_GATE_UTILITY_EMA_MOMENTUM" \
        --s3a-collapse-alpha-threshold "$S3A_COLLAPSE_ALPHA_THRESHOLD" \
        --s3a-collapse-self-threshold "$S3A_COLLAPSE_SELF_THRESHOLD" \
        --s3a-collapse-margin "$S3A_COLLAPSE_MARGIN" \
        --s3a-collapse-windows "$S3A_COLLAPSE_WINDOWS" \
        --s3a-collapse-utility-threshold "$S3A_COLLAPSE_UTILITY_THRESHOLD" \
        --s3a-collapse-mitigate-windows "$S3A_COLLAPSE_MITIGATE_WINDOWS" \
        --s3a-collapse-mitigate-cooldown-windows "$S3A_COLLAPSE_MITIGATE_COOLDOWN_WINDOWS" \
        --dinov2-repo-dir "$DINOV2_REPO_DIR" \
        --dinov2-weight-path "$DINOV2_WEIGHT_PATH" \
        "${OPTIONAL_ARGS[@]}" \
        "$@" \
    2>&1 | tee "$LAUNCH_LOG"
