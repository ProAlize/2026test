#!/usr/bin/env bash
# E0-E7 minimal causal package (single-seed screening)
# Strict policy for this round: fresh runs only, no --resume.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

ENV_PREFIX="${ENV_PREFIX:-/home/liuchunfa/anaconda3/envs/DiT}"
TORCHRUN_BIN="${TORCHRUN_BIN:-$ENV_PREFIX/bin/torchrun}"
PYTHON_BIN="${PYTHON_BIN:-$ENV_PREFIX/bin/python}"

DATA_PATH="${DATA_PATH:-/data/liuchunfa/2026qjx/repa_imagenet256_official/images}"
VAE_MODEL_DIR="${VAE_MODEL_DIR:-/home/liuchunfa/.cache/modelscope/hub/models/facebook/DiT-XL-2-256/vae}"
DINOV2_REPO_DIR="${DINOV2_REPO_DIR:-/data/liuchunfa/2026qjx/assets/dinov2}"
DINOV2_WEIGHT_PATH="${DINOV2_WEIGHT_PATH:-/data/liuchunfa/2026qjx/assets/dinov2_weights/dinov2_vitb14_pretrain.pth}"

RESULTS_ROOT="${RESULTS_ROOT:-/data/liuchunfa/2026qjx/2026test_runs/e0e7_single_seed}"
RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
NPROC_PER_NODE="${NPROC_PER_NODE:-4}"

MODEL="${MODEL:-DiT-B/2}"
IMAGE_SIZE="${IMAGE_SIZE:-256}"
NUM_CLASSES="${NUM_CLASSES:-auto}"
GLOBAL_SEED="${GLOBAL_SEED:-0}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-256}"
NUM_WORKERS="${NUM_WORKERS:-8}"
EPOCHS="${EPOCHS:-1400}"
MAX_STEPS="${MAX_STEPS:-40000}"
CKPT_EVERY="${CKPT_EVERY:-2000}"
LOG_EVERY="${LOG_EVERY:-200}"

# Common alignment defaults.
ALIGN_LAMBDA="${ALIGN_LAMBDA:-0.1}"
S3A_SELF_WARMUP_STEPS="${S3A_SELF_WARMUP_STEPS:-5000}"
S3A_DINO_ALPHA_FLOOR="${S3A_DINO_ALPHA_FLOOR:-0.1}"
S3A_DINO_ALPHA_FLOOR_STEPS="${S3A_DINO_ALPHA_FLOOR_STEPS:-8000}"
S3A_PROTECT_SOURCE0_MIN_ALPHA="${S3A_PROTECT_SOURCE0_MIN_ALPHA:-0.05}"
S3A_PROBE_EVERY="${S3A_PROBE_EVERY:-10}"
S3A_UTILITY_PROBE_MODE="${S3A_UTILITY_PROBE_MODE:-policy_loo}"
S3A_GATE_REOPEN_PROBE_ALPHA_FLOOR="${S3A_GATE_REOPEN_PROBE_ALPHA_FLOOR:-0.05}"
S3A_GATE_PATIENCE="${S3A_GATE_PATIENCE:-500}"
S3A_GATE_REOPEN_PATIENCE="${S3A_GATE_REOPEN_PATIENCE:-200}"
S3A_GATE_UTILITY_OFF_THRESHOLD="${S3A_GATE_UTILITY_OFF_THRESHOLD:-0.002}"
S3A_GATE_UTILITY_ON_THRESHOLD="${S3A_GATE_UTILITY_ON_THRESHOLD:-0.005}"
S3A_GATE_UTILITY_EMA_MOMENTUM="${S3A_GATE_UTILITY_EMA_MOMENTUM:-0.9}"
S3A_COLLAPSE_UTILITY_THRESHOLD="${S3A_COLLAPSE_UTILITY_THRESHOLD:-0.0}"
S3A_COLLAPSE_ALPHA_THRESHOLD="${S3A_COLLAPSE_ALPHA_THRESHOLD:-0.05}"
S3A_COLLAPSE_SELF_THRESHOLD="${S3A_COLLAPSE_SELF_THRESHOLD:-0.90}"
S3A_COLLAPSE_MARGIN="${S3A_COLLAPSE_MARGIN:-0.01}"
S3A_COLLAPSE_WINDOWS="${S3A_COLLAPSE_WINDOWS:-3}"
S3A_COLLAPSE_AUTO_MITIGATE="${S3A_COLLAPSE_AUTO_MITIGATE:-1}"
S3A_COLLAPSE_MITIGATE_WINDOWS="${S3A_COLLAPSE_MITIGATE_WINDOWS:-3}"
S3A_COLLAPSE_MITIGATE_COOLDOWN_WINDOWS="${S3A_COLLAPSE_MITIGATE_COOLDOWN_WINDOWS:-6}"

# Execution controls.
DRY_RUN="${DRY_RUN:-0}"
FORBID_RESUME="${FORBID_RESUME:-1}"
START_FROM="${START_FROM:-E0}"

if [[ "$FORBID_RESUME" == "1" ]]; then
    if [[ -n "${RESUME_CKPT:-}" || -n "${RESUME:-}" ]]; then
        echo "[ERROR] This launcher forbids resume for E0-E7 round. Unset RESUME_CKPT/RESUME." >&2
        exit 1
    fi
fi

exp_rank() {
    case "$1" in
        E0) echo 0 ;;
        E1) echo 1 ;;
        E1b) echo 2 ;;
        E2) echo 3 ;;
        E3) echo 4 ;;
        E4) echo 5 ;;
        E5) echo 6 ;;
        E6) echo 7 ;;
        E7) echo 8 ;;
        *) return 1 ;;
    esac
}

if ! START_RANK="$(exp_rank "$START_FROM")"; then
    echo "[ERROR] Invalid START_FROM=$START_FROM. Valid: E0,E1,E1b,E2,E3,E4,E5,E6,E7" >&2
    exit 1
fi

for required_path in \
    "$ROOT_DIR/train_sasa_dinov2.py" \
    "$ROOT_DIR/train_s3a_multisource_dinov2.py" \
    "$TORCHRUN_BIN" \
    "$PYTHON_BIN" \
    "$DATA_PATH" \
    "$VAE_MODEL_DIR" \
    "$DINOV2_REPO_DIR" \
    "$DINOV2_REPO_DIR/hubconf.py" \
    "$DINOV2_WEIGHT_PATH"; do
    if [[ ! -e "$required_path" ]]; then
        echo "[ERROR] Required path not found: $required_path" >&2
        exit 1
    fi
done

GPU_COUNT=$(echo "$CUDA_VISIBLE_DEVICES" | tr ',' '\n' | wc -l | tr -d ' ')
if [[ "$GPU_COUNT" -lt "$NPROC_PER_NODE" ]]; then
    echo "[ERROR] NPROC_PER_NODE=$NPROC_PER_NODE exceeds visible GPU count=$GPU_COUNT" >&2
    exit 1
fi

MODEL_DEPTH="$($PYTHON_BIN - <<PY
from model_sasa import DiT_models
model = DiT_models["$MODEL"](input_size=$((IMAGE_SIZE / 8)), num_classes=1000)
print(len(model.blocks))
PY
)"

if ! [[ "$MODEL_DEPTH" =~ ^[0-9]+$ ]]; then
    echo "[ERROR] Failed to infer model depth for MODEL=$MODEL" >&2
    exit 1
fi

if (( MODEL_DEPTH < 4 )); then
    echo "[ERROR] MODEL_DEPTH=$MODEL_DEPTH is too small for E3 late-4 experiment." >&2
    exit 1
fi

L_SINGLE=$((MODEL_DEPTH - 1))
L_LATE4_START=$((MODEL_DEPTH - 4))
L_LATE4="${L_LATE4_START},$((L_LATE4_START + 1)),$((L_LATE4_START + 2)),$((L_LATE4_START + 3))"

if [[ "$NUM_CLASSES" == "auto" ]]; then
    NUM_CLASSES="$($PYTHON_BIN - <<PY
import os
path = "$DATA_PATH"
print(sum(1 for x in os.scandir(path) if x.is_dir()))
PY
)"
fi

if ! [[ "$NUM_CLASSES" =~ ^[0-9]+$ ]] || (( NUM_CLASSES <= 0 )); then
    echo "[ERROR] Invalid NUM_CLASSES=$NUM_CLASSES" >&2
    exit 1
fi

RUN_DIR="$RESULTS_ROOT/$RUN_TAG"
mkdir -p "$RUN_DIR"
LOG_DIR="$RUN_DIR/logs"
mkdir -p "$LOG_DIR"

print_header() {
    echo "=============================================================="
    echo "E0-E7 single-seed screening (fresh runs, no resume)"
    echo "Run tag          : $RUN_TAG"
    echo "Start from       : $START_FROM"
    echo "Root dir         : $ROOT_DIR"
    echo "Results root     : $RUN_DIR"
    echo "Torchrun         : $TORCHRUN_BIN"
    echo "Python           : $PYTHON_BIN"
    echo "CUDA devices     : $CUDA_VISIBLE_DEVICES (nproc=$NPROC_PER_NODE)"
    echo "Model            : $MODEL"
    echo "Image size       : $IMAGE_SIZE"
    echo "Num classes      : $NUM_CLASSES"
    echo "Global seed      : $GLOBAL_SEED"
    echo "Global batch     : $GLOBAL_BATCH_SIZE"
    echo "Max steps        : $MAX_STEPS"
    echo "Ckpt every       : $CKPT_EVERY"
    echo "Log every        : $LOG_EVERY"
    echo "S3A warmup       : $S3A_SELF_WARMUP_STEPS"
    echo "S3A alpha floor  : $S3A_DINO_ALPHA_FLOOR (steps=$S3A_DINO_ALPHA_FLOOR_STEPS)"
    echo "S3A alpha min    : $S3A_PROTECT_SOURCE0_MIN_ALPHA"
    echo "S3A probe        : every $S3A_PROBE_EVERY, estimator=$S3A_UTILITY_PROBE_MODE"
    echo "S3A reopen αfl   : $S3A_GATE_REOPEN_PROBE_ALPHA_FLOOR (auto->0 when --no-s3a-use-ema-source)"
    echo "S3A gate patience: off=$S3A_GATE_PATIENCE, reopen=$S3A_GATE_REOPEN_PATIENCE"
    echo "S3A gate utility : off=$S3A_GATE_UTILITY_OFF_THRESHOLD, on=$S3A_GATE_UTILITY_ON_THRESHOLD"
    echo "S3A gate EMA mom : $S3A_GATE_UTILITY_EMA_MOMENTUM"
    echo "S3A collapse     : alpha<$S3A_COLLAPSE_ALPHA_THRESHOLD, self>$S3A_COLLAPSE_SELF_THRESHOLD, u>$S3A_COLLAPSE_UTILITY_THRESHOLD, margin=$S3A_COLLAPSE_MARGIN, windows=$S3A_COLLAPSE_WINDOWS"
    echo "S3A mitigate     : auto=$S3A_COLLAPSE_AUTO_MITIGATE, hold=$S3A_COLLAPSE_MITIGATE_WINDOWS, cooldown=$S3A_COLLAPSE_MITIGATE_COOLDOWN_WINDOWS"
    echo "Model depth      : $MODEL_DEPTH"
    echo "Single tap idx   : $L_SINGLE"
    echo "Late-4 tap idx   : $L_LATE4"
    echo "Spread-4 tap idx : auto"
    echo "DRY_RUN          : $DRY_RUN"
    echo "=============================================================="
}

run_if_reached() {
    local exp_id="$1"
    shift
    local exp_r
    exp_r="$(exp_rank "$exp_id")"
    if (( exp_r >= START_RANK )); then
        "$@"
    else
        echo "[$(date '+%F %T')] --- SKIP $exp_id (before START_FROM=$START_FROM)"
    fi
}

run_cmd() {
    local exp_name="$1"
    shift

    local exp_dir="$RUN_DIR/$exp_name"
    local exp_log="$LOG_DIR/${exp_name}.log"
    mkdir -p "$exp_dir"

    echo "[$(date '+%F %T')] >>> START $exp_name"
    if [[ "$DRY_RUN" == "1" ]]; then
        echo "[DRY_RUN] $*" | tee "$exp_log"
        return 0
    fi

    (
        cd "$ROOT_DIR"
        env CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" "$@"
    ) 2>&1 | tee "$exp_log"

    echo "[$(date '+%F %T')] <<< DONE  $exp_name"
}

run_sasa() {
    local exp_name="$1"
    local diff_schedule="$2"

    run_cmd "$exp_name" \
        "$TORCHRUN_BIN" \
        --standalone \
        --nnodes=1 \
        --nproc_per_node="$NPROC_PER_NODE" \
        "$ROOT_DIR/train_sasa_dinov2.py" \
        --data-path "$DATA_PATH" \
        --results-dir "$RUN_DIR/$exp_name" \
        --model "$MODEL" \
        --image-size "$IMAGE_SIZE" \
        --num-classes "$NUM_CLASSES" \
        --global-batch-size "$GLOBAL_BATCH_SIZE" \
        --global-seed "$GLOBAL_SEED" \
        --num-workers "$NUM_WORKERS" \
        --epochs "$EPOCHS" \
        --max-steps "$MAX_STEPS" \
        --ckpt-every "$CKPT_EVERY" \
        --log-every "$LOG_EVERY" \
        --vae-model-dir "$VAE_MODEL_DIR" \
        --repa \
        --repa-lambda "$ALIGN_LAMBDA" \
        --repa-train-schedule constant \
        --repa-schedule-steps "$MAX_STEPS" \
        --repa-diff-schedule "$diff_schedule" \
        --dinov2-repo-dir "$DINOV2_REPO_DIR" \
        --dinov2-weight-path "$DINOV2_WEIGHT_PATH"
}

run_s3a() {
    local exp_name="$1"
    local layer_indices="$2"
    local use_ema_source="$3"
    local use_gate="$4"
    local attn_weight="$5"
    local spatial_weight="$6"
    local diff_schedule="$7"
    local ema_flag="--no-s3a-use-ema-source"
    local gate_flag="--no-s3a-enable-selective-gate"
    local auto_mitigate_flag="--no-s3a-collapse-auto-mitigate"
    local floor_tag="${S3A_DINO_ALPHA_FLOOR//./p}"
    local floor_min_tag="${S3A_PROTECT_SOURCE0_MIN_ALPHA//./p}"
    local reopen_probe_alpha_effective="$S3A_GATE_REOPEN_PROBE_ALPHA_FLOOR"
    if [[ "$use_ema_source" != "1" ]]; then
        reopen_probe_alpha_effective="0.0"
    fi
    local reopen_probe_alpha_tag="${reopen_probe_alpha_effective//./p}"
    local utility_tag="${S3A_UTILITY_PROBE_MODE//[^a-zA-Z0-9]/_}"
    local contract_suffix="w${S3A_SELF_WARMUP_STEPS}_f${floor_tag}_fmin${floor_min_tag}_fs${S3A_DINO_ALPHA_FLOOR_STEPS}_p${S3A_PROBE_EVERY}_u${utility_tag}_rp${reopen_probe_alpha_tag}_cw${S3A_COLLAPSE_WINDOWS}_m${S3A_COLLAPSE_AUTO_MITIGATE}"
    local exp_name_with_contract="${exp_name}__${contract_suffix}"

    if [[ "$use_ema_source" == "1" ]]; then
        ema_flag="--s3a-use-ema-source"
    fi
    if [[ "$use_gate" == "1" ]]; then
        gate_flag="--s3a-enable-selective-gate"
    fi
    if [[ "$S3A_COLLAPSE_AUTO_MITIGATE" == "1" ]]; then
        auto_mitigate_flag="--s3a-collapse-auto-mitigate"
    fi

    run_cmd "$exp_name_with_contract" \
        "$TORCHRUN_BIN" \
        --standalone \
        --nnodes=1 \
        --nproc_per_node="$NPROC_PER_NODE" \
        "$ROOT_DIR/train_s3a_multisource_dinov2.py" \
        --s3a \
        --data-path "$DATA_PATH" \
        --results-dir "$RUN_DIR/$exp_name_with_contract" \
        --model "$MODEL" \
        --image-size "$IMAGE_SIZE" \
        --num-classes "$NUM_CLASSES" \
        --global-batch-size "$GLOBAL_BATCH_SIZE" \
        --global-seed "$GLOBAL_SEED" \
        --num-workers "$NUM_WORKERS" \
        --epochs "$EPOCHS" \
        --max-steps "$MAX_STEPS" \
        --ckpt-every "$CKPT_EVERY" \
        --log-every "$LOG_EVERY" \
        --vae-model-dir "$VAE_MODEL_DIR" \
        --s3a-lambda "$ALIGN_LAMBDA" \
        --s3a-layer-indices "$layer_indices" \
        --s3a-layer-weight-mode uniform \
        --s3a-train-schedule constant \
        --s3a-schedule-steps "$MAX_STEPS" \
        --s3a-diff-schedule "$diff_schedule" \
        --s3a-feat-weight 1.0 \
        --s3a-attn-weight "$attn_weight" \
        --s3a-spatial-weight "$spatial_weight" \
        --s3a-self-warmup-steps "$S3A_SELF_WARMUP_STEPS" \
        --s3a-dino-alpha-floor "$S3A_DINO_ALPHA_FLOOR" \
        --s3a-dino-alpha-floor-steps "$S3A_DINO_ALPHA_FLOOR_STEPS" \
        --s3a-protect-source0-min-alpha "$S3A_PROTECT_SOURCE0_MIN_ALPHA" \
        --s3a-probe-every "$S3A_PROBE_EVERY" \
        --s3a-utility-probe-mode "$S3A_UTILITY_PROBE_MODE" \
        --s3a-gate-reopen-probe-alpha-floor "$reopen_probe_alpha_effective" \
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
        --no-s3a-trainable-ema-adapters \
        "$auto_mitigate_flag" \
        "$ema_flag" \
        "$gate_flag" \
        --dinov2-repo-dir "$DINOV2_REPO_DIR" \
        --dinov2-weight-path "$DINOV2_WEIGHT_PATH"
}

print_header

# E0-E1b: SASA-dinov2, single layer, only diff schedule changes.
run_if_reached E0  run_sasa "E0_sasa_uniform" "uniform"
run_if_reached E1  run_sasa "E1_sasa_cosine" "cosine"
run_if_reached E1b run_sasa "E1b_sasa_linear_low" "linear_low"

# E2-E7: S3A ladder.
# Tap policy:
# - single layer = last block
# - late 4 layers = last four blocks
# - spread 4 layers = auto quarter/mid/three_quarter/last
run_if_reached E2 run_s3a "E2_s3a_degenerate_single_dino_feat_uniform" "$L_SINGLE" 0 0 0.0 0.0 "uniform"
run_if_reached E3 run_s3a "E3_s3a_late4_dino_feat_uniform"            "$L_LATE4"  0 0 0.0 0.0 "uniform"
run_if_reached E4 run_s3a "E4_s3a_spread4_dino_feat_uniform"          "auto"      0 0 0.0 0.0 "uniform"
run_if_reached E5 run_s3a "E5_s3a_spread4_dualsrc_feat_uniform"       "auto"      1 0 0.0 0.0 "uniform"
run_if_reached E6 run_s3a "E6_s3a_spread4_dualsrc_gate_feat_uniform"  "auto"      1 1 0.0 0.0 "uniform"
run_if_reached E7 run_s3a "E7_s3a_spread4_dualsrc_gate_holistic_cos"  "auto"      1 1 0.5 0.5 "cosine"

echo "All E0-E7 jobs finished. Logs: $LOG_DIR"
