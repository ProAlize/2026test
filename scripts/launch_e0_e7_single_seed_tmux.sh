#!/usr/bin/env bash
# Launch E0-E7 single-seed screening in a detached tmux session.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SESSION_NAME="${SESSION_NAME:-e0e7_s1_$(date +%m%d_%H%M)}"
RUNNER="$ROOT_DIR/scripts/run_e0_e7_single_seed.sh"

if [[ ! -x "$RUNNER" ]]; then
    echo "[ERROR] Runner not executable: $RUNNER" >&2
    exit 1
fi

if ! command -v tmux >/dev/null 2>&1; then
    echo "[ERROR] tmux not found" >&2
    exit 1
fi

TMUX_LOG_DIR="${TMUX_LOG_DIR:-$ROOT_DIR/monitor_logs}"
mkdir -p "$TMUX_LOG_DIR"
TMUX_LOG="$TMUX_LOG_DIR/${SESSION_NAME}.log"

if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
    echo "[ERROR] tmux session already exists: $SESSION_NAME" >&2
    exit 1
fi

# Whitelist of variables that can override runner defaults.
FORWARD_KEYS=(
    ENV_PREFIX TORCHRUN_BIN PYTHON_BIN
    DATA_PATH VAE_MODEL_DIR DINOV2_REPO_DIR DINOV2_WEIGHT_PATH
    RESULTS_ROOT RUN_TAG
    CUDA_VISIBLE_DEVICES NPROC_PER_NODE
    MODEL IMAGE_SIZE NUM_CLASSES
    GLOBAL_SEED GLOBAL_BATCH_SIZE NUM_WORKERS
    EPOCHS MAX_STEPS CKPT_EVERY LOG_EVERY
    ALIGN_LAMBDA DRY_RUN FORBID_RESUME START_FROM
    S3A_SELF_WARMUP_STEPS
    S3A_DINO_ALPHA_FLOOR S3A_DINO_ALPHA_FLOOR_STEPS
    S3A_PROBE_EVERY S3A_UTILITY_PROBE_MODE
    S3A_GATE_PATIENCE S3A_GATE_REOPEN_PATIENCE
    S3A_GATE_UTILITY_OFF_THRESHOLD S3A_GATE_UTILITY_ON_THRESHOLD
    S3A_GATE_UTILITY_EMA_MOMENTUM
    S3A_COLLAPSE_ALPHA_THRESHOLD S3A_COLLAPSE_SELF_THRESHOLD
    S3A_COLLAPSE_MARGIN S3A_COLLAPSE_WINDOWS
    S3A_COLLAPSE_UTILITY_THRESHOLD
    S3A_COLLAPSE_AUTO_MITIGATE
    S3A_COLLAPSE_MITIGATE_WINDOWS S3A_COLLAPSE_MITIGATE_COOLDOWN_WINDOWS
)

ENV_PREFIXES=""
for key in "${FORWARD_KEYS[@]}"; do
    if [[ -n "${!key+x}" ]]; then
        printf -v escaped_value '%q' "${!key}"
        ENV_PREFIXES+="$key=$escaped_value "
    fi
done

TMUX_CMD="cd '$ROOT_DIR' && env ${ENV_PREFIXES}bash '$RUNNER' 2>&1 | tee '$TMUX_LOG'"
tmux new-session -d -s "$SESSION_NAME" "$TMUX_CMD"

echo "Launched tmux session: $SESSION_NAME"
echo "Tail log: $TMUX_LOG"
echo "Attach : tmux attach -t $SESSION_NAME"
echo "Status : tmux ls | rg '$SESSION_NAME'"
