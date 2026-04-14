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
    ALIGN_LAMBDA DRY_RUN FORBID_RESUME
)

ENV_PREFIXES=""
for key in "${FORWARD_KEYS[@]}"; do
    if [[ -n "${!key+x}" ]]; then
        # Values in this workflow are paths/numbers/simple strings (no embedded quotes expected).
        ENV_PREFIXES+="$key=\"${!key}\" "
    fi
done

TMUX_CMD="cd '$ROOT_DIR' && env ${ENV_PREFIXES}bash '$RUNNER' 2>&1 | tee '$TMUX_LOG'"
tmux new-session -d -s "$SESSION_NAME" "$TMUX_CMD"

echo "Launched tmux session: $SESSION_NAME"
echo "Tail log: $TMUX_LOG"
echo "Attach : tmux attach -t $SESSION_NAME"
echo "Status : tmux ls | rg '$SESSION_NAME'"
