#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUNNER="$ROOT_DIR/scripts/run_taware_xl_single_seed.sh"
SESSION_NAME="${SESSION_NAME:-taware_xl_s1_$(date +%m%d_%H%M)}"

[[ -x "$RUNNER" ]] || { echo "[ERROR] Runner not executable: $RUNNER" >&2; exit 1; }
command -v tmux >/dev/null 2>&1 || { echo "[ERROR] tmux not found" >&2; exit 1; }

if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
    echo "[ERROR] tmux session already exists: $SESSION_NAME" >&2
    exit 1
fi

TMUX_LOG_DIR="${TMUX_LOG_DIR:-$ROOT_DIR/monitor_logs}"
mkdir -p "$TMUX_LOG_DIR"
TMUX_LOG="$TMUX_LOG_DIR/${SESSION_NAME}.log"

# Robust env forwarding for key overrides.
tmux new-session -d -s "$SESSION_NAME"
FORWARD_KEYS=(
    WORKTREE_DIR ENV_PREFIX TORCHRUN_BIN PYTHON_BIN
    DATA_PATH VAE_MODEL_DIR DINO_MODEL_DIR
    RESULTS_ROOT RUN_TAG
    CUDA_VISIBLE_DEVICES NPROC_PER_NODE
    MODEL IMAGE_SIZE NUM_CLASSES
    GLOBAL_SEED GLOBAL_BATCH_SIZE NUM_WORKERS
    EPOCHS MAX_STEPS CKPT_EVERY LOG_EVERY
    REPA_LAMBDA REPA_DIFF_SCHEDULE REPA_DIFF_THRESHOLD
    FORBID_RESUME
)
for key in "${FORWARD_KEYS[@]}"; do
    if [[ -n "${!key+x}" ]]; then
        tmux set-environment -t "$SESSION_NAME" "$key" "${!key}"
    fi
done

tmux send-keys -t "$SESSION_NAME" "cd '$ROOT_DIR' && bash '$RUNNER' 2>&1 | tee '$TMUX_LOG'" C-m

echo "Launched tmux session: $SESSION_NAME"
echo "Tail log: $TMUX_LOG"
echo "Attach : tmux attach -t $SESSION_NAME"
