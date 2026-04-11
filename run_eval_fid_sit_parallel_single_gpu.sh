#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

SESSION_PREFIX="${SESSION_PREFIX:-fid_sit_single_gpu}"
GPU_LIST_CSV="${GPU_LIST_CSV:-0,1,2,3}"
IFS=',' read -r -a GPUS <<< "$GPU_LIST_CSV"

CHECKPOINTS=("$@")
if [[ ${#CHECKPOINTS[@]} -eq 0 ]]; then
    echo "Usage: $0 /path/to/ckpt1.pt [/path/to/ckpt2.pt ...]" >&2
    exit 1
fi

if [[ ${#CHECKPOINTS[@]} -gt ${#GPUS[@]} ]]; then
    echo "Received ${#CHECKPOINTS[@]} checkpoints but only ${#GPUS[@]} GPUs were provided." >&2
    exit 1
fi

for ckpt in "${CHECKPOINTS[@]}"; do
    if [[ ! -f "$ckpt" ]]; then
        echo "Checkpoint not found: $ckpt" >&2
        exit 1
    fi
done

for idx in "${!CHECKPOINTS[@]}"; do
    gpu="${GPUS[$idx]}"
    ckpt="${CHECKPOINTS[$idx]}"
    ckpt_stem="$(basename "${ckpt%.pt}")"
    session_name="${SESSION_PREFIX}_${ckpt_stem}_gpu${gpu}"
    sweep_dir="$(dirname "$ckpt")/offline_eval/sweeps"
    log_path="$sweep_dir/${ckpt_stem}_sit_single_gpu${gpu}_$(date +%Y%m%d_%H%M%S).log"

    mkdir -p "$sweep_dir"

    tmux kill-session -t "$session_name" 2>/dev/null || true
    tmux new-session -d -s "$session_name" \
        "bash -lc 'cd \"$ROOT_DIR\" && echo \"===== SiT ${ckpt_stem} on GPU ${gpu} at \$(date +%F_%T) =====\" | tee -a \"$log_path\" && CUDA_VISIBLE_DEVICES=${gpu} NPROC_PER_NODE=1 OVERWRITE_EVAL=1 ./run_eval_fid_sit_offline.sh \"$ckpt\" 2>&1 | tee -a \"$log_path\"'"

    echo "Launched SiT $ckpt_stem on GPU $gpu"
    echo "  tmux session: $session_name"
    echo "  log: $log_path"
done
