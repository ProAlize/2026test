#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
READY_FILE="${READY_FILE:-/data/liuchunfa/2026qjx/repa_imagenet256_ditvae/vae-sd/dataset.json}"
POLL_SECONDS="${POLL_SECONDS:-300}"
LOG_DIR="${LOG_DIR:-/data/liuchunfa/2026qjx/repa_sit_dinov3_b36_160k}"

mkdir -p "$LOG_DIR"

while [[ ! -f "$READY_FILE" ]]; do
    echo "[$(date +%F_%T)] waiting for official REPA preprocessing to finish: $READY_FILE"
    sleep "$POLL_SECONDS"
done

cd "$ROOT_DIR"
exec ./run_sit_repa_dinov3_b24_160k.sh
