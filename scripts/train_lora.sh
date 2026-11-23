#!/usr/bin/env bash
# LoRA training (GPU). Usage: scripts/train_lora.sh <config.yaml> <name> [extra mlx_lm.lora args]
#   scripts/train_lora.sh train/lora_sft_a.yaml sft_a --iters 20     # timing test, ~2-3 min
#   scripts/train_lora.sh train/lora_sft_a.yaml sft_a                # full run
# Close VMs / heavy apps first. Adapter, config copy and log go to adapters/<name>/.
set -euo pipefail
cd "$(dirname "$0")/.."
CONFIG=$1; NAME=$2; shift 2
OUT="adapters/$NAME"
if [ -e "$OUT/adapters.safetensors" ]; then echo "$OUT already has an adapter; pick a new name"; exit 1; fi
mkdir -p "$OUT"
cp "$CONFIG" "$OUT/config.yaml"
echo "git_commit: $(git rev-parse HEAD)$(git diff --quiet || echo -dirty)  started: $(date '+%F %T')  extra: $*" > "$OUT/run_info.txt"
.venv/bin/mlx_lm.lora --config "$CONFIG" --adapter-path "$OUT" "$@" 2>&1 | tee "$OUT/train.log"
echo "finished: $(date '+%F %T')" >> "$OUT/run_info.txt"
