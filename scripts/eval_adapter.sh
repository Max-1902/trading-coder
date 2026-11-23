#!/usr/bin/env bash
# Evaluate a LoRA adapter on all 100 test tasks (GPU), then score (CPU).
# Usage: scripts/eval_adapter.sh <adapter_name> [shots] [extra gen.sample args]
#   scripts/eval_adapter.sh sft_a            # zero-shot, ~7 min
set -euo pipefail
cd "$(dirname "$0")/.."
NAME=$1; SHOTS=${2:-0}; shift $(( $# >= 2 ? 2 : 1 ))
RUN="ft_${NAME}_${SHOTS}shot_$(date +%Y%m%d_%H%M)"
.venv/bin/python -m gen.sample --model qwen3b --adapter "adapters/$NAME" --shots "$SHOTS" --run "$RUN" --pause 1 "$@"
.venv/bin/python -m eval.run_eval "results/$RUN"
