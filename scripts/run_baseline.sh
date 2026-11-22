#!/usr/bin/env bash
# One baseline run: generate (GPU) then score (CPU). One model per call.
# Usage: scripts/run_baseline.sh <model> <shots> [extra gen.sample args...]
#   scripts/run_baseline.sh qwen3b 0
#   scripts/run_baseline.sh qwen7b 3 --pause 2
# Before running: close VMs / heavy apps; watch Activity Monitor memory pressure.
set -euo pipefail
cd "$(dirname "$0")/.."
MODEL=$1; SHOTS=$2; shift 2
RUN="baseline_${MODEL}_${SHOTS}shot_$(date +%Y%m%d_%H%M)"
.venv/bin/python -m gen.sample --model "$MODEL" --shots "$SHOTS" --run "$RUN" "$@"
.venv/bin/python -m eval.run_eval "results/$RUN"
