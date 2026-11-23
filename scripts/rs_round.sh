#!/usr/bin/env bash
# One rejection-sampling round (Phase 5). Usage: scripts/rs_round.sh <round>
# Steps (each skipped if its output already exists, so an interrupted round resumes):
#   1 pool   CPU  25 fresh training tasks per family
#   2 sample GPU  8 samples/task at T=0.8 (base model in round 1, previous adapter after)
#   3 filter CPU  keep verified samples, dedupe, build data/rs_r<round>
#   4 train  GPU  LoRA from base, same budget as SFT-A (~17 min)
#   5 eval   GPU  100 test tasks, zero-shot greedy (~6 min)
# Close VMs / heavy apps first.
set -euo pipefail
cd "$(dirname "$0")/.."
R=$1; PY=.venv/bin/python
TASKS=data/rs/round${R}_tasks.jsonl; RUN=rs_r${R}_sample
[ -f "$TASKS" ] || $PY -m gen.rs pool --round "$R"
N_EXPECTED=$(( $(wc -l < "$TASKS") * 8 ))
if [ ! -f "results/$RUN/generations.jsonl" ] || [ "$(wc -l < "results/$RUN/generations.jsonl")" -lt "$N_EXPECTED" ]; then
  if [ -d "results/$RUN" ]; then echo "results/$RUN is incomplete: delete it and rerun"; exit 1; fi
  ADAPTER=(); [ "$R" -gt 1 ] && ADAPTER=(--adapter "adapters/rs_r$((R-1))")
  $PY -m gen.sample --model qwen3b ${ADAPTER[@]+"${ADAPTER[@]}"} --tasks-file "$TASKS" --n 8 --temp 0.8 --seed "$R" \
      --batch-size 16 --pause 1 --run "$RUN"
fi
[ -f "data/rs_r${R}/train.jsonl" ] || $PY -m gen.rs filter --round "$R" --run "results/$RUN"
[ -f "adapters/rs_r${R}/adapters.safetensors" ] || scripts/train_lora.sh train/lora_rs.yaml "rs_r${R}" --data "data/rs_r${R}"
ls -d results/ft_rs_r${R}_0shot_* >/dev/null 2>&1 || scripts/eval_adapter.sh "rs_r${R}"
echo "round $R done"
