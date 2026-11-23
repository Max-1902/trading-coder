# trading-coder

How far can a small local LLM, fine-tuned on a 16 GB MacBook (Apple M2, MLX),
get at writing **correct** trading-strategy code from a plain-English spec?

Main question: does LoRA fine-tuning a 3B coder model on its own verified
outputs close the gap to a 7–8B model that hasn't been fine-tuned?

Strategies are scored on correctness, not profit: a solution passes only if it
runs, matches a hand-written reference exactly on hidden synthetic price series,
and does not look ahead (changing future bars never changes past positions).

Status: work in progress. See [PROGRESS.md](PROGRESS.md) for the running log.

## Baselines (no fine-tuning, greedy pass@1, 95% bootstrap CI over 100 tasks)

| Model (4-bit MLX) | 0-shot | 3-shot |
|---|---|---|
| Qwen2.5-Coder-3B | 0.17 [0.10, 0.25] | 0.26 [0.18, 0.35] |
| Qwen2.5-Coder-7B | 0.11 [0.05, 0.17] | 0.53 [0.43, 0.63] |
| Llama-3.1-8B-Instruct | 0.10 [0.05, 0.16] | 0.40 [0.30, 0.50] |

**After LoRA fine-tuning on reference solutions (3B, 100 steps, ~17 min):** pass@1 0.67 [0.58, 0.76]:
0.93 on in-distribution tasks, but 0.00 on the two held-out strategy families. On those it forces
each new strategy into the nearest template it was trained on.

**Self-taught (rejection sampling, 2 rounds):** 0.17 → 0.24 → 0.25. It only improves families the
base model can already partly solve; stateful families yield no training data, and round 2 raised
look-ahead to 22% of running solutions through a habit learned from verified, look-ahead-free samples.

The 3B base model's main failure is not formatting but state handling: it doesn't carry a position from one
bar to the next (13 of 30 labelled failures), followed by pandas indexing/API mistakes (9 of 30).

## Setup

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r requirements.txt
.venv/bin/python -m pytest -q                          # harness + benchmark tests
.venv/bin/python -m gen.make_train --per-family 400     # training tasks -> data/
.venv/bin/python -m bench.leakage data/train_tasks.jsonl
scripts/run_baseline.sh qwen3b 0                         # one model per run; uses the GPU
```

## Layout

| Path | Contents |
|---|---|
| `harness/series.py` | seeded synthetic OHLCV series (random walk, trend, OU, regime switch, flat stretches, gaps) |
| `harness/sandbox.py` | AST import/call allowlist + sandboxed subprocess with timeout, no network, no file writes |
| `harness/checks.py` | runs / correct / no-look-ahead scoring and failure categories |
| `harness/backtest.py` | sanity stats only (Sharpe, drawdown, turnover), never a score |
| `bench/families/` | one module per strategy family: parameter grid, prompt phrasings, rules, pandas reference, independent loop implementation |
| `bench/tasks_test_id.jsonl`, `bench/tasks_test_ood.jsonl` | 72 in-distribution and 28 out-of-distribution test tasks (`python -m bench.build`) |
| `bench/leakage.py` | fails if any test prompt, parameter combination or parameter value appears in training data |
| `gen/make_train.py` | training tasks from the train split (`python -m gen.make_train`) |
| `gen/sample.py`, `gen/extract.py` | MLX generation (zero-/few-shot) and code extraction |
| `eval/run_eval.py`, `eval/report.py`, `eval/failures.py` | scoring with bootstrap CIs, run comparison, failure samples |
| `scripts/run_baseline.sh` | one baseline run (generate + score) for one model |
| `results/` | one folder per run with config, raw generations and scores |
