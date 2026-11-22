# trading-coder

How far can a small local LLM, fine-tuned on a 16 GB MacBook (Apple M2, MLX),
get at writing **correct** trading-strategy code from a plain-English spec?

Main question: does LoRA fine-tuning a 3B coder model on its own verified
outputs close the gap to a 7–8B model that hasn't been fine-tuned?

Strategies are scored on correctness, not profit: a solution passes only if it
runs, matches a hand-written reference exactly on hidden synthetic price series,
and does not look ahead (changing future bars never changes past positions).

Status: work in progress. See [PROGRESS.md](PROGRESS.md) for the running log.

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
