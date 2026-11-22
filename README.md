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
.venv/bin/python -m pytest -q      # harness tests
```

## Layout

| Path | Contents |
|---|---|
| `harness/series.py` | seeded synthetic OHLCV series (random walk, trend, OU, regime switch, flat stretches, gaps) |
| `harness/sandbox.py` | AST import/call allowlist + sandboxed subprocess with timeout, no network, no file writes |
| `harness/checks.py` | runs / correct / no-look-ahead scoring and failure categories |
| `harness/backtest.py` | sanity stats only (Sharpe, drawdown, turnover), never a score |
| `bench/` | test tasks and per-family generators + reference implementations |
| `gen/`, `train/`, `eval/` | sampling, rejection filtering, LoRA config, evaluation |
| `results/` | one folder per run with config, raw generations and scores |
