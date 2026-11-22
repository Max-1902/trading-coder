# PROGRESS

Running log: decisions, what broke, how it was diagnosed. Newest state first.

## Current state (2026-11-22)

- **Phase 0 (setup): done.** Phase 1 (harness): done, 55/55 tests pass.
- Next: Phase 2, the benchmark (families, ~100 test tasks, Test-ID / Test-OOD splits, leakage check).

## Setup decisions

- **Separate project.** Standalone repo next to `llama-project` (the RAG study). The only
  thing shared is the Llama-3.1-8B-Instruct 4-bit weights folder, read by path
  (`gen/models.py`, override with `LLAMA_8B_PATH`). Nothing in the RAG project is modified.
- **Environment:** own `.venv` (Python 3.12, uv). mlx 0.32.4, mlx-lm 0.32.2, numpy 2.5.5,
  pandas 3.1.1. Pinned in `requirements.txt`.
- **Models.** I checked for newer small coder models before fixing the choice: Qwen3-Coder
  only ships as large MoE models (30B-A3B and up), which are out of reach on 16 GB.
  No newer 1.5B/3B/7B coder generation exists in MLX 4-bit, so the Qwen2.5-Coder
  1.5B / 3B / 7B family stays. One family across three sizes also keeps the scaling comparison clean.
- **Smoke test:** Qwen2.5-Coder-3B-Instruct-4bit, greedy, 150 tokens: 88 tok/s, peak memory 2.0 GB.
- **CLI flags (mlx-lm 0.32.2), checked with `--help`:** `mlx_lm.lora` has `--mask-prompt`
  (loss on completion only), `--grad-checkpoint`, `--num-layers`, `--max-seq-length`, `--seed`;
  `mlx_lm.generate` uses `--temp`, `--top-p`, `--seed`, `--adapter-path`.
- **Disk is the tight constraint:** about 10 GB free at the start. Qwen 1.5B + 7B still to download
  (~5.2 GB). Llama is not downloaded again.

## Phase 1: harness

**Series (`harness/series.py`).** Six generators (random walk, strong trend, OU mean-reverting,
regime switch, flat stretches, gaps with missing dates), 500 bars each. Nine hidden series with fixed
seeds. No NaNs in the input, so the NaN convention only concerns indicator warm-up.

- *Float ties.* Two correct implementations can disagree on an exact tie (fast SMA == slow SMA in a
  flat stretch) only because they sum floats in a different order. Fix: all prices are rounded to a
  1/128 tick, which is exact in float64, so rolling sums are bit-identical across pandas rolling,
  numpy cumsum and Python loops. The test `test_differently_written_correct_code_passes` confirms it:
  a cumsum SMA and a pure-Python Wilder RSI both match the pandas references exactly.
  Indicators with division or square roots (std, z-score) can still split a tie. Phase 2 has to state
  a convention for those, and each family needs a second, independent implementation to show the
  reference isn't fragile.

**Sandbox (`harness/sandbox.py`, `harness/_runner.py`).**
- AST pass: imports only numpy/pandas/math; bans eval/exec/open/`__import__`/getattr etc., any dunder
  attribute, pandas/numpy file I/O; requires a top-level `generate_signals`.
- Execution: `python -I` under macOS `sandbox-exec` with network and file writes denied (verified:
  socket connect fails with EPERM); 30 s wall-clock timeout for the whole batch. Values come back as
  JSON; the parent never unpickles anything from the child process.
- One subprocess per solution runs all 9 hidden series plus about 100 look-ahead variants (about 0.4 s
  for vectorised code).

**Checks (`harness/checks.py`).** pass = runs ∧ correct ∧ no-look-ahead. Primary failure category in
precedence order; the `correct` and `no_lookahead` flags are kept separately.
- Look-ahead: 4 series × (6 random cuts + up to 6 cuts placed at the solution's own position changes) ×
  2 perturbations. Each perturbation replaces all bars after k with a 5× level jump (`spike`) or a
  drop to 0.2× (`crash`). Mild perturbations hardly flip a slow signal, so `shift(-1)` would often go
  unnoticed. Cuts at position changes are where a one-bar peek actually flips a position.
- Determinism: a second run on the same series must match. Otherwise a solution that uses unseeded
  randomness would be reported as look-ahead.
- If the solution crashes only on perturbed data, it fails as `exception` and no-look-ahead stays
  unknown. A run that can't be checked doesn't count as clean.

**Tests (`tests/`).** Four references (SMA cross, RSI with hysteresis, trailing stop, Donchian with
state), 4 alternative correct implementations, 35 broken solutions. Every reference passes; every
broken one gets the expected category. Look-ahead detection margin (hits/checks):

| Bug | Hits |
|---|---|
| full-sample z-score | 93/96 |
| `rolling(center=True)` | 59/96 |
| `pos.shift(-1)` | 47/96 |
| `close.shift(-1)` inside indicator | 47/96 |
| uses `close.iloc[-1]` | 47/96 |
| trailing-stop exit on next close | 24/96 |
| `bfill()` over warm-up | 12/96 |

**Something I got wrong along the way.** I expected a trailing stop with `rolling(20).max().shift(-1)`
to be flagged as look-ahead. It came back as `wrong_signals`. The cause: with `shift(-1)` the "prior
high" includes bar t, so `close[t] > high` can never be true. The strategy never trades, and its output
can't depend on the future. The check is behavioural: it flags future data that changes positions,
not code that merely mentions it. That is the right definition for the metric, but it means the
look-ahead rate measures *effective* look-ahead. I replaced the test case with one that really peeks
(an exit on the next bar's close).
