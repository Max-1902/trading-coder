# PROGRESS

Running log: decisions, what broke, how it was diagnosed. Newest state first.

## Current state (2026-11-22)

- **Phases 0–2 done.** Harness + benchmark: 159/159 tests pass.
- Benchmark: 100 test tasks (72 Test-ID over 8 families, 28 Test-OOD over Bollinger + trailing stop);
  3,200 training tasks generated; leakage check clean.
- Phase 3 (baselines): pipeline built and checked end to end on an 8-task slice with all three
  models. Full runs not done yet. I start them myself with gentle settings, one model at a time
  (`scripts/run_baseline.sh`).

## Phase 3: baselines (in progress)

**Pipeline.** `gen/sample.py` (batched MLX generation; config, git commit and versions logged next
to the outputs; refuses to overwrite a run folder) → `gen/extract.py` (fenced, multi-block, unclosed
and no-fence completions; never edits the code) → `eval/run_eval.py` (scores, pass@1, pass@k,
run rate, look-ahead rate among running samples, 95% bootstrap CIs over tasks, by split, family and
difficulty) → `eval/report.py` (side-by-side table, paired pass@1 difference vs a baseline) →
`eval/failures.py` (fixed random sample of 30 failures to read and label by hand).
3-shot uses three fixed training tasks (MA cross, oscillator, stop/TP, all long/short) as earlier
chat turns; they are leakage-checked like all training data.

**Models.** The Qwen2.5-Coder-7B (4-bit) is downloaded. The 1.5B waits for Phase 6. To free disk
space I deleted the 8-bit Llama copy (`llama-mlx-8bit`, 5 GB); the brief rules out 8-bit and nothing
loaded it.

**pandas pinned to 2.3.3 (was 3.1.1).** Reading the first slice's failures turned up two pandas 3
changes that hit code these models write all the time:
- `positions[i]` on a DatetimeIndex Series raises `KeyError: 0` (pandas 2 falls back to positional
  access with a warning). 2 of the first 8 failures were this error, and it hid the logic errors
  underneath.
- `df['position'].iloc[i] = 1` (chained assignment) *silently does nothing* under pandas 3
  copy-on-write. The trailing-stop solution stayed all zero without any error.
The models learned pandas from pre-3.0 code, and the benchmark asks about strategy logic, not about
an API break, so the harness now runs pandas 2.3.3. All 164 tests still pass, and the rebuilt
benchmark is byte-identical. `summary.json` records the pandas/numpy versions. The first slice run
(`results/slice_qwen3b_0shot`, pandas 3) is kept for the record but is not comparable.

**Slice (8 tasks, one per family; pipeline check only, CIs are meaningless at n=8).**

| Run | pass@1 | runs | Categories |
|---|---|---|---|
| Qwen 3B, 0-shot | 0/8 | 6/8 | 6 wrong_signals, 2 exception |
| Qwen 3B, 3-shot | 3/8 | 5/8 | 3 pass, 2 exception, 2 wrong, 1 syntax |
| Qwen 7B, 0-shot | 1/8 | 7/8 | 5 wrong, 1 look-ahead, 1 exception, 1 pass |
| Llama 8B, 0-shot | 0/8 | 3/8 | 5 exception, 3 wrong |

Failures I read (all real model errors, not benchmark issues): EMA without `min_periods` (warm-up
not 0); "crosses above" written as "is above"; sample std where the prompt says population std;
state never carried between bars (checks `positions[i]`, which is always 0 at that point); early
`return` that leaves everything flat; `df.at[i, ...]` with an integer on a DatetimeIndex; `NameError`.
- *A look-ahead bug that isn't `shift(-1)`.* The 7B's stop-loss solution tracks its state correctly
  in a loop, then returns `pd.Series(position, index=df.index)`, which broadcasts the **final**
  state to every bar. Every past position then depends on the last bar. The checker flagged it at
  36/48 cuts. I hadn't put this kind of bug in the hand-written broken-solution tests.

**GPU / memory incident.** I ran the four slice runs back to back with batch size 8. With a VM and
VS Code also open, the M2 Pro sat at 100% GPU and 93 °C, with 14.2/16 GB RAM in use and 5.6 GB of
swap. Thermals stayed "Normal" (Apple Silicon throttles before damage), but the swapping was the real
problem. Changes: generation defaults are now batch size 2 and max 768 tokens (the slice's longest
output was 564), there's an optional `--pause` between batches, one model per command, and I start
GPU jobs myself. Peak MLX memory at batch 8 was 5.5 GB (7B) and 6.3 GB (Llama 8B).

## Phase 2: benchmark

**Families (`bench/families/`, one module each).** Each family maps a parameter dict to a prompt,
a pandas reference (also the SFT target for training source A), and a second implementation in
plain Python loops (`alt`), written differently on purpose.

| Family | Split | Difficulty | Variants |
|---|---|---|---|
| ma_cross | ID | 1 | close vs MA / fast vs slow; SMA / EMA; long-only / long-short |
| oscillator | ID | 2 | RSI (Wilder) / stochastic %K with hysteresis |
| breakout | ID | 2 | N-bar high/low or highest close; stop-and-reverse / M-bar channel exit |
| zscore | ID | 2 | entry/exit bands around SMA ± k·STD |
| vol_filter | ID | 2 | trend signal gated by VOL(m) < threshold |
| stop_tp | ID | 3 | SMA cross / breakout entry, fixed stop-loss + take-profit |
| holding | ID | 3 | big move / new N-bar extreme / RSI entry, hold exactly H bars |
| combo | ID | 2–3 | trend filter + MA cross (stateless) / RSI pullback with trend exit (stateful) |
| bollinger | OOD | 2 | mean reversion / breakout, population std |
| trailing | OOD | 3 | breakout / SMA cross entry, % trailing stop on closes |

Bollinger mean reversion is close to the z-score family (population std instead of sample std,
exit at the middle band), so it is a *near* OOD family; trailing stops are the *far* one. I'll keep
that in mind when reading OOD results rather than treating the two as equal.

**Prompt format.** Fixed interface preamble → one-sentence strategy summary (the part that varies
in phrasing) → exact rules → definitions of only the indicators used. Conventions spelled out in
every prompt: warm-up = 0, the strategy starts flat, the cross definition, at most one position
change per bar (no exit and re-entry on the same bar), comparison strictness, sample vs population
std, and the EMA/RSI initialisation.

**Conventions forced by float behaviour.**
- pandas' rolling std of an exactly flat window is ~6e-7, not 0 (measured on the flat-stretch series,
  both ddof=0 and 1). A rule like "if std is 0, go flat" would split correct pandas code from correct
  numpy code. So the z-score rules are in band form (`close < m - k*s`) with strict entries away from
  the mean and non-strict exits toward it. Those decisions come out the same whether s is 0 or
  6e-7.
- pandas EMA and a plain loop differ by up to 2e-13. This only matters on exact ties, so it is
  handled by the agreement check below rather than by wording.

**Build (`bench/build.py`).** Test parameters are sampled per family with fixed seeds. A task is kept
only if (1) its reference passes the full harness against itself, (2) it agrees bar for bar with the
loop implementation on all 9 hidden series, and (3) it is non-trivial: at least 6 position changes,
at least 3% exposure, and every allowed side actually used. Result: 0 agreement failures and 4
candidates rejected as trivial (an RSI system and three RSI-pullback combos that were in the market
under 2% of the time).
- To check that the agreement test has teeth, I bumped one window parameter by +1 in the loop
  implementation for every task. All 100 then mismatch.
- *First build was lopsided.* With 9 random draws per family, z-score came out 8 long-only / 1
  long-short, combo had only 2 stateful tasks, and holding had 1 short task. Now each categorical variant
  gets a quota (`ceil(n_test / n_variants)`). Result: Test-ID difficulty 1/2/3 = 9/42/21.

**Splits and leakage (`bench/leakage.py`).** Every third value on each numeric grid is test-only;
phrasings 3–4 are test-only; OOD families are never generated for training. The check fails the
build on: a normalized test prompt in train, the same family+parameter combination, any shared
numeric value per (family, parameter), a shared phrasing, or an OOD family in train.
- *First run found a real leak.* In `holding`, the key `n` meant the N-bar window in one variant
  and the RSI period in another, with different grids, so RSI period 5 (test) collided with window 5
  (train). I renamed the RSI parameter to `rsi_n` so each key has one meaning and one split. After
  that: 0 violations.
- The tests also plant leaks (a copied test task, a reworded copy with the same parameters, an OOD
  task in train) and assert that each one is caught.

**Training tasks (`gen/make_train.py`).** 400 per ID family = 3,200, all prompts distinct,
regenerated deterministically (`data/` is not committed).

## Setup decisions

- **Separate project.** Own git repo in `llama-project/trading-coder/`, next to the RAG study (`rag/`) but tracked separately (excluded from the RAG repo via `.git/info/exclude`). The only
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
