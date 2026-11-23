# PROGRESS

Running log: decisions, what broke, how it was diagnosed. Newest state first.

## Current state (2026-11-22)

- **Phases 0–2 done.** Harness + benchmark + extractor: 167/167 tests pass.
- Benchmark: 100 test tasks (72 Test-ID over 8 families, 28 Test-OOD over Bollinger + trailing stop);
  3,200 training tasks generated; leakage check clean.
- **Phase 3 (greedy baselines): done.** Six runs: 3 models × zero-shot / 3-shot, 100 tasks each.
  Best non-fine-tuned model: Qwen 7B 3-shot at pass@1 0.53 [0.43, 0.63]. The fine-tuning target
  (Qwen 3B) gets 0.17 zero-shot and 0.26 3-shot.
- Still open for Phase 3: pass@5 with sampling (T=0.8) for the 3B and the best baseline.
- Next: Phase 4, LoRA on reference solutions (3B).

## Phase 4: LoRA on reference solutions (setup)

- **Data (`gen/make_sft.py` → `data/sft_a/`).** 3,051 train and 149 valid examples in the same
  zero-shot chat format as evaluation; valid holds out whole parameter combinations. Before writing,
  it re-runs the leakage check against the test set and verifies 160 random references with the full
  harness and the independent loop implementation (0 problems). Examples average 624 tokens (max 961).
  - *Bug caught:* the first token count said 2 tokens per example. transformers 5 returns a dict from
    `apply_chat_template(tokenize=True)`, so I was counting its keys. Now it tokenizes the rendered
    text instead.
- **Config (`train/lora_sft_a.yaml`).** Rank-16 LoRA on the top 16 of 36 layers; loss on the answer
  only (`mask_prompt`); batch 4, 250 steps (1,000 examples, ~125 per family, ~0.33 epoch);
  AdamW, lr 2e-4 with 15 warmup steps and cosine decay; max sequence length 1024; no gradient
  checkpointing. I first planned 500 steps at lr 1e-4 with checkpointing (est. 40–70 min) and cut it
  to fit a ~15–20 min training budget: half the steps, a higher learning rate, and no activation
  recomputation.
- **Model selection rule.** The final adapter is what gets evaluated. The checkpoints saved every 100
  steps may only be chosen by validation loss, never by test-set pass rate, otherwise the test set
  leaks into training decisions.

## Phase 3 results (greedy pass@1, 95% bootstrap CI over tasks)

| Model | Shots | pass@1 | Test-ID (72) | Test-OOD (28) | Runs | Look-ahead* |
|---|---|---|---|---|---|---|
| Qwen2.5-Coder-3B | 0 | 0.17 [0.10, 0.25] | 0.24 [0.14, 0.33] | 0.00 | 0.82 | 0.05 |
| Qwen2.5-Coder-3B | 3 | 0.26 [0.18, 0.35] | 0.26 [0.17, 0.36] | 0.25 [0.11, 0.43] | 0.70 | 0.00 |
| Qwen2.5-Coder-7B | 0 | 0.11 [0.05, 0.17] | 0.14 [0.07, 0.22] | 0.04 [0.00, 0.11] | 0.94 | 0.02 |
| Qwen2.5-Coder-7B | 3 | **0.53 [0.43, 0.63]** | 0.60 [0.49, 0.71] | 0.36 [0.18, 0.54] | 0.85 | 0.01 |
| Llama-3.1-8B | 0 | 0.10 [0.05, 0.16] | 0.14 [0.07, 0.22] | 0.00 | 0.55 | 0.02 |
| Llama-3.1-8B | 3 | 0.40 [0.30, 0.50] | 0.56 [0.44, 0.67] | 0.00 | 0.84 | 0.01 |

\* share of *running* solutions flagged for look-ahead. Run times (batch 2, 1 s pause): 3B 7–10 min,
7B 11–18 min, Llama 20–27 min; peak MLX memory 2.7 / 5.6 / 6.0 GB.

**What the numbers say (and don't).**
- *Zero-shot, model size barely matters.* All three sit at 0.10–0.17 with overlapping CIs. The 3B is
  nominally ahead of the 7B (0.17 vs 0.11, paired difference −0.06 [−0.13, +0.01]), which is within
  noise. The 7B almost always produces running code (94%) but gets the logic wrong (80 wrong_signals).
- *3-shot helps the bigger models a lot and the 3B barely.* 7B: +0.42; Llama: +0.30; 3B: +0.09
  [−0.02, +0.20], which is not distinguishable from 0.
- *The 3B copies the examples instead of learning from them.* One example is a stop-loss task, one
  an MA cross, one an oscillator. Splitting Test-ID by whether the family appears among the examples:

  | Model | Example families (27) | Other ID families (45) |
  |---|---|---|
  | 3B 0-shot → 3-shot | 0.26 → 0.63 | **0.22 → 0.04** |
  | 7B 0-shot → 3-shot | 0.19 → 0.70 | 0.11 → 0.53 |
  | Llama 0-shot → 3-shot | 0.15 → 0.67 | 0.13 → 0.49 |

  On families it has no example for, the 3B gets *worse* with examples. Reading its vol_filter and
  combo answers: it pastes the examples' loop-and-state scaffold into stateless tasks (adding
  position memory the rules forbid), invents warm-up cut-offs (`if t < 119: continue`) and misaligns
  `np.diff` (499 values against 500 bars → IndexError, 11 times). The bigger models take the
  conventions from the examples without copying the structure.
  This matters for the headline comparison: the fair "no fine-tuning" bar for the 3B to beat is the
  7B 3-shot at 0.53, and the 3-shot gains on the three example families are partly copying.
- *Test-OOD is weak everywhere.* Llama 3-shot gets 0/28 on Bollinger and trailing stops, while it is
  at 0.56 on Test-ID.
- *Look-ahead is rare (0–5% of running solutions)*, lower than I expected. The cases that do occur
  are not the textbook `shift(-1)` / `center=True`: see the failure taxonomy below.
- *Some families sit at 0 for every model zero-shot* (holding, z-score, Bollinger). I checked these
  for benchmark bugs before believing them. With 3 examples, z-score goes to 9/9 for both the 7B and
  Llama, so the task is solvable as written. Holding stays at 0 for all six runs. I read the best
  model's holding answers: the 7B tracks entry and exit correctly but never writes `pos[t] = state`,
  so it returns all zeros; others re-enter on the exit bar or never hold. Several models first fail at
  the same bar of holding-04 only because it is the first entry bar of that series. The prompt
  spells out the bar arithmetic (`t` to `t+H-1`, flat on `t+H`), so I count these as model errors.

**Failure taxonomy: Qwen 3B zero-shot** (30 of 83 failures, fixed random sample, each solution read
and labelled; `results/baseline_qwen3b_0shot_*/failure_labels.json`).

| Primary cause | n | What it looks like |
|---|---|---|
| State not carried / not held | 13 | checks `positions[i]` (still 0) instead of the previous state, or sets the position only on entry/exit bars and never holds it |
| Indexing / API / name errors | 9 | `df.at[i, ...]` with an integer on a DatetimeIndex (4); parameter scoped inside a helper → `NameError` (2); invented `.ewindow()`; `.rolling` on a numpy array; `Timestamp + int` |
| Warm-up handling | 4 | `if s.isnull().any(): return positions`, which returns all zeros because warm-up NaNs always exist (all 3 z-score failures); warm-up off by one bar |
| Look-ahead | 3 | `iloc[i-1]` / `iloc[i-12]` at the start of the series wraps to the **last** bar (2); writes `signals[i-1] = 1` into the past (1) |
| Holding-period logic | 1 | never holds |

Contributing causes on top: sample std where population std was specified (all 6 Bollinger cases),
RSI as a plain rolling mean instead of Wilder smoothing (5), "is above" instead of "crosses above" (4).

Against the brief's hypotheses, so far:
- *"Fine-tuning mostly fixes format/convention errors."* Format is not the problem: 99–100% of
  answers come in one clean code block and 1 of 600 hit the token cap. The 3B's main failure is
  **state handling** (13/30) and then **pandas indexing/API misuse** (9/30), with convention errors
  mostly contributing rather than primary. Fine-tuning on references will have to fix a logic
  pattern, not a format.
- *"Look-ahead bugs are common in base models."* Not supported at these rates. The ones that occur
  come from negative-index wrap-around, broadcasting the final state, and writing into past bars.
  These patterns weren't in my hand-written test set; all three are now harness unit tests
  (`negative_index_wrap`, `broadcast_final_state`, `writes_into_past_bar`), and each is flagged.

## Phase 3: pipeline and setup

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
