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
- **Phase 4 (LoRA on reference solutions): done.** 3B + SFT-A adapter: pass@1 **0.67 [0.58, 0.76]**,
  Test-ID 0.93 [0.86, 0.99], Test-OOD **0.00**. That beats the best non-fine-tuned model (7B 3-shot,
  0.53) by +0.14 [+0.01, +0.27] overall, entirely through Test-ID; on Test-OOD it is far behind (0 vs 0.36).
- **Phase 5 (rejection sampling): done after 2 rounds.** pass@1 0.17 (base) → 0.24 (round 1) → 0.25
  (round 2); round 2 adds nothing measurable (+0.01 [−0.05, +0.07]), so no round 3. Self-training
  never reaches the stateful families, Test-OOD stays 0, and round 2 **raised look-ahead from 1% to 22%**
  of running solutions, although every training sample passed the look-ahead check.
- Phase 6 (stretch) skipped for now. The 7B LoRA is the open question (the only model with OOD signal).
- **Phase 7 (report): first version done.** `Trading-Coder-LoRA-Report.pdf` (13 pages, plain
  technical-report layout), README with results and figures. Figures and every number in the tables are
  generated from `results/*/summary.json` (`report/make_figures.py`, `report/build_pdf.py`), so a rerun
  of the report can't drift from the data. Own photos go in `report/photos/` and are listed in
  `PHOTOS` in `build_pdf.py`.
- While drawing the look-ahead illustration I noticed that at one fixed cut point (k = 250, crash
  replacement) the `shift(-1)` bug does *not* change bar k, because the signal at k+1 happens to be the
  same. The spike replacement at the same cut does reveal it. That's the reason the harness uses ~96 cut
  points and two replacement directions per solution; a single cut would miss it.

## Phase 5 results: rejection sampling

**Round 1 sampling** (`results/rs_r1_sample`, base 3B, 200 fresh training tasks × 8 samples at T=0.8):
only 104/1,600 samples (6.5%) pass all checks; 101 unique kept after deduplication.

| Family | Tasks solved (of 25) | Kept samples |
|---|---|---|
| ma_cross | 22 | 54 |
| vol_filter | 15 | 18 |
| combo | 12 | 25 |
| breakout | 2 | 2 |
| oscillator | 2 | 2 |
| zscore, stop_tp, holding | **0** | **0** |

- The limitation the brief anticipated, in a strong form: three of eight families, including every
  stateful one except combo, produce **no** training data, because the base model never solves them.
- Sampling at T=0.8 breaks a lot of code outright: 663 exceptions, 114 banned imports (mostly TA-Lib),
  98 syntax errors, 79 module-level errors, 42 look-ahead.
- *Bug caught:* the first filter run crashed with `RecursionError` in `ast.unparse`. One sample had
  degenerated into a ~300-deep `.dropna().ewm(span=5).mean()` chain until the token cap. I was
  normalizing every sample for deduplication; now only passing samples are normalized, there's a
  fallback for code `ast` can't round-trip, and existing scores are reused (no rescoring of 1,600
  samples).

**Round 1 training** (`adapters/rs_r1`, 92 train / 9 valid, same config and 100 steps as SFT-A, i.e.
~4 passes over the data): training loss 0.34 → 0.07, validation loss flat at 0.34–0.38 (on only 9
examples, so a weak signal; it suggests memorizing the 92 samples rather than learning anything general).

**Round 1 evaluation** (`results/ft_rs_r1_0shot_*`):

| | pass@1 | Test-ID | Test-OOD | ma_cross | combo | vol_filter | oscillator | zscore / stop_tp / holding / breakout |
|---|---|---|---|---|---|---|---|---|
| 3B base | 0.17 | 0.24 | 0.00 | 0.78 | 0.44 | 0.67 | 0.00 | 0.00 |
| 3B + RS-r1 | 0.24 [0.16, 0.33] | 0.33 | 0.00 | 0.89 | 0.67 | 0.78 | 0.33 | 0.00 |
| 3B + SFT-A | 0.67 | 0.93 | 0.00 | 1.00 | 0.67 | 1.00 | 1.00 | 1.00 / 0.78 / 1.00 / 1.00 |

- +0.07 [+0.02, +0.13] over the base model: small but the CI excludes 0. All of it comes from families
  that had training data; oscillator went 0 → 0.33 from just 2 kept samples.
- Self-training sharpens what the model can already do and cannot reach what it can't: the families
  with zero samples stay at 0, exactly where SFT-A gains most.
- **Test-OOD stays at 0.00**, so this round cannot answer whether own-style data avoids SFT-A's template
  collapse: the base 3B is already at 0 on both held-out families, and neither method moves it off the
  floor. That's a limitation of using the 3B for the OOD question; the 7B (0.36 with 3 examples) is
  the only model here with OOD signal.

**Round 2 sampling** (`results/rs_r2_sample`, round-1 adapter, 200 new tasks × 8 at T=0.8, 64 min):
406/1,600 samples pass (25%, up from 6.5%), 282 unique kept. Training set = rounds 1+2: 358 train / 25 valid.

| Family | R1 solved | R2 solved | R2 kept |
|---|---|---|---|
| ma_cross | 22/25 | 25/25 | 97 |
| vol_filter | 15/25 | 25/25 | 100 |
| combo | 12/25 | 13/25 | 52 |
| oscillator | 2/25 | 9/25 | 18 |
| breakout | 2/25 | 6/25 | 7 |
| zscore | 0/25 | 5/25 | 8 |
| stop_tp, holding | 0/25 | **0/25** | **0** |

The round-1 adapter opens z-score and lifts breakout/oscillator a little, but the data stays dominated
by the two easiest families (ma_cross + vol_filter = 69%), and stop-loss and holding still yield nothing.
One training example (1,042 tokens) was truncated to the 1,024-token limit.

**Round 2 training:** validation loss 0.307 → 0.155 over 100 steps (25 validation examples), training
loss 0.237 → 0.111; still dropping slowly, no sign of the memorization seen with SFT-A (val 0.002).

**Pass@1 per round** (zero-shot, greedy; Test-OOD 0.00 in every row):

| | pass@1 | Test-ID | ma_cross | vol_filter | combo | oscillator | breakout / zscore / stop_tp / holding | Look-ahead* |
|---|---|---|---|---|---|---|---|---|
| 3B base | 0.17 [0.10, 0.25] | 0.24 | 0.78 | 0.67 | 0.44 | 0.00 | 0 / 0 / 0 / 0 | 0.05 |
| RS round 1 | 0.24 [0.16, 0.33] | 0.33 | 0.89 | 0.78 | 0.67 | 0.33 | 0 / 0 / 0 / 0 | 0.01 |
| RS round 2 | 0.25 [0.17, 0.34] | 0.35 | 0.78 | 1.00 | 0.67 | 0.33 | 0 / 0 / 0 / 0 | **0.22** |
| SFT-A | 0.67 [0.58, 0.76] | 0.93 | 1.00 | 1.00 | 0.67 | 1.00 | 1.00 / 1.00 / 0.78 / 1.00 | 0.00 |

\* share of running solutions flagged for look-ahead.

- Round 2 vs round 1: +0.01 [−0.05, +0.07], i.e. no measurable gain. The curve is flat after one round,
  so I stopped at two (the brief allows 2–3) instead of spending another ~1.7 h of GPU time on round 3.
- 5 z-score and 6 breakout training samples did not lift those families on the test set (still 0).
- **Look-ahead jumped from 1% to 22% of running solutions (17 of 79)**: 12 trailing-stop, 3 stop-loss,
  2 breakout. Every one changes bar 0, and every one is the negative-index wraparound from the baseline
  taxonomy: a loop from `i = 0` reading `indicator.iat[i-1]`, which at `i = 0` is the **last** bar.
  - *First suspicion: the checker let look-ahead into the training data.* I checked: 18 of the 383
    training samples use `x[i-1]` in a loop starting at 0. In all but one, `x` is the model's own
    position series, whose last element is still 0 when read at `i = 0`, so it's harmless. The one
    sample that applies it to an indicator (`prior_55_highest_close.iloc[i-1]`) is guarded by a
    warm-up counter (`if warm_up_count < 55: continue`); re-running the check on it gives 0/96 hits.
    No look-ahead sample entered training.
  - *What happened instead:* the model learned an idiom (`x[i-1]` in a loop from 0) that is harmless in
    every verified sample and carried it to new tasks, where it indexes indicators without a guard.
    **Behavioural verification filters samples, not habits:** a sample can be correct and look-ahead-free
    and still teach a pattern that produces look-ahead elsewhere. SFT-A didn't show this because its
    references never index with `i-1` at `i = 0`.
- **Against the hypotheses:** "fine-tuning on look-ahead-filtered data reduces look-ahead" is
  contradicted for rejection sampling (5% → 1% → 22%) and holds for SFT-A (5% → 0%). "Families where
  the base model never passes produce no data" holds, and it's exactly the stateful families, the ones
  that matter most.

**Answer to the Phase 5 question.** Self-generated data did not avoid the Test-OOD collapse, but it
couldn't have shown that here: the 3B is at 0 on both held-out families with every method. On Test-ID,
reference SFT is far stronger (0.93 vs 0.35), because it covers the families the model can't yet solve.
Self-training only sharpens what the model can already do.

## Phase 5: rejection sampling (setup)

- **Design (`gen/rs.py`, `scripts/rs_round.sh`).** Each round draws 25 *fresh* training tasks per
  in-distribution family (200 tasks; rounds never share tasks; leakage-checked), samples k=8 answers per
  task at T=0.8 (base 3B in round 1, the previous round's adapter afterwards), keeps only samples that
  pass every check (runs, exact match, no look-ahead, deterministic), dedupes by normalized AST, keeps at
  most 4 per task, and trains a new adapter **from the base model** on the kept samples of all rounds
  so far. Training uses exactly the SFT-A config (100 steps, same LoRA, schedule and data format), so
  the comparison is between data sources, not training budgets.
- The cap of 4 per task stops easy tasks from dominating. Families where the model never passes get no
  data; the per-family yield is logged each round (`rs_stats.json`).
- **Checked before any GPU run:** a fake-generation test (reference ×2, reference + comment, broken
  solution per task) keeps exactly 1 per task and rejects the broken one. The round script resumes from
  the last finished step.
  - *Bug caught before it ran:* macOS ships bash 3.2, where `"${ADAPTER[@]}"` on an empty array under
    `set -u` aborts. That is exactly round 1 (no adapter yet), so the round would have crashed at the
    sampling step. Fixed with `${ADAPTER[@]+"${ADAPTER[@]}"}`.
- `results/pass5_qwen3b_0shot` is an interrupted pass@5 run (8/500 samples). It's not used and has to
  be deleted before pass@5 is rerun.

## Phase 4 results: LoRA on reference solutions (SFT-A)

**Training** (`adapters/sft_a`, 100 steps, ~17 min incl. validation):

| Step | 1 | 25 | 50 | 75 | 100 |
|---|---|---|---|---|---|
| Validation loss | 0.663 | 0.035 | 0.004 | 0.003 | 0.002 |

Training loss ended at 0.001. A validation loss of 0.002 on held-out parameter combinations means the
model reproduces the reference template almost token for token. The final adapter was evaluated; no
checkpoint selection was done.

**Evaluation** (`results/ft_sft_a_0shot_*`, zero-shot, greedy):

| | pass@1 | Test-ID | Test-OOD | d1 | d2 | d3 | Runs | Look-ahead |
|---|---|---|---|---|---|---|---|---|
| 3B base, 0-shot | 0.17 | 0.24 | 0.00 | 0.78 | 0.18 | 0.00 | 0.82 | 0.05 |
| 7B base, 3-shot (best baseline) | 0.53 | 0.60 | 0.36 | 0.44 | 0.64 | 0.37 | 0.85 | 0.01 |
| **3B + SFT-A, 0-shot** | **0.67** [0.58, 0.76] | **0.93** [0.86, 0.99] | **0.00** | 1.00 | 0.70 | 0.54 | 0.81 | 0.00 |

- vs 3B base: +0.50. vs 7B 3-shot: +0.14 [+0.01, +0.27], so the CI just excludes 0. The whole gain
  comes from Test-ID.
- **Test-ID by family:** 1.00 on ma_cross, oscillator, breakout, zscore, vol_filter and **holding**
  (0.00 for every baseline), 0.78 stop_tp, 0.67 combo. The state-handling failure that dominated the
  base model's taxonomy is gone on families it has seen.
- **The 5 Test-ID failures** are all slips inside a memorized template: a long/short stop-loss that
  uses `cross_down` but never defines it (the template line for the short cross is dropped, 2×);
  `series > x and x > y` on Series instead of `&` (2×, combo); one wrong combo condition.
- **Test-OOD: 0/28. It forces every new strategy into the nearest trained template.** I checked all 28:
  - *Trailing stop (0/14):* not one solution tracks the highest close since entry, which is the
    defining part of a trailing stop and appears in no trained family. 7/14 reuse the breakout
    family's channel exit (`c[t] < prior_low[t]`) instead; the rest write a stop/state skeleton
    without the peak.
  - *Bollinger (0/14):* 13/14 drop the band width k, none uses population std, and 8/14 refer to the
    z-score template's variables `m[t]`/`s[t]` without defining them (the `NameError`s).
  - The 3B base model was also at 0.00 zero-shot on these families, so SFT didn't *lose* OOD
    ability in absolute terms, but it gained none while the 7B gets 0.36 with examples.
- *Against the hypotheses:* "gains on Test-ID larger than Test-OOD" holds in the most extreme form
  (+0.69 vs +0.00). "Fine-tuning mostly fixes format/convention errors" is wrong here: format was never
  the problem. What SFT-A did was install one code template per family, including the state logic.
  That fixes in-distribution logic errors but works by recall, not by reading the rules.
- Open question for Phase 5: is the OOD collapse caused by the *reference* style (a single rigid
  template per family, val loss → 0.002), or would it happen with any SFT? Rejection-sampled data is
  in the model's own, more varied style.

## Phase 4: LoRA on reference solutions (setup)

- **Data (`gen/make_sft.py` → `data/sft_a/`).** 3,051 train and 149 valid examples in the same
  zero-shot chat format as evaluation; valid holds out whole parameter combinations. Before writing,
  it re-runs the leakage check against the test set and verifies 160 random references with the full
  harness and the independent loop implementation (0 problems). Examples average 624 tokens (max 961).
  - *Bug caught:* the first token count said 2 tokens per example. transformers 5 returns a dict from
    `apply_chat_template(tokenize=True)`, so I was counting its keys. Now it tokenizes the rendered
    text instead.
- **Config (`train/lora_sft_a.yaml`).** Rank-16 LoRA on the top 16 of 36 layers; loss on the answer
  only (`mask_prompt`); batch 4, 100 steps (400 examples); AdamW, lr 1e-4 with 10 warmup steps and
  cosine decay; max sequence length 1024; gradient checkpointing on (same gradients, lower memory).
- **Why 100 steps, not the planned 500.** A 20-step timing run (`adapters/sft_a_timing`, ~9 s/step
  with checkpointing) took validation loss from 0.696 to 0.049, with training loss 0.041. The
  references are one template per variant with the parameters copied from the prompt, so the model
  picks them up almost at once. 500 steps would have taken ~75 min for no visible gain and more
  risk of memorizing the template (bad for Test-OOD). 100 steps is still 5x past where the loss
  levelled off.
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
