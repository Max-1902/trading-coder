"""Build the PDF report (ReportLab, plain technical-report layout).

Usage: python -m report.make_figures && python -m report.build_pdf
Writes Trading-Coder-LoRA-Report.pdf at the repository root.

Own photos/screenshots: put them in report/photos/ and list them in PHOTOS below
(file name, caption, section key). They are placed at the end of that section.
"""

from __future__ import annotations

import json
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import (Image, KeepTogether, PageBreak, Paragraph, Preformatted, SimpleDocTemplate,
                                Spacer, Table, TableStyle)

ROOT = Path(__file__).resolve().parent.parent
FIG = ROOT / "report" / "figures"
PHOTOS_DIR = ROOT / "report" / "photos"
OUT = ROOT / "Trading-Coder-LoRA-Report.pdf"

AUTHOR = "Maximilian Wojciechowski"
DATE = "November 2026"
REPO = "github.com/Max-1902/trading-coder"

# (file in report/photos/, caption, section key: "setup" | "baselines" | "sft" | "rs" | "compute")
PHOTOS: list[tuple[str, str, str]] = []

N = json.loads((ROOT / "report" / "numbers.json").read_text())

# ------------------------------------------------------------------ styles
BODY = ParagraphStyle("body", fontName="Times-Roman", fontSize=10.5, leading=13.6, alignment=TA_JUSTIFY,
                      spaceAfter=6)
TITLE = ParagraphStyle("title", fontName="Times-Bold", fontSize=17, leading=21, alignment=TA_CENTER, spaceAfter=4)
SUB = ParagraphStyle("sub", fontName="Times-Italic", fontSize=11.5, leading=14, alignment=TA_CENTER, spaceAfter=10)
AUTH = ParagraphStyle("auth", fontName="Times-Roman", fontSize=11, leading=14, alignment=TA_CENTER)
ABS_H = ParagraphStyle("absh", fontName="Times-Bold", fontSize=10.5, alignment=TA_CENTER, spaceBefore=14,
                       spaceAfter=4)
ABS = ParagraphStyle("abs", parent=BODY, fontSize=10, leading=13, leftIndent=0.45 * inch, rightIndent=0.45 * inch)
H1 = ParagraphStyle("h1", fontName="Times-Bold", fontSize=12.5, leading=15, spaceBefore=12, spaceAfter=5)
H2 = ParagraphStyle("h2", fontName="Times-Bold", fontSize=10.8, leading=13.5, spaceBefore=8, spaceAfter=3)
CAP = ParagraphStyle("cap", fontName="Times-Roman", fontSize=9, leading=11.2, alignment=TA_JUSTIFY, spaceBefore=3,
                     spaceAfter=10)
TCAP = ParagraphStyle("tcap", parent=CAP, spaceBefore=6, spaceAfter=4)
CELL = ParagraphStyle("cell", fontName="Times-Roman", fontSize=8.8, leading=10.6, alignment=TA_LEFT)
CELLB = ParagraphStyle("cellb", parent=CELL, fontName="Times-Bold")
BUL = ParagraphStyle("bul", parent=BODY, leftIndent=14, bulletIndent=3, spaceAfter=3)
CODE = ParagraphStyle("code", fontName="Courier", fontSize=7.6, leading=9.2, leftIndent=12, spaceAfter=8)

story: list = []
counters = {"fig": 0, "tab": 0}


def p(text: str, style=BODY):
    story.append(Paragraph(text, style))


def h1(text: str):
    story.append(Paragraph(text, H1))


def h2(text: str):
    story.append(Paragraph(text, H2))


def bullets(items: list[str]):
    for it in items:
        story.append(Paragraph(it, BUL, bulletText="•"))
    story.append(Spacer(1, 3))


def figure(path: Path, caption: str, width: float = 6.2 * inch):
    counters["fig"] += 1
    from reportlab.lib.utils import ImageReader

    iw, ih = ImageReader(str(path)).getSize()
    img = Image(str(path), width=width, height=width * ih / iw)
    story.append(KeepTogether([Spacer(1, 4), img,
                               Paragraph(f"<b>Figure {counters['fig']}.</b> {caption}", CAP)]))


def table(rows: list[list], caption: str, widths: list[float], bold_rows: tuple = (), align_right_from: int = 1):
    counters["tab"] += 1
    data = []
    for i, r in enumerate(rows):
        st = CELLB if i == 0 or i in bold_rows else CELL
        data.append([c if not isinstance(c, str) else Paragraph(c, st) for c in r])
    t = Table(data, colWidths=[w * inch for w in widths], repeatRows=1, hAlign="CENTER")
    t.setStyle(TableStyle([
        ("LINEABOVE", (0, 0), (-1, 0), 0.9, colors.black),
        ("LINEBELOW", (0, 0), (-1, 0), 0.5, colors.black),
        ("LINEBELOW", (0, -1), (-1, -1), 0.9, colors.black),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("TOPPADDING", (0, 0), (-1, -1), 2.2),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 2.2),
        ("LEFTPADDING", (0, 0), (-1, -1), 4),
        ("RIGHTPADDING", (0, 0), (-1, -1), 4),
    ]))
    story.append(KeepTogether([Paragraph(f"<b>Table {counters['tab']}.</b> {caption}", TCAP), t, Spacer(1, 8)]))


def code(text: str, width: int = 98):
    import textwrap

    lines = []
    for ln in text.strip("\n").splitlines():
        indent = len(ln) - len(ln.lstrip())
        lines += textwrap.wrap(ln, width, subsequent_indent=" " * (indent + 4), drop_whitespace=False) or [""]
    story.append(KeepTogether([Preformatted("\n".join(lines), CODE)]))


def photos(section: str):
    for fname, caption, sec in PHOTOS:
        if sec == section and (PHOTOS_DIR / fname).exists():
            figure(PHOTOS_DIR / fname, caption, width=5.6 * inch)


def ci(t) -> str:
    return f"{t[0]:.2f} [{t[1]:.2f}, {t[2]:.2f}]"


def f2(x) -> str:
    return f"{x:.2f}"


def footer(canvas, doc):
    canvas.saveState()
    canvas.setFont("Times-Roman", 9)
    canvas.drawCentredString(letter[0] / 2, 0.55 * inch, str(doc.page))
    canvas.restoreState()


# =================================================================== content
story.append(Spacer(1, 0.25 * inch))
p("Fine-Tuning a 3B Code Model to Write Correct Trading Strategies on a 16 GB Laptop", TITLE)
p("A student project on evaluation, LoRA fine-tuning and self-training with MLX", SUB)
p(f"{AUTHOR}<br/>{DATE} · {REPO}", AUTH)

p("Abstract", ABS_H)
p("I tested whether a small code model, fine-tuned locally on a MacBook Pro (Apple M2 Pro, 16 GB), can write "
  "trading-strategy code that does exactly what a plain-English specification says. I built a benchmark of 100 "
  "tasks from ten strategy families and scored every solution on exact agreement with a hand-written reference "
  "on synthetic price series, not on backtest profit. Each solution is also checked for look-ahead: I replace all "
  "prices after a cut point with different data and verify that earlier positions do not change. Without "
  "fine-tuning, Qwen2.5-Coder-3B passes 0.17 of the tasks, and the best untuned baseline, Qwen2.5-Coder-7B with "
  "three examples, passes 0.53. A 17-minute LoRA fine-tune of the 3B model on reference solutions raises it to "
  "0.67 [0.58, 0.76] and beats the 7B, but the entire gain is in-distribution: 0.93 on the eight trained families "
  "and 0.00 on the two held-out families, where the model forces each new strategy into the closest template it "
  "was trained on. Self-training by rejection sampling reaches only 0.25 after two rounds, produces no training "
  "data at all for the stateful families, and raised the look-ahead rate from 1% to 22% through a coding habit "
  "the model picked up from samples that were themselves verified to be free of look-ahead. Everything ran "
  "locally, with 5.3 GPU-hours in total.", ABS)

# ---------------------------------------------------------------- 1
h1("1 Introduction")
p("Language models are increasingly used to write trading code, and there are two ways that code can be wrong "
  "that a quick backtest does not show. It can simply not implement the specification (a wrong warm-up, a "
  "crossover written as a level comparison, a stop that never resets). Or it can use information from the future, "
  "for example through <font face='Courier'>shift(-1)</font> or a centered rolling window. Look-ahead makes a "
  "backtest look better, which is exactly why it is dangerous. For this reason I never use profit as the score in "
  "this project. A strategy counts as correct only if it does what the specification says, bar for bar, and does "
  "not peek at future data.")
p("This project follows my earlier student project, a local RAG system over SEC 10-K filings, and uses the same "
  "local setup and the same standard for measuring: confidence intervals on every number, a written log of what "
  "broke and how I diagnosed it, and every lost point traced to a cause. The question I set out to answer:")
p("<i>Does LoRA fine-tuning a 3B code model on its own verified outputs close the gap to, or beat, a 7–8B "
  "model that has not been fine-tuned?</i>", ParagraphStyle("q", parent=BODY, leftIndent=24, rightIndent=24))
p("What I built and found:")
bullets([
    "A benchmark of 100 strategy tasks (72 in-distribution, 28 from two held-out families) with verified "
    "references, a leakage check against the training data, and a sandboxed scoring harness with a behavioural "
    "look-ahead test.",
    "Baselines for three local models with and without examples, and a hand-labelled failure taxonomy. The 3B "
    "model's main problem is not formatting but carrying state from one bar to the next.",
    "LoRA fine-tuning on reference solutions (SFT-A) beats the 7B baseline overall, but only by recalling templates; "
    "it generalizes to none of the held-out strategy types.",
    "Self-training by rejection sampling improves only what the model can already partly do, and it can teach "
    "look-ahead even when every training sample passed the look-ahead check.",
])
figure(FIG / "fig0_pipeline.png", "Overview of the study. Test and training tasks come from the same generator but "
       "never share prompts, parameter values or held-out families. Every model answer goes through the same scoring "
       "harness; its pass/fail verdicts also filter the self-generated training data (source B).")

# ---------------------------------------------------------------- 2
h1("2 Setup")
p("All experiments ran on my own laptop, an Apple M2 Pro MacBook Pro with 16 GB of unified memory, using MLX "
  "(mlx 0.32.4, mlx-lm 0.32.2). There are no cloud GPUs and no API calls anywhere in the pipeline. I only used "
  "4-bit models and ran one model at a time. Table 1 lists the models. The Llama model is the same 4-bit "
  "checkpoint I used in my RAG project.")
table([["Role", "Model (4-bit MLX)", "Peak memory"],
       ["Fine-tuning target", "Qwen2.5-Coder-3B-Instruct", "2.7 GB (inference), 3.1 GB (batch 16)"],
       ["Larger reference, not fine-tuned", "Qwen2.5-Coder-7B-Instruct", "5.6 GB"],
       ["General model", "Llama-3.1-8B-Instruct", "6.0 GB"]],
      "Models. Peak memory is the MLX peak reported by my generation script.", [1.9, 2.1, 2.2])
p("Before fixing the model choice I checked whether a newer small coder generation existed in 4-bit MLX form. "
  "Qwen3-Coder only comes as large mixture-of-experts models that do not fit in 16 GB, so I stayed with one model "
  "family across 3B and 7B, which also keeps the size comparison clean. The harness runs pandas 2.3.3 rather than "
  "pandas 3; Section 4.4 explains why.")
photos("setup")

# ---------------------------------------------------------------- 3
h1("3 Benchmark")
h2("3.1 Task format")
p("Every task asks for one function, <font face='Courier'>generate_signals(df) -&gt; pd.Series</font>, that "
  "returns a target position of 1, 0 or −1 for every bar of an OHLCV DataFrame. The prompt has a fixed "
  "interface block, a one-sentence strategy summary (the part whose wording varies), an exact list of rules, and "
  "definitions of only the indicators the task uses. Every convention a correct solution depends on is written "
  "out: warm-up bars are 0, the strategy starts flat, what “crosses above” means, that a position changes "
  "at most once per bar, strict or non-strict comparisons, sample or population standard deviation, and how EMA "
  "and RSI are initialized. My rule was that if a convention is ambiguous, a model failing on it is the "
  "benchmark's fault, not the model's. Appendix A shows a full prompt.")
h2("3.2 Strategy families")
table([["Family", "Split", "Diff.", "What it tests"],
       ["ma_cross", "ID", "1", "close vs. MA or fast vs. slow MA; SMA/EMA; long-only or long/short"],
       ["oscillator", "ID", "2", "RSI (Wilder) or stochastic %K thresholds with hysteresis"],
       ["breakout", "ID", "2", "N-bar channel breakouts, stop-and-reverse or a shorter exit channel"],
       ["zscore", "ID", "2", "entry/exit bands around SMA ± k·STD"],
       ["vol_filter", "ID", "2", "trend signal switched off when return volatility is high"],
       ["combo", "ID", "2–3", "trend filter plus MA cross, or RSI pullback with a trend exit"],
       ["stop_tp", "ID", "3", "entry plus fixed stop-loss and take-profit from the entry price"],
       ["holding", "ID", "3", "enter on a signal, hold exactly H bars, then exit"],
       ["bollinger", "OOD", "2", "Bollinger Bands, mean reversion or breakout"],
       ["trailing", "OOD", "3", "entry plus a percentage trailing stop on the close"]],
      "The ten strategy families. ID = in-distribution (seen in training, with new parameters and wordings); "
      "OOD = held out from all training data.", [0.85, 0.5, 0.45, 4.4])
p("I generated 9 test tasks per in-distribution family and 14 per held-out family, 100 in total. Test-ID "
  "difficulty is 9 / 42 / 21 tasks at levels 1 / 2 / 3. Of the two held-out families, Bollinger mean reversion is "
  "close to the z-score family, so it is a <i>near</i> out-of-distribution test; the trailing stop is the "
  "<i>far</i> one.")
h2("3.3 Splits and leakage")
p("Every third value on each numeric parameter grid is reserved for testing, two of the five wordings per family "
  "are test-only, and the two held-out families are never generated for training. A leakage check fails the "
  "build if a normalized test prompt, a test parameter combination, any test parameter value or a test wording "
  "appears in the training data. On its first run it found a real leak. In the holding family one parameter "
  "name meant the window length in one variant and the RSI period in another, so a test-only RSI period collided "
  "with a training window length. I gave each meaning its own parameter name, and the check has reported zero "
  "violations since. The unit tests also plant leaks on purpose (a copied task, a reworded copy, an "
  "out-of-distribution task in training) and make sure each one is caught.")
h2("3.4 Making sure the references are right")
p("A benchmark is only as good as its references, so every test task had to pass three checks before I kept it: "
  "the reference passes the full harness against itself; it agrees bar for bar with a second implementation of "
  "the same family, written as plain Python loops; and it actually trades (at least six position changes, 3% "
  "exposure, and every side it is allowed to take). To show the agreement check has teeth, I changed one window "
  "length by one bar in the loop implementation for every task, and all 100 then failed it. Two numerical issues "
  "came up while building this:")
bullets([
    "<b>Float ties.</b> Two correct implementations can disagree when fast SMA equals slow SMA exactly in a flat "
    "stretch, only because they add numbers in a different order. I round all synthetic prices to a 1/128 tick, "
    "which is exact in floating point, so rolling sums come out bit-identical however they are computed.",
    "<b>pandas' rolling standard deviation of a perfectly flat window is about 6×10<super>−7</super>, not 0.</b> "
    "A rule such as “if the standard deviation is 0, go flat” would therefore separate correct pandas code "
    "from correct numpy code. I wrote the z-score rules as band comparisons instead "
    "(<font face='Courier'>close &lt; m - k*s</font>), with strict entries and non-strict exits, which give the same "
    "decision whether s is 0 or the tiny residue.",
])
p("The first build was also lopsided (eight long-only z-score tasks and one long/short), so I added a quota per "
  "categorical variant to the sampler.")

# ---------------------------------------------------------------- 4
h1("4 Scoring harness")
h2("4.1 Sandbox")
p("Generated code is first checked statically: only numpy, pandas and math may be imported, and builtins like "
  "<font face='Courier'>eval</font>, <font face='Courier'>open</font> and "
  "<font face='Courier'>__import__</font>, dunder attributes and pandas/numpy file I/O are rejected. It then "
  "runs in a separate Python process under macOS <font face='Courier'>sandbox-exec</font> with network access and "
  "file writes denied (I verified that a socket connection fails) and a 30-second time limit. The child process "
  "returns plain JSON, so the scoring process never unpickles anything produced by untrusted code. Each failure "
  "gets one category: syntax error, banned import, exception, timeout, wrong output type, length, index or "
  "values, non-determinism, look-ahead, or wrong signals.")
h2("4.2 Correctness")
p("A solution is correct if its positions equal the reference exactly on nine hidden synthetic series of 500 "
  "bars each, generated with fixed seeds: random walks, strong trends, mean-reverting (Ornstein-Uhlenbeck) "
  "prices, regime switches, flat stretches with exactly constant prices, and a series with price gaps and missing "
  "dates.")
h2("4.3 Look-ahead test")
p("The look-ahead test is behavioural. For four of the series I pick cut points k and replace every bar after k "
  "with very different data: a jump to five times the price or a crash to a fifth, after which the path keeps "
  "moving. If any position at bars 0 to k changes, the solution used the future. Mild perturbations hardly flip "
  "a slow signal, which is why the replacement is extreme. Half of the cut points are placed right where the "
  "solution's own position changes, because that is where a one-bar peek actually shows up. Each solution gets "
  "about 96 checks, plus a second run on unchanged data so that unseeded randomness is not mistaken for "
  "look-ahead. The test is independent of correctness: a solution can match the reference on the hidden series "
  "and still be flagged.")
p("I validated the harness with 38 deliberately broken solutions and 4 alternative correct ones. Every reference "
  "and alternative passes, and every broken solution gets the category I expected. Look-ahead bugs are caught "
  "at between 12 and 93 of the 96 cut points. One expectation of mine was wrong: I had assumed that a trailing "
  "stop using <font face='Courier'>rolling(20).max().shift(-1)</font> would be flagged. It was scored as wrong "
  "signals instead, because with that shift the entry condition can never be true, so the strategy never trades "
  "and its output does not depend on the future. The test measures look-ahead that changes positions, not code "
  "that merely mentions future data. I kept that definition and replaced the test case with one that really "
  "peeks.")
figure(FIG / "fig_lookahead_test.png", "The look-ahead test on one hidden series. Top: prices after the cut "
       "point k are replaced by a crash or a spike. Middle: a correct SMA-crossover strategy keeps identical "
       "positions up to k. Bottom: the same strategy with <font face='Courier'>pos.shift(-1)</font> changes its "
       "position at bar k, so it is flagged. With the crash replacement this particular cut does not reveal the "
       "bug, which is why each solution is tested at about 96 cut points in both directions.")
h2("4.4 Metrics, and a pandas version problem")
p("The primary metric is pass@1 with greedy decoding: the solution runs, is correct, and shows no look-ahead. I "
  "also report the run rate and the look-ahead rate among solutions that run. All numbers come with 95% "
  "percentile-bootstrap confidence intervals over tasks (10,000 resamples), and comparisons between conditions "
  "use paired differences on the same tasks. With 100 tasks, differences below about 0.1 are usually noise.")
p("Before the full runs I ran every model end to end on an 8-task slice and read every failure. Two of the first "
  "eight failures were pandas 3 behaviour rather than model mistakes: <font face='Courier'>positions[i]</font> "
  "on a DatetimeIndex series raises <font face='Courier'>KeyError: 0</font> in pandas 3 (pandas 2 falls back to "
  "positional access), and chained assignment such as <font face='Courier'>df['p'].iloc[i] = 1</font> silently "
  "does nothing under pandas 3's copy-on-write. The models learned pandas from code written before 3.0, and the "
  "benchmark asks about strategy logic, not about an API break, so I pinned the harness to pandas 2.3.3. All "
  "references and tests behave identically under it.")

# ---------------------------------------------------------------- 5
h1("5 Baselines")
b = {k: N[k] for k in N}
rows = [["Model", "Shots", "pass@1", "Test-ID", "Test-OOD", "Runs", "Look-ahead"]]
for lab in ["Qwen 3B, 0-shot", "Qwen 3B, 3-shot", "Qwen 7B, 0-shot", "Qwen 7B, 3-shot", "Llama 8B, 0-shot",
            "Llama 8B, 3-shot"]:
    m, shots = lab.split(", ")
    r = b[lab]
    rows.append([m, shots.split("-")[0], ci(r["pass@1"]), f2(r["test_id"][0]), f2(r["test_ood"][0]),
                 f2(r["run_rate"]), f"{r['lookahead']:.0%}"])
table(rows, "Baselines without fine-tuning, greedy decoding, 100 tasks. Look-ahead is the share of running "
      "solutions flagged by the look-ahead test.", [1.05, 0.5, 1.35, 0.7, 0.75, 0.55, 0.85], bold_rows=(4,))
figure(FIG / "fig1_pass_at_1.png", "pass@1 on Test-ID and Test-OOD for every condition, with 95% bootstrap "
       "confidence intervals. Rows below the line are fine-tuned 3B models (Sections 6 and 7).")
p("<b>Without examples, model size barely matters.</b> All three models land between 0.10 and 0.17 with "
  "overlapping intervals. The 7B model nearly always produces code that runs (94%) but gets the logic wrong.")
p("<b>Three examples help the larger models a lot and the 3B model hardly at all</b>: +0.42 for the 7B, +0.30 "
  "for Llama, and +0.09 [−0.02, +0.20] for the 3B. Splitting Test-ID by whether a family appears among the "
  "three examples (MA cross, oscillator, stop-loss) shows why. On the families it has no example for, the 3B "
  "model drops from 0.22 to 0.04, while the 7B rises from 0.11 to 0.53. Reading its answers, the 3B model pastes "
  "the examples' loop-and-state structure even into stateless tasks, invents warm-up cut-offs, and misaligns "
  "arrays. It copies the examples instead of learning the conventions from them. The best untuned result, and "
  "the bar for the fine-tuned 3B model, is therefore the 7B model with three examples at 0.53.")
h2("5.1 Why the 3B model fails")
p("I sampled 30 of the 83 failures of the 3B model without examples (fixed seed) and labelled each one by reading "
  "the code.")
figure(FIG / "fig4_failure_taxonomy.png", "Primary cause of 30 sampled failures of Qwen 3B, zero-shot, labelled "
       "by hand.", width=4.4 * inch)
bullets([
    "<b>State not carried or held (13).</b> The code checks <font face='Courier'>positions[i]</font>, which is "
    "still 0 at that point, instead of the previous state, or it sets a position only on the entry and exit bars "
    "and never holds it in between.",
    "<b>Indexing, API and name errors (9)</b>, such as <font face='Courier'>df.at[i, ...]</font> with an integer on a "
    "DatetimeIndex, a parameter scoped inside a helper function, or invented methods like "
    "<font face='Courier'>.ewindow()</font>.",
    "<b>Warm-up handling (4)</b>, for example <font face='Courier'>if s.isnull().any(): return positions</font>, which "
    "returns all zeros because warm-up values are always missing.",
    "<b>Look-ahead (3).</b> None of them is a textbook <font face='Courier'>shift(-1)</font>. Two read "
    "<font face='Courier'>iloc[i-1]</font> at <font face='Courier'>i = 0</font>, which wraps around to the "
    "<i>last</i> bar, and one writes a position into the previous bar. I added all three patterns, plus the 7B's "
    "habit of broadcasting the final state to every bar, as new harness tests.",
])
p("Formatting was not the problem: 99–100% of answers arrived as one clean code block, and only one of 600 "
  "hit the token limit.")
photos("baselines")

# ---------------------------------------------------------------- 6
h1("6 Fine-tuning on reference solutions (SFT-A)")
p("For the first training source I fine-tuned the 3B model on pairs of prompt and reference solution: 3,051 "
  "training and 149 validation examples from the training split, in exactly the zero-shot chat format used for "
  "evaluation. I verified that the tokenized evaluation prompt is identical to the training prompt prefix (506 "
  "tokens) and that the loss covers only the answer. I used rank-16 LoRA on the top 16 of 36 layers, batch "
  "size 4, AdamW and learning rate 10<super>−4</super> with warmup and cosine decay.")
p("I had planned 500 steps. A 20-step timing run changed that decision: validation loss had already dropped "
  "from 0.70 to 0.05, so 500 steps (about 75 minutes) would mainly have added time and the risk of memorizing "
  "the template. I trained for 100 steps (17 minutes). Validation loss went 0.66, 0.035, 0.004, 0.003, 0.002 at "
  "steps 1, 25, 50, 75 and 100. I evaluated the final adapter only; I decided in advance never to pick a "
  "checkpoint by test score.")
figure(FIG / "fig_training_curves.png", "Training curves, log scale. On reference solutions the validation "
       "loss falls to 0.002 within 100 steps, i.e. the model learns to reproduce the template almost exactly. On its "
       "own verified samples the validation loss stays flat (round 1, 92 examples) or falls slowly (round 2).")
s = N["3B + SFT-A"]
table([["Condition", "pass@1", "Test-ID", "Test-OOD", "Diff. 1 / 2 / 3"],
       ["Qwen 3B, 0-shot", ci(N["Qwen 3B, 0-shot"]["pass@1"]), f2(N["Qwen 3B, 0-shot"]["test_id"][0]), "0.00",
        " / ".join(f2(N["Qwen 3B, 0-shot"]["by_difficulty"][d]) for d in "123")],
       ["Qwen 7B, 3-shot", ci(N["Qwen 7B, 3-shot"]["pass@1"]), f2(N["Qwen 7B, 3-shot"]["test_id"][0]),
        f2(N["Qwen 7B, 3-shot"]["test_ood"][0]), " / ".join(f2(N["Qwen 7B, 3-shot"]["by_difficulty"][d]) for d in "123")],
       ["3B + SFT-A, 0-shot", ci(s["pass@1"]), ci(s["test_id"]), f2(s["test_ood"][0]),
        " / ".join(f2(s["by_difficulty"][d]) for d in "123")]],
      "SFT-A compared with the untuned 3B model and the best baseline.", [1.4, 1.35, 1.35, 0.75, 1.3], bold_rows=(3,))
p("The fine-tuned 3B model beats the 7B baseline by +0.14 [+0.01, +0.27], so the interval only just excludes "
  "zero, and the whole gain comes from Test-ID (0.93). It solves families every baseline failed, including all "
  "holding-period tasks, so the state-handling failure from Section 5.1 is gone on families it has seen. Its five "
  "Test-ID failures are slips inside a memorized template, such as a short-side variable that is used but never "
  "defined.")
p("<b>On the held-out families it scores 0 of 28, and I read all 28 solutions to find out why.</b> The model does "
  "not read the new rules; it forces each new strategy into the closest template it was trained on. Not one of "
  "the 14 trailing-stop solutions tracks the highest close since entry, which is the defining part of a trailing "
  "stop; seven reuse the breakout family's channel exit instead. Of the 14 Bollinger solutions, 13 drop the band "
  "width, none uses the population standard deviation, and 8 refer to the z-score template's variables "
  "<font face='Courier'>m</font> and <font face='Courier'>s</font> without defining them. A validation loss of "
  "0.002 tells the same story: the model reproduces the reference template almost token for token. The untuned 3B "
  "model was also at 0 on these families, so fine-tuning did not lose anything there, but it gained nothing either.")
figure(FIG / "fig2_family_heatmap.png", "pass@1 per strategy family. SFT-A is close to perfect on the eight trained "
       "families and at zero on the two held-out ones.", width=5.7 * inch)
photos("sft")

# ---------------------------------------------------------------- 7
h1("7 Self-training by rejection sampling")
p("The second training source uses the model's own code. In each round the model samples 8 answers at "
  "temperature 0.8 for 200 fresh training tasks (25 per family, never reused across rounds). I keep only samples "
  "that pass every check, remove duplicates by normalized syntax tree, keep at most four per task, and train a "
  "new adapter from the base model on all kept samples so far, with exactly the SFT-A configuration. Round 1 "
  "samples from the base model, round 2 from the round-1 adapter.")
figure(FIG / "fig3_rejection_sampling.png", "Rejection sampling. Left: Test-ID pass@1 after each round, with "
       "SFT-A for reference. Middle: look-ahead rate among running solutions. Right: verified training samples "
       "per family. The stop-loss and holding families produced none in either round.")
r1, r2 = N["3B + RS round 1"], N["3B + RS round 2"]
table([["", "Samples passing", "Kept", "pass@1", "Test-ID", "Look-ahead"],
       ["Base 3B", "–", "–", ci(N["Qwen 3B, 0-shot"]["pass@1"]), f2(N["Qwen 3B, 0-shot"]["test_id"][0]),
        f"{N['Qwen 3B, 0-shot']['lookahead']:.0%}"],
       ["Round 1", "104 / 1,600 (6.5%)", "101", ci(r1["pass@1"]), f2(r1["test_id"][0]), f"{r1['lookahead']:.0%}"],
       ["Round 2", "406 / 1,600 (25%)", "282", ci(r2["pass@1"]), f2(r2["test_id"][0]), f"{r2['lookahead']:.0%}"]],
      "Rejection sampling per round. Test-OOD is 0.00 in every row. Round 2 trains on the kept samples of both "
      "rounds (358 training examples).", [0.75, 1.3, 0.5, 1.35, 0.65, 0.8])
p("Round 1 gives a small but real improvement of +0.07 [+0.02, +0.13] over the base model, entirely on families "
  "it could already partly solve. Round 2 adds nothing measurable (+0.01 [−0.05, +0.07]), so I stopped after "
  "two rounds instead of spending another 1.7 hours on a third. The method has a built-in limit that shows up "
  "clearly here: the base model never solves a stop-loss or holding task, so those families never get a single "
  "training example. Self-training sharpens what a model can already do; it cannot teach what the model never "
  "gets right. This is why reference fine-tuning is far stronger in-distribution (0.93 against 0.35).")
h2("7.1 Look-ahead learned from clean data")
p("After round 2 the look-ahead rate jumped from 1% to 22% of running solutions (17 of 79), mostly on "
  "trailing-stop and stop-loss tasks, and every flagged solution changes bar 0. All 17 use the same pattern: a loop "
  "from <font face='Courier'>i = 0</font> that reads an indicator at <font face='Courier'>[i-1]</font>, which at "
  "<font face='Courier'>i = 0</font> is the last bar of the series.")
code("""
for i in range(len(df)):
    if position.iat[i-1] == 0:                          # own position: harmless
        if df['close'].iat[i] > rolling_max_12.iat[i-1]:    # at i = 0: the LAST bar
            position.iat[i] = 1
""")
p("My first suspicion was that the checker had let look-ahead samples into the training data, so I audited it. "
  "Of the 383 training samples, 18 use <font face='Courier'>x[i-1]</font> in a loop starting at 0. In all but one, "
  "<font face='Courier'>x</font> is the model's own position series, whose last element is still 0 when it is "
  "read at <font face='Courier'>i = 0</font>, which is harmless. The one sample that applies it to an indicator is "
  "protected by a warm-up counter, and re-running the check on it gives 0 of 96 hits. No look-ahead sample "
  "entered training. What happened instead is that the model learned an idiom that was harmless in every verified "
  "sample and carried it over to new tasks, where it reads indicators without a guard. <b>Verifying samples does "
  "not verify the habits they teach.</b> SFT-A did not show this, because its references never index with "
  "<font face='Courier'>i-1</font> at <font face='Courier'>i = 0</font>.")
photos("rs")

# ---------------------------------------------------------------- 8
h1("8 Discussion")
p("<b>The headline question.</b> Yes, a fine-tuned 3B model beats the untuned 7B model overall (0.67 against "
  "0.53), but only through recall on the strategy types it was trained on. Training on the model's own verified "
  "outputs, which is what the question actually asked about, does not close the gap: it reaches 0.25.")
table([["Hypothesis from my project plan", "Result"],
       ["Fine-tuning mostly fixes format and convention errors, less so logic errors.",
        "Not supported. Format was never the problem (99–100% clean code blocks). SFT-A fixed logic, including "
        "state handling, but by recalling one template per family."],
       ["Stateful strategies stay hard for everyone.",
        "True for every untuned model and for self-training (stop-loss and holding stay at 0). Not true for SFT-A, "
        "which solves all holding tasks."],
       ["Look-ahead bugs are common in base models; training on look-ahead-filtered data reduces them.",
        "Rare in base models (1–5%). Reduced by SFT-A (to 0%), but raised by self-training (to 22%)."],
       ["Gains on Test-ID are larger than on Test-OOD.",
        "True in the most extreme form: +0.69 on Test-ID and +0.00 on Test-OOD for SFT-A."]],
      "The four hypotheses I wrote down before running anything, and what the results say.", [2.6, 3.8])
p("Two things I take away beyond the numbers. First, the harness found its own bugs and the benchmark's: the "
  "pandas 3 confounder, the leakage between parameter meanings, and the float ties would each have produced "
  "wrong conclusions if I had only looked at aggregate scores. Reading individual failures was the most useful "
  "thing I did. Second, “verified” training data is only verified for the property you check. The "
  "look-ahead habit passed every behavioural test it was given and still taught the model something unsafe.")

# ---------------------------------------------------------------- 9
h1("9 Limitations and future work")
bullets([
    "The tasks are synthetic and templated, and the training data comes from the same templates. The 0.93 on "
    "Test-ID is mostly recall, not understanding.",
    "100 tasks give wide intervals, and the out-of-distribution conclusions rest on only two families (28 tasks).",
    "Each condition was trained once with one seed, so I cannot say how much results vary between runs.",
    "The 3B model is at 0 on both held-out families with every method, so this study cannot answer whether "
    "self-generated data avoids template collapse. A LoRA fine-tune of the 7B model, the only model with signal "
    "there (0.36 with examples), is the natural next experiment, but it needs more memory and time than I had.",
    "The prompts are much more precise than real specifications. A tier of vaguer prompts would be more realistic.",
])
p("With more time, I would first add four to six more held-out families, train each condition with three "
  "seeds, and run the 7B fine-tune.")

# ---------------------------------------------------------------- 10
h1("10 Compute and effort")
p("Everything in this report ran on my laptop. Table 7 lists the GPU time of every run, taken from the timing "
  "logs that my scripts write next to each result. Scoring and the look-ahead checks run on the CPU and are not "
  "included.")
figure(FIG / "fig_compute.png", "GPU time per stage. Sampling for rejection sampling (3,200 completions) "
       "took the largest share.", width=4.6 * inch)
table([["Stage", "Runs", "Model completions", "GPU time"],
       ["Pipeline checks (8-task slices)", "5", "40", "0.1 h"],
       ["Baselines (3 models × 0/3-shot)", "6", "600", "1.5 h"],
       ["SFT-A: timing run + training + evaluation", "3", "100", "0.5 h"],
       ["Rejection sampling: sampling (2 rounds)", "2", "3,200", "2.3 h"],
       ["Rejection sampling: training + evaluation (2 rounds)", "4", "200", "0.9 h"],
       ["Total", "20", "4,140", "5.3 h"]],
      "GPU time per stage, from the run logs. All runs on an Apple M2 Pro with 16 GB.", [3.0, 0.6, 1.3, 0.9],
      bold_rows=(6,))
p("Not every run went smoothly. During the first slice runs I used batch size 8 with long few-shot prompts while "
  "a virtual machine and my editor were also open. The GPU sat at 100% and 93 °C, memory use reached 14.2 of "
  "16 GB, and macOS swapped 5.6 GB to disk. I stopped the runs, reduced the batch size to 2, capped answers at 768 "
  "tokens (the longest so far had been 564), added a pause between batches, and from then on started every GPU job "
  "myself, one model at a time. Peak memory stayed between 2.7 and 6.0 GB afterwards. Table 8 lists the problems "
  "I ran into and how I fixed them.")
table([["Problem", "How I found it", "What I changed"],
       ["Test expected look-ahead for an inert shift(-1)", "harness test failed", "kept the behavioural definition, fixed the test"],
       ["Float ties between correct implementations", "design review of SMA ties", "prices on a 1/128 tick"],
       ["pandas std of a flat window is 6e-7, not 0", "measured on the flat series", "rules in band form"],
       ["Parameter name with two meanings leaked test values", "leakage check, first run", "separate parameter names"],
       ["Lopsided task variants", "variant counts after first build", "quota per variant"],
       ["pandas 3 KeyError and silent chained assignment", "read the first 8 failures", "pinned pandas 2.3.3"],
       ["GPU at 100%, 5.6 GB swap", "system monitor during runs", "batch 2, one model per run"],
       ["Token counts of 2 per example", "sanity check of data stats", "count tokens from rendered text"],
       ["500 planned steps not needed", "20-step timing run", "trained 100 steps"],
       ["Round script would crash on macOS bash 3.2", "checked before running", "bash-3.2-safe array syntax"],
       ["RecursionError in deduplication", "round 1 stopped", "normalize only passing samples"],
       ["Look-ahead jumped to 22%", "round 2 results", "audited training data; checker was clean"]],
      "Problems I ran into, how I found them, and what I changed. Details are in PROGRESS.md.", [2.35, 1.95, 2.1])
import subprocess  # noqa: E402

_commits = int(subprocess.check_output(["git", "rev-list", "--count", "HEAD"], cwd=ROOT, text=True))
_loc = sum(len(f.read_text().splitlines()) for f in ROOT.rglob("*.py") if ".venv" not in f.parts)
p(f"The experiments took one long working session, from the first commit at 13:44 to the last results at 23:45. "
  f"The repository has {_commits} commits, about {round(_loc, -2):,} lines of Python, 167 tests, and a running log "
  f"(PROGRESS.md) in which every decision and every problem is written down as it happened.")
photos("compute")

h1("Tools and acknowledgements")
p("I used Claude Code as an AI coding assistant while implementing the harness, the benchmark generators and the "
  "analysis scripts. I designed the study, decided how to handle each problem that came up, ran every model job "
  "on my machine, and checked the results. The models, the training and all evaluation run locally; no external "
  "model is called anywhere in the pipeline.")

h1("Reproducibility")
p(f"Code, benchmark, raw generations, scores and training logs: {REPO}. Each result folder contains the run "
  "configuration, the git commit, library versions and timing.")
code("""
uv venv --python 3.12 .venv && uv pip install --python .venv/bin/python -r requirements.txt
.venv/bin/python -m pytest -q                       # 167 tests: harness, benchmark, extractor
.venv/bin/python -m gen.make_train --per-family 400 # training tasks
.venv/bin/python -m bench.leakage data/train_tasks.jsonl
scripts/run_baseline.sh qwen7b 3 --pause 1          # one baseline (GPU)
.venv/bin/python -m gen.make_sft && scripts/train_lora.sh train/lora_sft_a.yaml sft_a
scripts/eval_adapter.sh sft_a
scripts/rs_round.sh 1 && scripts/rs_round.sh 2      # rejection sampling
""")

# ---------------------------------------------------------------- appendix
story.append(PageBreak())
h1("Appendix A: Example task")
task = next(json.loads(l) for l in (ROOT / "bench/tasks_test_id.jsonl").read_text().splitlines()
            if json.loads(l)["id"] == "holding-04")
p("The complete prompt of test task holding-04 (difficulty 3), followed by its reference solution. Every model "
  "in the baselines failed this task; SFT-A solved it.")
code(task["prompt"])
code(task["reference"])

def glue_headings(flow: list) -> list:
    """Before each heading, break the page only if the heading and the element after it don't fit."""
    import io

    from reportlab.pdfgen.canvas import Canvas
    from reportlab.platypus import CondPageBreak

    canv = Canvas(io.BytesIO())
    width, max_h = letter[0] - 2 * inch, letter[1] - 1.8 * inch
    def height(fl) -> float:
        if hasattr(fl, "_content"):  # KeepTogether reports a huge height on purpose; measure its parts
            return sum(height(c) for c in fl._content)
        h = fl.wrapOn(canv, width, max_h)[1]
        style = getattr(fl, "style", None)
        return h + (getattr(style, "spaceBefore", 0) + getattr(style, "spaceAfter", 0) if style else 0)

    is_heading = lambda fl: isinstance(fl, Paragraph) and fl.style.name in ("h1", "h2")  # noqa: E731
    out = []
    for i, f in enumerate(flow):
        if is_heading(f) and (i == 0 or not is_heading(flow[i - 1])):
            j, need = i, 0.0
            while j < len(flow) and is_heading(flow[j]):  # the heading run ...
                need += height(flow[j])
                j += 1
            if j < len(flow):  # ... plus the first element after it
                need += height(flow[j])
            out.append(CondPageBreak(min(need + 4, max_h)))
        out.append(f)
    return out


doc = SimpleDocTemplate(str(OUT), pagesize=letter, leftMargin=1 * inch, rightMargin=1 * inch, topMargin=0.9 * inch,
                        bottomMargin=0.9 * inch, title="Fine-Tuning a 3B Code Model to Write Correct Trading Strategies",
                        author=AUTHOR)
doc.build(glue_headings(story), onFirstPage=footer, onLaterPages=footer)
print(f"wrote {OUT}")
