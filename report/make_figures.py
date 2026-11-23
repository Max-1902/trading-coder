"""Figures and the numbers table for the report, computed from results/*/summary.json.

Usage: python -m report.make_figures   (writes report/figures/*.png and report/numbers.json)
"""

from __future__ import annotations

import glob
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
FIG = ROOT / "report" / "figures"
BLUE, ORANGE, INK, MUTED, GRID = "#3B6EA5", "#D08C2E", "#222222", "#666666", "#E3E3E3"

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "font.size": 9, "axes.titlesize": 9, "axes.labelsize": 9, "xtick.labelsize": 8.5, "ytick.labelsize": 8.5,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": INK,
    "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.6,
    "legend.frameon": False, "legend.fontsize": 8.5, "savefig.dpi": 300, "savefig.bbox": "tight",
})

RUNS = [  # (label, glob, group)
    ("Qwen 3B, 0-shot", "baseline_qwen3b_0shot_*", "base"),
    ("Qwen 3B, 3-shot", "baseline_qwen3b_3shot_*", "base"),
    ("Qwen 7B, 0-shot", "baseline_qwen7b_0shot_*", "base"),
    ("Qwen 7B, 3-shot", "baseline_qwen7b_3shot_*", "base"),
    ("Llama 8B, 0-shot", "baseline_llama8b_0shot_*", "base"),
    ("Llama 8B, 3-shot", "baseline_llama8b_3shot_*", "base"),
    ("3B + RS round 1", "ft_rs_r1_0shot_*", "ft"),
    ("3B + RS round 2", "ft_rs_r2_0shot_*", "ft"),
    ("3B + SFT-A", "ft_sft_a_0shot_*", "ft"),
]
FAMILIES = ["ma_cross", "vol_filter", "oscillator", "breakout", "zscore", "combo", "stop_tp", "holding",
            "bollinger", "trailing"]


def summary(pattern: str) -> dict:
    paths = sorted(glob.glob(str(ROOT / "results" / pattern)))
    assert len(paths) == 1, (pattern, paths)
    return json.loads((Path(paths[0]) / "summary.json").read_text())


def fig_pass_at_1(S):
    fig, ax = plt.subplots(figsize=(6.3, 3.2))
    y = np.arange(len(RUNS))[::-1]
    for (label, _, _), yi in zip(RUNS, y):
        s = S[label]["by_split"]
        for split, color, dy in (("test_id", BLUE, 0.17), ("test_ood", ORANGE, -0.17)):
            m, lo, hi = s[split]["pass@1"]
            ax.plot([lo, hi], [yi + dy] * 2, color=color, lw=1.4, solid_capstyle="round")
            ax.plot(m, yi + dy, "o", color=color, ms=4.5, mec="white", mew=0.8)
            ax.text(hi + 0.015, yi + dy, f"{m:.2f}", va="center", fontsize=7.5, color=INK)
    ax.set_yticks(y, [r[0] for r in RUNS])
    ax.set_xlim(0, 1.08)
    ax.set_xlabel("pass@1 (greedy), 95% bootstrap CI")
    ax.axhline(y[5] - 0.5, color=GRID, lw=0.8)
    ax.grid(axis="x", color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    ax.plot([], [], "o-", color=BLUE, ms=4, lw=1.4, label="Test-ID (72 tasks)")
    ax.plot([], [], "o-", color=ORANGE, ms=4, lw=1.4, label="Test-OOD (28 tasks)")
    ax.legend(loc="lower left", bbox_to_anchor=(0.0, 1.0), ncol=2)
    fig.savefig(FIG / "fig1_pass_at_1.png")
    plt.close(fig)


def fig_family_heatmap(S):
    labels = [r[0] for r in RUNS]
    M = np.array([[S[l]["by_family"][f]["pass@1"][0] for f in FAMILIES] for l in labels])
    fig, ax = plt.subplots(figsize=(6.3, 3.3))
    ax.imshow(M, cmap="Blues", vmin=0, vmax=1.15, aspect="auto")
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            ax.text(j, i, f"{M[i, j]:.2f}".replace("0.00", "0").replace("1.00", "1"), ha="center", va="center",
                    fontsize=7.5, color="white" if M[i, j] > 0.6 else INK)
    ax.set_xticks(range(len(FAMILIES)), FAMILIES, rotation=35, ha="right")
    ax.set_yticks(range(len(labels)), labels)
    ax.axvline(7.5, color=INK, lw=1.0)
    ax.axhline(5.5, color="white", lw=2.0)
    ax.text(3.5, -0.85, "in-distribution families", ha="center", fontsize=8, color=MUTED)
    ax.text(8.5, -0.85, "held out (OOD)", ha="center", fontsize=8, color=MUTED)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(length=0)
    fig.savefig(FIG / "fig2_family_heatmap.png")
    plt.close(fig)


def fig_rs_rounds(S, yields):
    fig, (a, b, c) = plt.subplots(1, 3, figsize=(6.6, 2.3), gridspec_kw={"wspace": 0.55})
    xs = [0, 1, 2]
    labs = ["3B + RS round 1", "3B + RS round 2"]
    vals = [S["Qwen 3B, 0-shot"]] + [S[l] for l in labs]
    for ax, key, title, fmt in ((a, "pass@1", "pass@1, Test-ID", "{:.2f}"),
                                (b, "lookahead_rate_among_running", "look-ahead rate\n(share of running solutions)", "{:.0%}")):
        src = [v["by_split"]["test_id"] if key == "pass@1" else v["overall"] for v in vals]
        m = [s[key][0] for s in src]
        lo = [s[key][1] for s in src]
        hi = [s[key][2] for s in src]
        ax.vlines(xs, lo, hi, color=BLUE, lw=1.4)
        ax.plot(xs, m, "o-", color=BLUE, ms=4.5, lw=1.2, mec="white", mew=0.8)
        for x, v, h in zip(xs, m, hi):
            ax.text(x, h + (0.03 if key == "pass@1" else 0.012), fmt.format(v), ha="center", va="bottom", fontsize=7.5)
        ax.set_xticks(xs, ["base", "round 1", "round 2"])
        ax.set_xlim(-0.45, 2.45)
        ax.set_title(title, loc="left")
        ax.grid(axis="y", color=GRID, lw=0.6)
        ax.set_axisbelow(True)
    sft = S["3B + SFT-A"]["by_split"]["test_id"]["pass@1"][0]
    a.axhline(sft, color=MUTED, lw=0.8, ls="--")
    a.text(-0.4, sft + 0.025, "SFT-A", va="bottom", fontsize=7.5, color=MUTED)
    a.set_ylim(0, 1.0)
    b.set_ylim(0, 0.35)
    fams = [f for f in FAMILIES[:8]]
    xi = np.arange(len(fams))
    r1 = [yields[1].get(f, 0) for f in fams]
    r2 = [yields[2].get(f, 0) for f in fams]
    c.barh(xi - 0.2, r1, height=0.38, color=BLUE, label="round 1")
    c.barh(xi + 0.2, r2, height=0.38, color=ORANGE, label="round 2")
    for i, (u, v) in enumerate(zip(r1, r2)):
        if u == 0 and v == 0:
            c.text(2, i, "none", va="center", fontsize=7.5, color=MUTED)
    c.set_yticks(xi, fams)
    c.invert_yaxis()
    c.set_title("verified training samples\nkept per family", loc="left")
    c.legend(loc="upper center", bbox_to_anchor=(0.45, -0.13), ncol=2)
    c.grid(axis="x", color=GRID, lw=0.6)
    c.set_axisbelow(True)
    fig.savefig(FIG / "fig3_rejection_sampling.png")
    plt.close(fig)


def fig_taxonomy(labels):
    counts = labels["primary_counts"]
    names = {"state": "state not carried / held", "api": "indexing / API / name error",
             "warmup": "warm-up handling", "lookahead": "look-ahead", "holding": "holding-period logic"}
    keys = sorted(counts, key=counts.get)
    fig, ax = plt.subplots(figsize=(4.6, 1.8))
    ax.barh(range(len(keys)), [counts[k] for k in keys], color=BLUE, height=0.6)
    for i, k in enumerate(keys):
        ax.text(counts[k] + 0.2, i, str(counts[k]), va="center", fontsize=8)
    ax.set_yticks(range(len(keys)), [names[k] for k in keys])
    ax.set_xlabel("failures (of 30 sampled)")
    ax.grid(axis="x", color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    fig.savefig(FIG / "fig4_failure_taxonomy.png")
    plt.close(fig)


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    S = {label: summary(pat) for label, pat, _ in RUNS}
    yields = {r: {f: v["kept"] for f, v in json.loads(
        (ROOT / f"results/rs_r{r}_sample/rs_stats.json").read_text())["by_family"].items()} for r in (1, 2)}
    labels = json.loads(Path(glob.glob(str(ROOT / "results/baseline_qwen3b_0shot_*/failure_labels.json"))[0]).read_text())
    fig_pass_at_1(S)
    fig_family_heatmap(S)
    fig_rs_rounds(S, yields)
    fig_taxonomy(labels)
    numbers = {l: {"pass@1": S[l]["overall"]["pass@1"], "test_id": S[l]["by_split"]["test_id"]["pass@1"],
                   "test_ood": S[l]["by_split"]["test_ood"]["pass@1"], "run_rate": S[l]["overall"]["run_rate"][0],
                   "lookahead": S[l]["overall"]["lookahead_rate_among_running"][0],
                   "by_difficulty": {d: v["pass@1"][0] for d, v in S[l]["by_difficulty"].items()},
                   "by_family": {f: S[l]["by_family"][f]["pass@1"][0] for f in FAMILIES}} for l in S}
    (ROOT / "report" / "numbers.json").write_text(json.dumps(numbers, indent=1) + "\n")
    print("figures:", sorted(p.name for p in FIG.glob("*.png")))


if __name__ == "__main__":
    main()


# ----------------------------------------------------------------- diagrams and extra graphs
import re  # noqa: E402

from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402


def _box(ax, x, y, w, h, title, sub="", fc="white", ec=INK):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.008,rounding_size=0.012", fc=fc, ec=ec, lw=0.8))
    ax.text(x + w / 2, y + h * (0.64 if sub else 0.5), title, ha="center", va="center", fontsize=8.6,
            fontweight="bold", color=INK)
    if sub:
        ax.text(x + w / 2, y + h * 0.3, sub, ha="center", va="center", fontsize=7.4, color=MUTED, linespacing=1.15)


def _arrow(ax, x0, y0, x1, y1, text="", curve=0.0):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=9, lw=0.8, color=INK,
                                 connectionstyle=f"arc3,rad={curve}"))
    if text:
        ax.text((x0 + x1) / 2, (y0 + y1) / 2 + 0.025, text, ha="center", va="bottom", fontsize=7, color=MUTED)


def fig_pipeline():
    fig, ax = plt.subplots(figsize=(6.6, 3.0))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    W, H = 0.18, 0.2
    _box(ax, 0.01, 0.68, W, H, "Task generator", "10 families\nparams + wordings")
    _box(ax, 0.27, 0.68, W, H, "Test set", "100 tasks\n72 ID + 28 OOD")
    _box(ax, 0.27, 0.12, W, H, "Training tasks", "3,200 tasks\n8 ID families")
    _box(ax, 0.53, 0.68, W, H, "Model", "3B / 7B / 8B\n(+ LoRA adapter)")
    _box(ax, 0.79, 0.68, W + 0.02, H, "Scoring harness", "sandbox, exact match,\nlook-ahead test")
    _box(ax, 0.53, 0.12, W, H, "LoRA training", "A: reference solutions\nB: verified own code", fc="#EEF3F9")
    _box(ax, 0.79, 0.12, W + 0.02, H, "Rejection sampling", "8 samples per task,\nkeep verified ones", fc="#FBF1E4")
    _arrow(ax, 0.19, 0.78, 0.27, 0.78)
    _arrow(ax, 0.11, 0.68, 0.27, 0.22, curve=0.25)
    _arrow(ax, 0.45, 0.78, 0.53, 0.78, "prompt")
    _arrow(ax, 0.71, 0.78, 0.79, 0.78, "code")
    _arrow(ax, 0.45, 0.22, 0.53, 0.22)
    _arrow(ax, 0.62, 0.32, 0.62, 0.68)
    ax.text(0.63, 0.5, "adapter", fontsize=7, color=MUTED, va="center")
    _arrow(ax, 0.89, 0.68, 0.89, 0.32)
    ax.text(0.90, 0.5, "pass / fail", fontsize=7, color=MUTED, va="center")
    _arrow(ax, 0.79, 0.22, 0.71, 0.22)
    ax.text(0.75, 0.03, "source B is repeated for 2 rounds", ha="center", fontsize=7.2, color=MUTED)
    ax.text(0.36, 0.50, "leakage check:\nno shared prompts,\nparameter values\nor OOD families", ha="center",
            va="center", fontsize=7, color=MUTED)
    fig.savefig(FIG / "fig0_pipeline.png")
    plt.close(fig)


def fig_lookahead():
    from harness import series as S
    from tests.fixtures import BROKEN, SMA_CROSS

    df = S.random_walk(101)
    k = 250
    crash, spike = S.perturb_after(df, k, 7, "crash"), S.perturb_after(df, k, 8, "spike")

    def run(code, d):
        ns = {}
        exec(code, ns)
        return ns["generate_signals"](d).to_numpy()

    bug = BROKEN["shift_minus_1"][1]
    fig, (a, b, c) = plt.subplots(3, 1, figsize=(6.4, 3.9), sharex=True,
                                  gridspec_kw={"height_ratios": [2.2, 1, 1], "hspace": 0.35})
    x = np.arange(len(df))
    a.plot(x, df["close"], color=INK, lw=0.9, label="original prices")
    a.plot(x[k:], crash["close"].iloc[k:], color=ORANGE, lw=0.9, label="replaced after k (crash)")
    a.plot(x[k:], spike["close"].iloc[k:], color=BLUE, lw=0.9, label="replaced after k (spike)")
    a.set_yscale("log")
    a.set_ylabel("close (log)")
    a.legend(loc="upper left", ncol=3, fontsize=7.5)
    a.set_ylim(1, 1e4)
    for ax, code, title in ((b, SMA_CROSS, "correct strategy"), (c, bug, "same strategy with pos.shift(-1)")):
        p0, p1 = run(code, df), run(code, spike)
        ax.step(x, p0, where="post", color=INK, lw=0.8, label="positions, original")
        ax.step(x, p1, where="post", color=ORANGE, lw=0.8, ls="--", label="positions, after spike")
        diff = np.flatnonzero(p0[: k + 1] != p1[: k + 1])
        verdict = "bars 0..k identical: no look-ahead" if diff.size == 0 else f"bar {diff[0]} ≤ k changed: look-ahead"
        ax.set_title(f"{title}: {verdict}", loc="left", fontsize=8.3)
        ax.set_yticks([-1, 0, 1])
        ax.set_ylim(-1.5, 1.5)
    b.legend(loc="upper right", ncol=2, fontsize=7.2, bbox_to_anchor=(1.0, 1.42))
    for ax in (a, b, c):
        ax.axvspan(0, k, color="#F1F1F1", zorder=0)
        ax.axvline(k, color=MUTED, lw=0.8, ls=":")
    a.text(0.29, 0.06, "bars 0..k: positions must not change", transform=a.transAxes, ha="center", fontsize=7.5,
           color=MUTED)
    c.set_xlabel(f"bar (cut point k = {k})")
    c.set_xlim(150, 320)
    fig.savefig(FIG / "fig_lookahead_test.png")
    plt.close(fig)


def _parse_log(path):
    txt = re.sub(r"\x1b\[[0-9;]*m", "", Path(path).read_text(errors="replace"))
    val = [(int(m.group(1)), float(m.group(2))) for m in re.finditer(r"^\s*(\d+)\s+val\s+([\d.]+)", txt, re.M)]
    tr = [(int(m.group(1)), float(m.group(2))) for m in re.finditer(r"^\s*(\d+)\s+([\d.]+)\s+[▼▲]", txt, re.M)]
    return val, tr


def fig_training():
    runs = [("SFT-A (3,051 ref. solutions)", "sft_a", BLUE), ("RS round 1 (92 own samples)", "rs_r1", ORANGE),
            ("RS round 2 (358 own samples)", "rs_r2", "#7A7A7A")]
    fig, (a, b) = plt.subplots(1, 2, figsize=(6.4, 2.3), gridspec_kw={"wspace": 0.3})
    for label, name, color in runs:
        val, tr = _parse_log(ROOT / "adapters" / name / "train.log")
        a.plot(*zip(*val), "o-", color=color, ms=3.5, lw=1.2, label=label)
        b.plot(*zip(*tr), "-", color=color, lw=1.2, label=label)
    for ax, t in ((a, "validation loss"), (b, "training loss")):
        ax.set_yscale("log")
        ax.set_xlabel("training step")
        ax.set_title(t, loc="left")
        ax.grid(color=GRID, lw=0.6)
        ax.set_axisbelow(True)
    h, l = a.get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", bbox_to_anchor=(0.5, 0.97), ncol=3, fontsize=7.6)
    fig.savefig(FIG / "fig_training_curves.png")
    plt.close(fig)


def fig_compute():
    stages = [("Pipeline checks (slices)", 0.07), ("Baselines (6 runs)", 1.54), ("SFT-A train + eval", 0.45),
              ("RS sampling (2 rounds)", 2.29), ("RS train + eval (2 rounds)", 0.95)]
    fig, ax = plt.subplots(figsize=(4.8, 1.9))
    ys = np.arange(len(stages))[::-1]
    ax.barh(ys, [s[1] for s in stages], color=BLUE, height=0.6)
    for y, (_, h) in zip(ys, stages):
        ax.text(h + 0.03, y, f"{h:.1f} h" if h >= 0.1 else f"{h * 60:.0f} min", va="center", fontsize=8)
    ax.set_yticks(ys, [s[0] for s in stages])
    ax.set_xlabel("GPU hours (total 5.3 h)")
    ax.set_xlim(0, 2.7)
    ax.grid(axis="x", color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    fig.savefig(FIG / "fig_compute.png")
    plt.close(fig)


if __name__ == "__main__":
    fig_pipeline()
    fig_lookahead()
    fig_training()
    fig_compute()
