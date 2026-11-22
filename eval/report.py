"""Compare scored runs side by side.

Usage: python -m eval.report results/run_a results/run_b ... [--baseline results/run_a]

Prints pass@1 (overall, Test-ID, Test-OOD, by difficulty) with 95% CIs, run rate,
look-ahead rate, and the failure categories. With --baseline, adds the paired
difference in pass@1 vs that run (same tasks), with its CI.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from eval.bootstrap import paired_diff_ci


def load(run: Path):
    s = json.loads((run / "summary.json").read_text())
    rows = [json.loads(l) for l in (run / "scores.jsonl").read_text().splitlines()]
    p1 = {r["task_id"]: float(r["passed"]) for r in rows if r["sample"] == 0}
    return s, p1


def c(t) -> str:
    return f"{t[0]:.2f} [{t[1]:.2f},{t[2]:.2f}]"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--baseline")
    a = ap.parse_args()
    base = load(Path(a.baseline))[1] if a.baseline else None
    hdr = f"{'run':38s} {'pass@1':18s} {'Test-ID':18s} {'Test-OOD':18s} {'d1':6s} {'d2':6s} {'d3':6s} {'runs':6s} {'look-ahead':10s}"
    print(hdr + ("  Δ vs baseline" if base else ""))
    for r in a.runs:
        s, p1 = load(Path(r))
        o, bs, bd = s["overall"], s["by_split"], s["by_difficulty"]
        line = (f"{s['run'][:38]:38s} {c(o['pass@1']):18s} {c(bs['test_id']['pass@1']) if 'test_id' in bs else '-':18s} "
                f"{c(bs['test_ood']['pass@1']) if 'test_ood' in bs else '-':18s} "
                + " ".join(f"{bd[d]['pass@1'][0]:.2f}  " if d in bd else "-     " for d in ("1", "2", "3"))
                + f" {o['run_rate'][0]:.2f}   {o['lookahead_rate_among_running'][0]:.2f}")
        if base:
            ids = sorted(set(base) & set(p1))
            m, lo, hi = paired_diff_ci([p1[i] for i in ids], [base[i] for i in ids])
            line += f"  {m:+.2f} [{lo:+.2f},{hi:+.2f}]"
        print(line)
    print()
    for r in a.runs:
        s, _ = load(Path(r))
        print(f"{s['run'][:38]:38s} {s['categories']}")


if __name__ == "__main__":
    main()
