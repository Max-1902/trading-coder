"""Score a run's generations.

Usage: python -m eval.run_eval results/<run>

Writes results/<run>/scores.jsonl (one row per sample) and summary.json.
Metrics (per task, then averaged over tasks, 95% bootstrap CI over tasks):
- pass@1: sample 0 passes (greedy runs have one sample).
- pass@k: unbiased estimator 1 - C(n-c, k) / C(n, k) when n >= k samples exist.
- run rate, correct rate, look-ahead rate (share of *running* samples that look ahead).
"""

from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from math import comb
from pathlib import Path

from eval.bootstrap import ci
from gen.extract import extract
from gen.sample import load_tasks
from harness.checks import evaluate


def pass_at_k(n: int, c: int, k: int) -> float:
    if n - c < k:
        return 1.0
    return 1.0 - comb(n - c, k) / comb(n, k)


def score_run(run: Path, workers: int = 6) -> dict:
    tasks = {t["id"]: t for t in load_tasks("all")}
    gens = [json.loads(l) for l in (run / "generations.jsonl").read_text().splitlines() if l.strip()]

    def one(g):
        code, method = extract(g["text"])
        r = evaluate(code, tasks[g["task_id"]]["reference"])
        return {"task_id": g["task_id"], "sample": g["sample"], "extract": method,
                "hit_max_tokens": g.get("hit_max_tokens", False), "code": code, **r.to_dict()}

    with ThreadPoolExecutor(workers) as ex:
        rows = list(ex.map(one, gens))
    with open(run / "scores.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

    by_task = defaultdict(list)
    for r in rows:
        by_task[r["task_id"]].append(r)

    def summarize(task_ids: list[str]) -> dict:
        p1, runs, corr, la_runs = [], [], [], []
        pk = defaultdict(list)
        for tid in task_ids:
            rs = sorted(by_task[tid], key=lambda r: r["sample"])
            p1.append(float(rs[0]["passed"]))
            runs.append(sum(r["runs"] for r in rs) / len(rs))
            corr.append(sum(r["correct"] for r in rs) / len(rs))
            running = [r for r in rs if r["runs"]]
            if running:
                la_runs.append(sum(r["no_lookahead"] is False for r in running) / len(running))
            n, c = len(rs), sum(r["passed"] for r in rs)
            for k in (5,):
                if n >= k:
                    pk[k].append(pass_at_k(n, c, k))
        out = {"n_tasks": len(task_ids), "pass@1": ci(p1), "run_rate": ci(runs), "correct_rate": ci(corr),
               "lookahead_rate_among_running": ci(la_runs)}
        for k, v in pk.items():
            out[f"pass@{k}"] = ci(v)
        return out

    ids = list(by_task)
    import numpy
    import pandas

    summary = {"run": run.name, "n_samples": len(rows),
               "harness": {"pandas": pandas.__version__, "numpy": numpy.__version__},
               "overall": summarize(ids)}
    for key in ("split", "family", "difficulty"):
        groups = defaultdict(list)
        for tid in ids:
            groups[str(tasks[tid][key])].append(tid)
        summary[f"by_{key}"] = {g: summarize(v) for g, v in sorted(groups.items())}
    summary["categories"] = dict(Counter(r["category"] for r in rows).most_common())
    summary["extract_methods"] = dict(Counter(r["extract"] for r in rows))
    summary["hit_max_tokens"] = sum(r["hit_max_tokens"] for r in rows)
    (run / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def fmt_ci(t) -> str:
    m, lo, hi = t
    return f"{m:.3f} [{lo:.3f}, {hi:.3f}]"


def print_summary(s: dict):
    o = s["overall"]
    print(f"== {s['run']}: {o['n_tasks']} tasks, {s['n_samples']} samples")
    for k in ("pass@1", "pass@5", "run_rate", "correct_rate", "lookahead_rate_among_running"):
        if k in o:
            print(f"  {k:30s} {fmt_ci(o[k])}")
    for split, v in s["by_split"].items():
        print(f"  pass@1 {split:23s} {fmt_ci(v['pass@1'])}  (n={v['n_tasks']})")
    print("  categories:", s["categories"])
    print("  extraction:", s["extract_methods"], " hit max tokens:", s["hit_max_tokens"])


if __name__ == "__main__":
    print_summary(score_run(Path(sys.argv[1])))
