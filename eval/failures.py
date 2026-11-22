"""Write a fixed random sample of failures from a run for manual categorisation.

Usage: python -m eval.failures results/<run> [--n 30]
Writes results/<run>/failure_sample.md with task rules, the harness verdict and the code.
Manual labels go in results/<run>/failure_labels.json (written by hand while reading).
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from gen.sample import load_tasks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    run = Path(a.run)
    tasks = {t["id"]: t for t in load_tasks("all")}
    rows = [json.loads(l) for l in (run / "scores.jsonl").read_text().splitlines()]
    fails = [r for r in rows if not r["passed"] and r["sample"] == 0]
    sample = random.Random(a.seed).sample(fails, min(a.n, len(fails)))
    out = [f"# Failure sample: {run.name}\n\n{len(sample)} of {len(fails)} failures (seed {a.seed}).\n"]
    for r in sorted(sample, key=lambda r: r["task_id"]):
        t = tasks[r["task_id"]]
        rules = t["prompt"].split("Rules:")[1].split("Return only")[0].strip()
        out.append(f"\n## {r['task_id']} (difficulty {t['difficulty']}): `{r['category']}`\n\n"
                   f"{r['detail']}\n\n<details><summary>rules</summary>\n\n{rules}\n\n</details>\n\n"
                   f"```python\n{r['code'].rstrip()}\n```\n")
    (run / "failure_sample.md").write_text("".join(out))
    print(f"wrote {run / 'failure_sample.md'} ({len(sample)} failures)")


if __name__ == "__main__":
    main()
