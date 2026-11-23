"""Rejection sampling (training source B): self-generated, verified training data.

Round r:
  1. pool    : 25 fresh training tasks per in-distribution family (never reused across rounds)
  2. sample  : k=8 samples per task at T=0.8 from the current model (base in round 1,
               the previous round's adapter afterwards)                      [GPU, gen.sample]
  3. filter  : keep samples that pass every check (runs, exact match, no look-ahead,
               deterministic); dedupe by normalized AST; at most 4 per task
  4. build   : train data = kept samples of rounds 1..r, in the model's own code style
  5. train   : LoRA from the base model with the same budget as SFT-A        [GPU]

Usage:
  python -m gen.rs pool   --round 1
  python -m gen.rs filter --round 1 --run results/rs_r1_sample
"""

from __future__ import annotations

import argparse
import ast
import json
import random
import zlib
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from bench.families import ID_FAMILIES
from bench.leakage import check, load_jsonl
from gen.extract import extract
from gen.prompts import answer

ROOT = Path(__file__).resolve().parent.parent
RS = ROOT / "data" / "rs"
TASKS_PER_FAMILY = 25
MAX_PER_TASK = 4
VALID_FRAC = 0.08


def pool(round_: int, src: str = "data/train_tasks.jsonl") -> Path:
    tasks = load_jsonl(ROOT / src)
    out = []
    for fam in ID_FAMILIES:
        ts = [t for t in tasks if t["family"] == fam]
        random.Random(zlib.crc32(f"rs:{fam}".encode())).shuffle(ts)
        out += ts[(round_ - 1) * TASKS_PER_FAMILY : round_ * TASKS_PER_FAMILY]
    test = load_jsonl(ROOT / "bench/tasks_test_id.jsonl") + load_jsonl(ROOT / "bench/tasks_test_ood.jsonl")
    leaks = check(test, out)
    if leaks:
        raise SystemExit(f"leakage: {leaks[0]}")
    RS.mkdir(parents=True, exist_ok=True)
    path = RS / f"round{round_}_tasks.jsonl"
    path.write_text("".join(json.dumps(t) + "\n" for t in out))
    print(f"wrote {len(out)} tasks ({TASKS_PER_FAMILY}/family, {len(ID_FAMILIES)} families) to {path}")
    return path


def norm(code: str) -> str:
    try:
        return ast.unparse(ast.parse(code))
    except SyntaxError:
        return code


def filter_round(round_: int, run: Path, workers: int = 6) -> None:
    from harness.checks import evaluate

    tasks = {t["id"]: t for t in load_jsonl(RS / f"round{round_}_tasks.jsonl")}
    gens = load_jsonl(run / "generations.jsonl")

    def one(g):
        code, method = extract(g["text"])
        r = evaluate(code, tasks[g["task_id"]]["reference"])
        return {"task_id": g["task_id"], "sample": g["sample"], "code": code, "extract": method,
                "passed": r.passed, "category": r.category}

    with ThreadPoolExecutor(workers) as ex:
        rows = list(ex.map(one, gens))
    with open(run / "scores.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

    kept, seen = [], defaultdict(set)
    for r in sorted(rows, key=lambda r: (r["task_id"], r["sample"])):
        key = norm(r["code"])
        if r["passed"] and key not in seen[r["task_id"]] and len(seen[r["task_id"]]) < MAX_PER_TASK:
            seen[r["task_id"]].add(key)
            kept.append({"task_id": r["task_id"], "family": tasks[r["task_id"]]["family"],
                         "prompt": tasks[r["task_id"]]["prompt"], "code": r["code"]})
    (RS / f"round{round_}_kept.jsonl").write_text("".join(json.dumps(k) + "\n" for k in kept))

    fam_of = {tid: t["family"] for tid, t in tasks.items()}
    stats = {"round": round_, "run": run.name, "samples": len(rows),
             "sample_pass_rate": round(sum(r["passed"] for r in rows) / len(rows), 4),
             "kept_unique": len(kept),
             "by_family": {}}
    for fam in ID_FAMILIES:
        fr = [r for r in rows if fam_of[r["task_id"]] == fam]
        solved = {r["task_id"] for r in fr if r["passed"]}
        stats["by_family"][fam] = {"sample_pass_rate": round(sum(r["passed"] for r in fr) / max(1, len(fr)), 3),
                                   "tasks_solved": f"{len(solved)}/{TASKS_PER_FAMILY}",
                                   "kept": sum(k["family"] == fam for k in kept)}
    stats["categories"] = dict(Counter(r["category"] for r in rows).most_common())
    (run / "rs_stats.json").write_text(json.dumps(stats, indent=2) + "\n")
    build(round_)
    print(json.dumps(stats, indent=2))


def build(round_: int) -> None:
    """Training data for round r = kept samples from rounds 1..r, split by task into train/valid."""
    kept = []
    for r in range(1, round_ + 1):
        kept += load_jsonl(RS / f"round{r}_kept.jsonl")
    if not kept:
        raise SystemExit("no kept samples: nothing to train on")
    ex = lambda k: {"messages": [{"role": "user", "content": k["prompt"]},
                                 {"role": "assistant", "content": answer(k["code"])}]}
    tids = sorted({k["task_id"] for k in kept})
    random.Random(round_).shuffle(tids)
    held = set(tids[: max(1, round(len(tids) * VALID_FRAC))])
    train = [ex(k) for k in kept if k["task_id"] not in held]
    valid = [ex(k) for k in kept if k["task_id"] in held]
    random.Random(0).shuffle(train)
    out = ROOT / "data" / f"rs_r{round_}"
    out.mkdir(parents=True, exist_ok=True)
    (out / "train.jsonl").write_text("".join(json.dumps(e) + "\n" for e in train))
    (out / "valid.jsonl").write_text("".join(json.dumps(e) + "\n" for e in valid))
    fams = Counter(k["family"] for k in kept)
    (out / "stats.json").write_text(json.dumps({"rounds": list(range(1, round_ + 1)), "train": len(train),
                                                "valid": len(valid), "by_family": dict(fams)}, indent=2) + "\n")
    print(f"built {out}: {len(train)} train / {len(valid)} valid, by family {dict(fams)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["pool", "filter", "build"])
    ap.add_argument("--round", type=int, required=True)
    ap.add_argument("--run", help="results/<sampling run> (for filter)")
    a = ap.parse_args()
    if a.cmd == "pool":
        pool(a.round)
    elif a.cmd == "filter":
        filter_round(a.round, ROOT / a.run if not Path(a.run).is_absolute() else Path(a.run))
    else:
        build(a.round)


if __name__ == "__main__":
    main()
