"""Build mlx-lm chat JSONL for training source A (prompt -> reference solution).

Usage: python -m gen.make_sft --tasks data/train_tasks.jsonl --out data/sft_a --per-family 400

- Same zero-shot prompt format as evaluation (one user turn, answer = one fenced code block).
- valid.jsonl holds whole tasks (by parameter combination) not in train.jsonl.
- Verifies a fixed sample of references with the full harness (runs, no look-ahead,
  matches the independent loop implementation) and checks token lengths with the
  model's tokenizer (tokenizer only; no model weights are loaded).
- Runs the leakage check against the test set and refuses to write on any violation.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from bench.families import FAMILIES
from bench.leakage import check, load_jsonl
from gen.prompts import answer

ROOT = Path(__file__).resolve().parent.parent


def to_example(task: dict) -> dict:
    return {"messages": [{"role": "user", "content": task["prompt"]},
                         {"role": "assistant", "content": answer(task["reference"])}]}


def verify_sample(tasks: list[dict], n: int, seed: int = 0) -> list[str]:
    from harness.checks import _hidden, evaluate, reference_positions

    problems = []
    for t in random.Random(seed).sample(tasks, min(n, len(tasks))):
        r = evaluate(t["reference"], t["reference"])
        if not r.passed:
            problems.append(f"{t['id']}: self-eval {r.category} {r.detail}")
            continue
        pos = reference_positions(t["reference"])
        fam = FAMILIES[t["family"]]
        for name, df in _hidden().items():
            if not np.array_equal(fam.alt(df, t["params"]), pos[name]):
                problems.append(f"{t['id']}: alt mismatch on {name}")
                break
    return problems


def token_lengths(examples: list[dict], model: str) -> list[int]:
    from huggingface_hub import snapshot_download
    from transformers import AutoTokenizer

    path = snapshot_download(model, local_files_only=True, allow_patterns=["*.json", "*.txt", "*.model", "*.jinja"])
    tok = AutoTokenizer.from_pretrained(path)
    texts = [tok.apply_chat_template(e["messages"], tokenize=False) for e in examples]
    return [len(ids) for ids in tok(texts, add_special_tokens=False)["input_ids"]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", default="data/train_tasks.jsonl")
    ap.add_argument("--out", default="data/sft_a")
    ap.add_argument("--per-family", type=int, default=400, help="cap per family (train+valid)")
    ap.add_argument("--valid-frac", type=float, default=0.05)
    ap.add_argument("--verify", type=int, default=160, help="references to verify with the harness")
    ap.add_argument("--model", default="mlx-community/Qwen2.5-Coder-3B-Instruct-4bit")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    tasks = load_jsonl(a.tasks)
    test = load_jsonl(ROOT / "bench/tasks_test_id.jsonl") + load_jsonl(ROOT / "bench/tasks_test_ood.jsonl")
    leaks = check(test, tasks)
    if leaks:
        raise SystemExit(f"{len(leaks)} leakage violations, e.g. {leaks[0]}")

    rng = random.Random(a.seed)
    by_fam = defaultdict(list)
    for t in tasks:
        by_fam[t["family"]].append(t)
    train, valid = [], []
    for fam, ts in sorted(by_fam.items()):
        ts = ts[: a.per_family]
        combos = sorted({json.dumps(t["params"], sort_keys=True) for t in ts})
        rng.shuffle(combos)
        held = set(combos[: max(1, round(len(combos) * a.valid_frac))])
        for t in ts:
            (valid if json.dumps(t["params"], sort_keys=True) in held else train).append(t)
    rng.shuffle(train)

    problems = verify_sample(train + valid, a.verify, a.seed)
    if problems:
        raise SystemExit("reference verification failed:\n" + "\n".join(problems[:10]))

    out = ROOT / a.out
    out.mkdir(parents=True, exist_ok=True)
    stats = {}
    for name, ts in (("train", train), ("valid", valid)):
        ex = [to_example(t) for t in ts]
        with open(out / f"{name}.jsonl", "w") as f:
            for e in ex:
                f.write(json.dumps(e) + "\n")
        lens = token_lengths(ex, a.model)
        stats[name] = {"n": len(ex), "families": dict(Counter(t["family"] for t in ts)),
                       "tokens_mean": round(float(np.mean(lens)), 1), "tokens_max": int(max(lens)),
                       "tokens_total": int(sum(lens))}
    stats["verified_references"] = a.verify
    stats["source_tasks"] = a.tasks
    (out / "stats.json").write_text(json.dumps(stats, indent=2) + "\n")
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
