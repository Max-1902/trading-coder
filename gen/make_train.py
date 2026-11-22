"""Generate training tasks (prompt + reference) from the train split of each in-distribution family.

Train parameters come from grid values disjoint from the test values, and
prompts use phrasings never used in test. OOD families are never generated.

Usage: python -m gen.make_train --per-family 400 --out data/train_tasks.jsonl
"""

from __future__ import annotations

import argparse
import json
import zlib
from pathlib import Path

import numpy as np

from bench.families import FAMILIES, ID_FAMILIES


def generate(per_family: int, seed: int = 0) -> list[dict]:
    tasks = []
    for name in ID_FAMILIES:
        fam = FAMILIES[name]
        rng = np.random.default_rng(zlib.crc32(f"{name}:train:{seed}".encode()))
        seen, tries = set(), 0
        while sum(t["family"] == name for t in tasks) < per_family and tries < per_family * 50:
            tries += 1
            p = fam.sample_params(rng, "train")
            phr = int(rng.choice(fam.phrasing_ids("train")))
            key = json.dumps(p, sort_keys=True) + f"|{phr}"
            if key in seen:
                continue
            seen.add(key)
            i = sum(t["family"] == name for t in tasks)
            tasks.append({"id": f"train-{name}-{i:04d}", "family": name, "difficulty": fam.difficulty(p),
                          "params": p, "phrasing": phr, "prompt": fam.prompt(p, phr),
                          "reference": fam.reference(p)})
    return tasks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-family", type=int, default=400)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="data/train_tasks.jsonl")
    a = ap.parse_args()
    tasks = generate(a.per_family, a.seed)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w") as f:
        for t in tasks:
            f.write(json.dumps(t) + "\n")
    print(f"wrote {len(tasks)} training tasks to {a.out}")


if __name__ == "__main__":
    main()
