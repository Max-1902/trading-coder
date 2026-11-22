"""Build the test benchmark: sample test-split parameters per family, verify, write JSONL.

A task is kept only if its reference
1. passes the full harness against itself (runs, no look-ahead, deterministic),
2. agrees bar-for-bar with the family's independent loop implementation on every
   hidden series (rejects tasks that hinge on float ties), and
3. is non-trivial: trades on the hidden series, using every side it is allowed.

Usage: python -m bench.build
"""

from __future__ import annotations

import json
import zlib
from collections import Counter
from pathlib import Path

import numpy as np

from bench.families import FAMILIES
from harness.checks import _hidden, evaluate, reference_positions

BENCH = Path(__file__).parent
MIN_CHANGES = 6        # position changes summed over the hidden series
MIN_EXPOSURE = 0.03    # fraction of bars not flat


def family_seed(name: str, split: str) -> int:
    return zlib.crc32(f"{name}:{split}".encode())


def verify(fam, p) -> tuple[bool, str, dict]:
    ref = fam.reference(p)
    r = evaluate(ref, ref)
    if not r.passed:
        return False, f"self-eval {r.category}: {r.detail}", {}
    pos = reference_positions(ref)
    for name, df in _hidden().items():
        alt = fam.alt(df, p)
        if not np.array_equal(alt, pos[name]):
            i = int(np.flatnonzero(alt != pos[name])[0])
            return False, f"alt mismatch on {name} at bar {i} (ref {pos[name][i]}, alt {alt[i]})", {}
    allp = np.concatenate(list(pos.values()))
    changes = int(sum((np.diff(v) != 0).sum() for v in pos.values()))
    exposure = float((allp != 0).mean())
    sides = set(np.unique(allp).tolist()) - {0}
    want = {1, -1} if p.get("side") == "long_short" else ({-1} if p.get("direction") == "short" else {1})
    if changes < MIN_CHANGES or exposure < MIN_EXPOSURE or not want <= sides:
        return False, f"trivial: {changes} changes, exposure {exposure:.3f}, sides {sorted(sides)}", {}
    return True, "", {"changes": changes, "exposure": round(exposure, 4),
                      "long_frac": round(float((allp == 1).mean()), 4),
                      "short_frac": round(float((allp == -1).mean()), 4)}


def variant(p: dict) -> tuple:
    """Categorical signature of a task (long-only vs long/short, indicator type, ...)."""
    return tuple(v for _, v in sorted(p.items()) if isinstance(v, str))


def build_family(fam, max_tries: int = 2000) -> tuple[list[dict], Counter]:
    rng = np.random.default_rng(family_seed(fam.name, "test"))
    split = "test_ood" if fam.ood else "test_id"
    # Balance variants: each categorical signature gets at most `quota` tasks.
    probe = np.random.default_rng(0)
    n_variants = len({variant(fam.sample_params(probe, "test")) for _ in range(3000)})
    quota = -(-fam.n_test // n_variants)
    tasks, seen, rejects, per_variant = [], set(), Counter(), Counter()
    tries = 0
    while len(tasks) < fam.n_test and tries < max_tries:
        tries += 1
        if tries % 300 == 0:
            quota += 1  # a variant that keeps failing verification must not block the build
        p = fam.sample_params(rng, "test")
        key = json.dumps(p, sort_keys=True)
        if key in seen or per_variant[variant(p)] >= quota:
            continue
        seen.add(key)
        ok, why, stats = verify(fam, p)
        if not ok:
            rejects[why.split(":")[0].split(" on ")[0]] += 1
            print(f"  reject {fam.name} {key}: {why}")
            continue
        per_variant[variant(p)] += 1
        phr = int(rng.choice(fam.phrasing_ids("test")))
        tasks.append({"id": f"{fam.name}-{len(tasks):02d}", "family": fam.name, "split": split,
                      "difficulty": fam.difficulty(p), "params": p, "phrasing": phr,
                      "prompt": fam.prompt(p, phr), "reference": fam.reference(p), "ref_stats": stats})
    if len(tasks) < fam.n_test:
        raise RuntimeError(f"{fam.name}: only {len(tasks)} valid tasks after {tries} tries")
    return tasks, rejects


def main():
    out = {"test_id": [], "test_ood": []}
    summary = {}
    for name, fam in FAMILIES.items():
        tasks, rejects = build_family(fam)
        out[tasks[0]["split"]].extend(tasks)
        summary[name] = {"kept": len(tasks), "rejected": dict(rejects),
                         "difficulty": dict(Counter(t["difficulty"] for t in tasks))}
        print(f"{name:12s} kept {len(tasks):2d}  rejected {sum(rejects.values()):2d} {dict(rejects)}")
    for split, tasks in out.items():
        with open(BENCH / f"tasks_{split}.jsonl", "w") as f:
            for t in tasks:
                f.write(json.dumps(t) + "\n")
    (BENCH / "build_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print({k: len(v) for k, v in out.items()})


if __name__ == "__main__":
    main()
