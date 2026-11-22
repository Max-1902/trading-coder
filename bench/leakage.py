"""Leakage check between test tasks and training tasks. Any violation fails the build."""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

from bench.families import FAMILIES, OOD_FAMILIES
from bench.families.common import normalize_prompt

BENCH = Path(__file__).parent


def load_jsonl(path) -> list[dict]:
    return [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]


def combo_key(t: dict) -> str:
    return t["family"] + "|" + json.dumps(t["params"], sort_keys=True)


def check(test: list[dict], train: list[dict]) -> list[str]:
    errs = []
    train_prompts = {normalize_prompt(t["prompt"]) for t in train}
    train_combos = {combo_key(t) for t in train}
    train_vals = defaultdict(set)      # (family, key) -> numeric values
    train_phr = defaultdict(set)       # family -> phrasing ids
    for t in train:
        if t["family"] in OOD_FAMILIES:
            errs.append(f"OOD family {t['family']} in training data ({t['id']})")
        train_phr[t["family"]].add(t["phrasing"])
        for k in FAMILIES[t["family"]].numeric_keys(t["params"]):
            train_vals[(t["family"], k)].add(t["params"][k])
    for t in test:
        if normalize_prompt(t["prompt"]) in train_prompts:
            errs.append(f"{t['id']}: prompt appears in training data")
        if combo_key(t) in train_combos:
            errs.append(f"{t['id']}: parameter combination appears in training data")
        if t["phrasing"] in train_phr[t["family"]]:
            errs.append(f"{t['id']}: phrasing {t['phrasing']} used in training data")
        for k in FAMILIES[t["family"]].numeric_keys(t["params"]):
            if t["params"][k] in train_vals[(t["family"], k)]:
                errs.append(f"{t['id']}: value {k}={t['params'][k]} used in training data")
    return errs


def main(train_path: str):
    test = load_jsonl(BENCH / "tasks_test_id.jsonl") + load_jsonl(BENCH / "tasks_test_ood.jsonl")
    errs = check(test, load_jsonl(train_path))
    for e in errs[:50]:
        print("LEAK:", e)
    print(f"{len(errs)} leakage violations ({len(test)} test tasks checked against {train_path})")
    sys.exit(1 if errs else 0)


if __name__ == "__main__":
    main(sys.argv[1])
