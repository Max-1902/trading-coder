import json
from collections import Counter
from pathlib import Path

import numpy as np
import pytest

from bench.families import FAMILIES, ID_FAMILIES, OOD_FAMILIES
from bench.leakage import check
from gen.make_train import generate
from harness.checks import _hidden, reference_positions

BENCH = Path(__file__).parent.parent / "bench"
TEST_ID = [json.loads(l) for l in (BENCH / "tasks_test_id.jsonl").read_text().splitlines()]
TEST_OOD = [json.loads(l) for l in (BENCH / "tasks_test_ood.jsonl").read_text().splitlines()]
ALL = TEST_ID + TEST_OOD


@pytest.fixture(scope="module")
def train():
    return generate(per_family=300)


def test_benchmark_shape():
    assert len(ALL) == 100 and len(TEST_ID) == 72 and len(TEST_OOD) == 28
    assert len({t["id"] for t in ALL}) == 100
    assert len({t["prompt"] for t in ALL}) == 100
    assert {t["family"] for t in TEST_ID} == set(ID_FAMILIES)
    assert {t["family"] for t in TEST_OOD} == set(OOD_FAMILIES)
    assert Counter(t["difficulty"] for t in TEST_ID)[3] >= 15
    for t in ALL:
        assert set(t) >= {"id", "family", "split", "difficulty", "params", "prompt", "reference"}


def test_no_leakage(train):
    assert check(ALL, train) == []
    assert not {t["family"] for t in train} & set(OOD_FAMILIES)


def test_leakage_check_catches_planted_leaks(train):
    t = TEST_ID[0]
    leaked = train + [dict(t, id="planted")]
    errs = check(ALL, leaked)
    assert any("prompt appears" in e for e in errs)
    assert any("parameter combination" in e for e in errs)
    assert any("value" in e for e in errs)
    ood = dict(TEST_OOD[0], id="planted-ood")
    assert any("OOD family" in e for e in check(ALL, train + [ood]))
    # same params as a test task but reworded: still caught by the combination check
    reworded = dict(t, id="reworded", prompt="something else entirely", phrasing=0)
    assert any("parameter combination" in e for e in check(ALL, train + [reworded]))


def test_train_prompts_follow_their_family(train):
    for t in train[::97]:
        fam = FAMILIES[t["family"]]
        assert t["prompt"] == fam.prompt(t["params"], t["phrasing"])
        assert t["phrasing"] not in fam.test_phrasings


@pytest.mark.parametrize("task", ALL, ids=[t["id"] for t in ALL])
def test_reference_matches_independent_implementation(task):
    fam = FAMILIES[task["family"]]
    assert task["reference"] == fam.reference(task["params"])
    pos = reference_positions(task["reference"])
    for name, df in _hidden().items():
        assert np.array_equal(fam.alt(df, task["params"]), pos[name]), name
