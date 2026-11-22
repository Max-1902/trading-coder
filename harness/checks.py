"""Score one generated solution against a reference implementation.

pass = runs AND correct AND no_lookahead

- runs:         static check OK, no timeout/crash, and every hidden series
                returns a valid Series (right index, values in {-1, 0, 1}).
- correct:      positions equal the reference exactly on every hidden series.
- no_lookahead: for several cut points k, replacing all bars after k with
                different data leaves positions 0..k unchanged. Computed from
                the solution's own outputs, independently of correctness.
"""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass, field
from functools import lru_cache

import numpy as np

from harness import series as S
from harness.sandbox import run_in_sandbox, static_check

LOOKAHEAD_SERIES = ["random_walk_101", "mean_reverting_301", "regime_switch_401", "gaps_601"]
LOOKAHEAD_CFG = {"n_random": 6, "n_transition": 6, "modes": ["crash", "spike"], "seed": 12345}

# Primary category precedence (first that applies wins). Flags are kept separately.
CATEGORIES = [
    "syntax_error", "banned_import", "banned_call", "no_function",
    "timeout", "crash", "module_error",
    "exception", "not_series", "wrong_length", "wrong_index", "bad_dtype", "nan", "bad_values",
    "nondeterministic", "lookahead", "wrong_signals", "pass",
]


@dataclass
class EvalResult:
    passed: bool
    category: str
    runs: bool = False
    correct: bool = False
    no_lookahead: bool | None = None
    detail: str = ""
    mismatches: dict = field(default_factory=dict)   # series -> {n, first_idx, first_date, ref, got}
    lookahead_hits: list = field(default_factory=list)  # [series, k, mode, first_changed_bar]
    lookahead_checks: int = 0
    seconds: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)


@lru_cache(maxsize=1)
def _hidden():
    return S.hidden_series()


@lru_cache(maxsize=4096)
def reference_positions(ref_code: str) -> dict[str, np.ndarray]:
    """Reference outputs on the hidden series (cached by code text)."""
    st = static_check(ref_code)
    if not st.ok:
        raise ValueError(f"reference fails static check: {st.category} {st.detail}")
    rr = run_in_sandbox(ref_code, _hidden())
    if rr.status != "ok":
        raise ValueError(f"reference failed to run: {rr.status} {rr.detail}")
    out = {}
    for name, o in rr.outputs.items():
        if not o["ok"]:
            raise ValueError(f"reference invalid on {name}: {o['error']} {o['detail']}")
        out[name] = np.asarray(o["pos"])
    return out


def evaluate(code: str, ref_code: str, timeout: float | None = None) -> EvalResult:
    st = static_check(code)
    if not st.ok:
        return EvalResult(False, st.category, detail=st.detail)

    ref = reference_positions(ref_code)
    hidden = _hidden()
    la = {"series": LOOKAHEAD_SERIES, **LOOKAHEAD_CFG}
    kw = {"timeout": timeout} if timeout else {}
    rr = run_in_sandbox(code, hidden, lookahead=la, **kw)
    if rr.status != "ok":
        return EvalResult(False, rr.status, detail=rr.detail, seconds=rr.seconds)

    out = rr.outputs
    res = EvalResult(False, "pass", seconds=rr.seconds)

    # 1. runs
    for name in hidden:
        o = out[name]
        if not o["ok"]:
            res.category, res.detail = o["error"], f"{name}: {o['detail']}"
            return res
    res.runs = True

    # 2. correct
    for name, df in hidden.items():
        got, want = np.asarray(out[name]["pos"]), ref[name]
        diff = np.flatnonzero(got != want)
        if diff.size:
            i = int(diff[0])
            res.mismatches[name] = {"n": int(diff.size), "first_idx": i,
                                    "first_date": str(df.index[i].date()),
                                    "ref": int(want[i]), "got": int(got[i])}
    res.correct = not res.mismatches

    # 3. determinism + look-ahead (independent of correctness)
    nondet = [n for n in LOOKAHEAD_SERIES
              if not out[f"det::{n}"]["ok"] or out[f"det::{n}"]["pos"] != out[n]["pos"]]
    la_errors = []
    for name, cuts in rr.la_cuts.items():
        base = np.asarray(out[name]["pos"])
        for k, mode in cuts:
            o = out[f"la::{name}::{k}::{mode}"]
            if not o["ok"]:
                la_errors.append(f"{name}@{k}/{mode}: {o['error']} {o['detail']}")
                continue
            res.lookahead_checks += 1
            pre = np.asarray(o["pos"][: k + 1])
            changed = np.flatnonzero(pre != base[: k + 1])
            if changed.size:
                res.lookahead_hits.append([name, k, mode, int(changed[0])])

    if nondet:
        res.no_lookahead = None
        res.category, res.detail = "nondeterministic", f"different output on rerun: {nondet}"
        return res
    if res.lookahead_hits:
        res.no_lookahead = False
        h = res.lookahead_hits[0]
        res.category = "lookahead"
        res.detail = (f"{len(res.lookahead_hits)}/{res.lookahead_checks} cuts changed past positions; "
                      f"e.g. {h[0]} cut k={h[1]} ({h[2]}) changed bar {h[3]}")
        return res
    if la_errors:
        # crashes only on perturbed data: can't certify no-look-ahead
        res.no_lookahead = None
        res.category, res.detail = "exception", f"on look-ahead variant: {la_errors[0]}"
        return res
    res.no_lookahead = True

    if not res.correct:
        name, m = next(iter(res.mismatches.items()))
        res.category = "wrong_signals"
        res.detail = (f"{len(res.mismatches)}/{len(hidden)} series differ; e.g. {name}: "
                      f"{m['n']} bars, first at {m['first_idx']} (ref {m['ref']}, got {m['got']})")
        return res

    res.passed = True
    return res


def code_hash(code: str) -> str:
    return hashlib.sha256(code.encode()).hexdigest()[:12]
