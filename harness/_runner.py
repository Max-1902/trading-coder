"""Subprocess entry point: executes untrusted strategy code.

Reads a pickled job from stdin (written by the trusted parent), runs
generate_signals on each series and on look-ahead variants, and prints a
single JSON line. Only plain lists/strings go back to the parent; it never
unpickles anything produced here.
"""

import json
import pickle
import sys
import traceback
import warnings

real_stdout = sys.stdout
sys.stdout = sys.stderr  # anything the solution prints goes to stderr
warnings.simplefilter("ignore")

job = pickle.loads(sys.stdin.buffer.read())
sys.path.insert(0, job["project_root"])

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from harness.series import perturb_after  # noqa: E402


def emit(obj):
    real_stdout.write(json.dumps(obj) + "\n")
    real_stdout.flush()


ns = {"__name__": "solution"}
try:
    exec(compile(job["code"], "<solution>", "exec"), ns)
    fn = ns["generate_signals"]
except BaseException as e:  # noqa: BLE001
    emit({"module_error": f"{type(e).__name__}: {e}"})
    sys.exit(0)


def call(df: pd.DataFrame) -> dict:
    try:
        res = fn(df.copy())
    except Exception as e:  # noqa: BLE001
        tb = traceback.extract_tb(e.__traceback__)
        line = next((f.lineno for f in reversed(tb) if f.filename == "<solution>"), None)
        return {"ok": False, "error": "exception", "detail": f"{type(e).__name__}: {e} (line {line})"}
    if not isinstance(res, pd.Series):
        return {"ok": False, "error": "not_series", "detail": type(res).__name__}
    if len(res) != len(df):
        return {"ok": False, "error": "wrong_length", "detail": f"{len(res)} != {len(df)}"}
    if not res.index.equals(df.index):
        return {"ok": False, "error": "wrong_index", "detail": "index differs from df.index"}
    try:
        vals = np.asarray(res.to_numpy(), dtype=float)
    except (TypeError, ValueError) as e:
        return {"ok": False, "error": "bad_dtype", "detail": f"{res.dtype}: {e}"}
    if np.isnan(vals).any():
        return {"ok": False, "error": "nan", "detail": f"{int(np.isnan(vals).sum())} NaN values"}
    bad = ~np.isin(vals, [-1.0, 0.0, 1.0])
    if bad.any():
        return {"ok": False, "error": "bad_values", "detail": f"e.g. {vals[bad][:3].tolist()}"}
    return {"ok": True, "pos": vals.astype(int).tolist()}


outputs = {name: call(df) for name, df in job["series"].items()}

la_cuts = {}
la = job.get("lookahead")
if la:
    rng = np.random.default_rng(la["seed"])
    for name in la["series"]:
        df = job["series"][name]
        orig = outputs[name]
        # determinism: a second run on the same data must match exactly
        outputs[f"det::{name}"] = call(df)
        n = len(df)
        cuts = set(int(k) for k in rng.integers(5, n - 3, la["n_random"]))
        if orig["ok"]:
            # bars just before / at a position change are where peeking flips a signal
            pos = np.asarray(orig["pos"])
            changes = np.flatnonzero(np.diff(pos) != 0)  # pos[t+1] != pos[t]
            cand = sorted(set(changes.tolist()) | set((changes + 1).tolist()))
            cand = [k for k in cand if 0 <= k < n - 2]
            if cand:
                pick = rng.choice(len(cand), min(la["n_transition"], len(cand)), replace=False)
                cuts |= {cand[i] for i in pick}
        la_cuts[name] = []
        for k in sorted(cuts):
            for j, mode in enumerate(la["modes"]):
                pdf = perturb_after(df, k, seed=la["seed"] * 7919 + k * 31 + j, mode=mode)
                outputs[f"la::{name}::{k}::{mode}"] = call(pdf)
                la_cuts[name].append([k, mode])

emit({"outputs": outputs, "la_cuts": la_cuts})
