import numpy as np
import pytest

from harness import series as S
from harness.backtest import sanity_stats
from harness.checks import evaluate, reference_positions
from harness.sandbox import static_check
from tests.fixtures import ALT_CORRECT, ALT_REF, BROKEN, REFERENCES


# ------------------------------------------------------------------ series
def test_series_deterministic_and_valid():
    a, b = S.hidden_series(), S.hidden_series()
    assert a.keys() == b.keys() and len(a) == 9
    for name, df in a.items():
        assert df.equals(b[name]), name
        assert list(df.columns) == ["open", "high", "low", "close", "volume"]
        assert df.index.is_monotonic_increasing and df.index.is_unique
        assert not df.isna().any().any()
        assert (df["high"] >= df[["open", "close"]].max(axis=1)).all()
        assert (df["low"] <= df[["open", "close"]].min(axis=1)).all()
        assert (df[["open", "high", "low", "close"]] > 0).all().all()
        # prices on the 1/128 tick, so sums are exact
        assert np.all((df["close"] * 128) == np.round(df["close"] * 128))


def test_flat_series_has_exact_flats_and_gaps_has_missing_days():
    h = S.hidden_series()
    flat = h["flat_stretches_501"]["close"]
    assert (flat.diff() == 0).sum() > 50
    gaps = h["gaps_601"]
    assert (gaps.index.to_series().diff().dt.days > 4).any()


@pytest.mark.parametrize("mode", ["crash", "spike", "noise"])
def test_perturb_keeps_prefix(mode):
    df = S.random_walk(1)
    k = 200
    p = S.perturb_after(df, k, seed=3, mode=mode)
    assert p.index.equals(df.index)
    assert p.iloc[: k + 1].equals(df.iloc[: k + 1])
    assert (p["close"].iloc[k + 1 :] != df["close"].iloc[k + 1 :]).all()


# ------------------------------------------------------------ static check
def test_static_allows_plain_numpy_pandas_math():
    assert static_check("import numpy as np\nimport pandas as pd\nimport math\n"
                        "from pandas import Series\ndef generate_signals(df): return df").ok


# ----------------------------------------------------- references accepted
@pytest.mark.parametrize("name", sorted(REFERENCES))
def test_reference_passes_itself(name):
    r = evaluate(REFERENCES[name], REFERENCES[name])
    assert r.passed, r.to_dict()
    assert r.lookahead_checks >= 40


@pytest.mark.parametrize("name", sorted(REFERENCES))
def test_reference_is_not_trivial(name):
    """Each reference takes every position it is supposed to on some series."""
    pos = np.concatenate(list(reference_positions(REFERENCES[name]).values()))
    assert (pos == 1).sum() > 20 and (pos == 0).sum() > 20
    if name in ("sma_cross", "donchian"):
        assert (pos == -1).sum() > 20


@pytest.mark.parametrize("name", sorted(ALT_CORRECT))
def test_differently_written_correct_code_passes(name):
    r = evaluate(ALT_CORRECT[name], REFERENCES[ALT_REF[name]])
    assert r.passed, r.to_dict()


# ------------------------------------------------------ broken code caught
@pytest.mark.parametrize("name", sorted(BROKEN))
def test_broken_is_caught(name):
    ref, code, expected = BROKEN[name]
    timeout = 5 if expected == "timeout" else None
    r = evaluate(code, REFERENCES[ref], timeout=timeout)
    assert not r.passed
    assert r.category == expected, r.to_dict()


def test_lookahead_flag_independent_of_correctness():
    """A peeking solution is flagged even where its signals happen to be right."""
    ref, code, _ = BROKEN["shift_minus_1"]
    r = evaluate(code, REFERENCES[ref])
    assert r.runs and r.no_lookahead is False
    # and a merely wrong (non-peeking) solution is not flagged
    ref, code, _ = BROKEN["off_by_one_window"]
    r = evaluate(code, REFERENCES[ref])
    assert r.no_lookahead is True and not r.correct


# ---------------------------------------------------------------- backtest
def test_sanity_stats():
    df = S.trend(201)
    pos = reference_positions(REFERENCES["sma_cross"])["trend_201"]
    s = sanity_stats(df, pos)
    assert set(s) >= {"sharpe", "max_drawdown", "turnover", "n_trades"}
    assert -1 <= s["max_drawdown"] <= 0 and s["n_trades"] > 0
