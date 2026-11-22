"""Reference strategies and deliberately broken variants for harness tests."""

# ---------------------------------------------------------------- references
SMA_CROSS = """
import pandas as pd

def generate_signals(df):
    c = df["close"]
    fast = c.rolling(10).mean()
    slow = c.rolling(30).mean()
    pos = pd.Series(0, index=df.index)
    pos[fast > slow] = 1
    pos[fast < slow] = -1
    return pos
"""

# RSI(14), Wilder smoothing (ewm alpha=1/14, adjust=False, min_periods=14).
# Long-only with hysteresis: enter when RSI < 30, exit when RSI > 50.
RSI_HYST = """
import numpy as np
import pandas as pd

def generate_signals(df):
    d = df["close"].diff()
    gain = d.clip(lower=0)
    loss = (-d).clip(lower=0)
    ag = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    al = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rsi = 100 - 100 / (1 + ag / al)
    rsi = rsi.where(al != 0, 100.0).where(ag.notna())
    pos = np.zeros(len(df), dtype=int)
    state = 0
    for t in range(len(df)):
        r = rsi.iloc[t]
        if np.isnan(r):
            state = 0
        elif state == 0 and r < 30:
            state = 1
        elif state == 1 and r > 50:
            state = 0
        pos[t] = state
    return pd.Series(pos, index=df.index)
"""

# Long on close above the prior 20-bar high close; exit when close falls
# more than 5% below the highest close since entry (trailing stop).
TRAIL_STOP = """
import numpy as np
import pandas as pd

def generate_signals(df):
    c = df["close"].to_numpy()
    prior_high = df["close"].rolling(20).max().shift(1).to_numpy()
    pos = np.zeros(len(c), dtype=int)
    in_pos, peak = False, 0.0
    for t in range(len(c)):
        if in_pos:
            peak = max(peak, c[t])
            if c[t] < peak * 0.95:
                in_pos = False
        elif not np.isnan(prior_high[t]) and c[t] > prior_high[t]:
            in_pos, peak = True, c[t]
        pos[t] = 1 if in_pos else 0
    return pd.Series(pos, index=df.index)
"""

# Donchian 20: long on close > prior 20-bar high, short on close < prior
# 20-bar low, otherwise keep the previous position.
DONCHIAN = """
import numpy as np
import pandas as pd

def generate_signals(df):
    hi = df["high"].rolling(20).max().shift(1)
    lo = df["low"].rolling(20).min().shift(1)
    c = df["close"]
    sig = pd.Series(np.nan, index=df.index)
    sig[c > hi] = 1
    sig[c < lo] = -1
    sig[hi.isna()] = 0
    return sig.ffill().fillna(0).astype(int)
"""

REFERENCES = {"sma_cross": SMA_CROSS, "rsi_hyst": RSI_HYST,
              "trail_stop": TRAIL_STOP, "donchian": DONCHIAN}

# ------------------------------------- correct but differently written code
ALT_CORRECT = {
    "sma_cross": """
import numpy as np
import pandas as pd

def generate_signals(df):
    c = df["close"].to_numpy()
    cs = np.concatenate([[0.0], np.cumsum(c)])
    out = np.zeros(len(c), dtype=int)
    for t in range(29, len(c)):
        fast = (cs[t + 1] - cs[t + 1 - 10]) / 10
        slow = (cs[t + 1] - cs[t + 1 - 30]) / 30
        out[t] = 1 if fast > slow else (-1 if fast < slow else 0)
    return pd.Series(out, index=df.index)
""",
    "rsi_hyst": """
import math
import pandas as pd

def generate_signals(df):
    c = list(df["close"])
    n, a = 14, 1 / 14
    ag = al = None
    out, state = [], 0
    for t in range(len(c)):
        rsi = None
        if t >= 1:
            d = c[t] - c[t - 1]
            g, l = max(d, 0.0), max(-d, 0.0)
            if ag is None:
                ag, al = g, l
            else:
                ag = (1 - a) * ag + a * g
                al = (1 - a) * al + a * l
            if t >= n:
                rsi = 100.0 if al == 0 else 100 - 100 / (1 + ag / al)
        if rsi is None:
            state = 0
        elif state == 0 and rsi < 30:
            state = 1
        elif state == 1 and rsi > 50:
            state = 0
        out.append(state)
    return pd.Series(out, index=df.index)
""",
    "trail_stop_float_output": TRAIL_STOP.replace("dtype=int", "dtype=float"),
    "sma_bool_long_short": """
import pandas as pd

def generate_signals(df):
    c = df["close"]
    fast, slow = c.rolling(10).mean(), c.rolling(30).mean()
    return (fast > slow).astype(int) - (fast < slow).astype(int)
""",
}
ALT_REF = {"sma_cross": "sma_cross", "rsi_hyst": "rsi_hyst",
           "trail_stop_float_output": "trail_stop", "sma_bool_long_short": "sma_cross"}

# ------------------------------------------------ broken: (ref, code, category)
def _sma(body_fast="c.rolling(10).mean()", body_slow="c.rolling(30).mean()", tail="return pos"):
    return f"""
import numpy as np
import pandas as pd

def generate_signals(df):
    c = df["close"]
    fast = {body_fast}
    slow = {body_slow}
    pos = pd.Series(0, index=df.index)
    pos[fast > slow] = 1
    pos[fast < slow] = -1
    {tail}
"""


BROKEN = {
    # convention / logic errors -> wrong_signals
    "off_by_one_window": ("sma_cross", _sma(body_fast="c.rolling(11).mean()"), "wrong_signals"),
    "extra_lag": ("sma_cross", _sma(tail="return pos.shift(1).fillna(0)"), "wrong_signals"),
    "warmup_min_periods_1": ("sma_cross", _sma(body_slow="c.rolling(30, min_periods=1).mean()"), "wrong_signals"),
    "ewm_instead_of_sma": ("sma_cross", _sma(body_fast="c.ewm(span=10).mean()"), "wrong_signals"),
    "donchian_no_shift": ("donchian", DONCHIAN.replace(".max().shift(1)", ".max()").replace(".min().shift(1)", ".min()"), "wrong_signals"),
    "rsi_sma_smoothing": ("rsi_hyst", RSI_HYST.replace("ewm(alpha=1/14, adjust=False, min_periods=14)", "rolling(14)"), "wrong_signals"),
    "trail_stop_from_entry": ("trail_stop", TRAIL_STOP.replace("peak = max(peak, c[t])", "pass"), "wrong_signals"),
    # look-ahead
    "shift_minus_1": ("sma_cross", _sma(tail="return pos.shift(-1).fillna(0)"), "lookahead"),
    "center_true": ("sma_cross", _sma(body_fast="c.rolling(10, center=True).mean()"), "lookahead"),
    "future_price_in_indicator": ("sma_cross", _sma(body_fast="c.shift(-1).rolling(10).mean()"), "lookahead"),
    "full_sample_zscore": ("sma_cross", _sma(tail="z = (c - c.mean()) / c.std(); "
                           "return pd.Series(np.where(z > 0.5, 1, np.where(z < -0.5, -1, 0)), index=df.index)"),
                           "lookahead"),
    "last_bar_peek": ("sma_cross", _sma(tail="return pos if c.iloc[-1] > c.iloc[0] else -pos"), "lookahead"),
    "bfill_warmup": ("sma_cross", _sma(body_slow="c.rolling(30).mean().bfill()"), "lookahead"),
    # shift(-1) on the prior high makes the entry impossible (bar t is inside its own
    # max), so the output never depends on the future: behaviourally not look-ahead.
    "trail_stop_shift_minus_1_inert": ("trail_stop", TRAIL_STOP.replace(".shift(1)", ".shift(-1)"), "wrong_signals"),
    "trail_stop_exit_on_next_close": ("trail_stop", TRAIL_STOP.replace(
        "if c[t] < peak * 0.95:", "if c[min(t + 1, len(c) - 1)] < peak * 0.95:"), "lookahead"),
    # output shape / values
    "returns_ndarray": ("sma_cross", _sma(tail="return pos.to_numpy()"), "not_series"),
    "returns_dataframe": ("sma_cross", _sma(tail="return pos.to_frame()"), "not_series"),
    "drops_warmup": ("sma_cross", _sma(tail="return pos.iloc[29:]"), "wrong_length"),
    "reset_index": ("sma_cross", _sma(tail="return pos.reset_index(drop=True)"), "wrong_index"),
    "nan_in_warmup": ("sma_cross", _sma(tail="return np.sign(fast - slow)"), "nan"),
    "values_out_of_range": ("sma_cross", _sma(tail="return pos * 2"), "bad_values"),
    "string_values": ("sma_cross", _sma(tail="return pos.map({1: 'long', -1: 'short', 0: 'flat'})"), "bad_dtype"),
    # execution
    "syntax_error": ("sma_cross", "def generate_signals(df)\n    return df", "syntax_error"),
    "import_os": ("sma_cross", "import os\n" + SMA_CROSS, "banned_import"),
    "import_talib": ("sma_cross", "import talib\n" + SMA_CROSS, "banned_import"),
    "from_sklearn": ("sma_cross", "from sklearn.linear_model import LinearRegression\n" + SMA_CROSS, "banned_import"),
    "dunder_import": ("sma_cross", SMA_CROSS + "\nos = __import__('os')\n", "banned_call"),
    "open_file": ("sma_cross", SMA_CROSS + "\nopen('/tmp/x', 'w')\n", "banned_call"),
    "to_csv": ("sma_cross", _sma(tail="pos.to_csv('/tmp/x.csv'); return pos"), "banned_call"),
    "dunder_escape": ("sma_cross", _sma(tail="x = ().__class__.__bases__; return pos"), "banned_call"),
    "wrong_name": ("sma_cross", SMA_CROSS.replace("generate_signals", "signals"), "no_function"),
    "raises": ("sma_cross", _sma(tail="return pos['missing_key']"), "exception"),
    "module_level_error": ("sma_cross", SMA_CROSS + "\nx = 1 / 0\n", "module_error"),
    "infinite_loop": ("sma_cross", _sma(tail="while True: pass"), "timeout"),
    "nondeterministic": ("sma_cross", _sma(tail="return pd.Series(np.random.choice([-1, 0, 1], len(df)), index=df.index)"), "nondeterministic"),
}
