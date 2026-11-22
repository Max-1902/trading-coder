"""Shared prompt pieces, parameter splits and the Family base class.

Each family turns a parameter dict into:
- a prompt (contract preamble + strategy summary + exact rules + definitions)
- reference code (pandas, idiomatic; also the SFT target for training source A)
- an independent loop implementation (`alt`), run in-process, used only to
  show the reference isn't fragile to float details on the hidden series.
"""

from __future__ import annotations

import re

import numpy as np

PREAMBLE = """Write a Python function `generate_signals(df: pd.DataFrame) -> pd.Series` for the trading strategy below.

Interface:
- `df` has a DatetimeIndex sorted by time and float columns open, high, low, close, volume. There are no missing values.
- Return a pd.Series of integer target positions with the same index as `df`: 1 = long, -1 = short, 0 = flat.
- The position at bar t may use only data up to and including bar t. Never use future bars.
- On bars where an indicator the strategy needs is not yet defined (warm-up), the position must be 0.
- Use only numpy, pandas and math. No other imports, no file or network access."""

FOOTER = "Return only the code, in a single ```python block."

# Definitions are included in a prompt only when the task uses them.
GLOSSARY = {
    "SMA": "SMA(n): the mean of the last n closes, including the current bar. Undefined for the first n-1 bars.",
    "EMA": ("EMA(n): exponential moving average of close with alpha = 2/(n+1), computed recursively as "
            "ema[t] = alpha*close[t] + (1-alpha)*ema[t-1], starting from ema[0] = close[0]. "
            "Treated as undefined for the first n-1 bars."),
    "RSI": ("RSI(n): change[t] = close[t] - close[t-1] (from bar 1); gain = max(change, 0), loss = max(-change, 0). "
            "avg_gain starts at avg_gain[1] = gain[1] and then avg_gain[t] = avg_gain[t-1] + (gain[t] - avg_gain[t-1]) / n; "
            "avg_loss likewise. RSI = 100 - 100 / (1 + avg_gain / avg_loss), and RSI = 100 when avg_loss is 0. "
            "RSI is undefined for bars 0 to n-1 (first defined at bar n)."),
    "STOCH": ("Stochastic %K(n): 100 * (close - LL) / (HH - LL), where HH is the highest high and LL the lowest low "
              "of the last n bars including the current bar; %K = 50 when HH equals LL. Undefined for the first n-1 bars."),
    "STD": "STD(n): sample standard deviation (divide by n-1) of the last n closes, including the current bar. Undefined for the first n-1 bars.",
    "PSTD": "PSTD(n): population standard deviation (divide by n) of the last n closes, including the current bar. Undefined for the first n-1 bars.",
    "RET": "One-bar return: r[t] = close[t] / close[t-1] - 1. Undefined at bar 0.",
    "VOL": ("VOL(m): sample standard deviation (divide by m-1) of the last m one-bar returns, including the current bar. "
            "First defined at bar m."),
    "PRIOR_HIGH": ("Prior N-bar high: the highest `high` over the N bars before the current bar (bars t-N to t-1, "
                   "not including bar t). Undefined for bars 0 to N-1. Prior N-bar low is the same with the lowest `low`."),
    "PRIOR_CLOSE_HIGH": ("Prior N-bar highest close: the highest close over the N bars before the current bar "
                         "(bars t-N to t-1, not including bar t). Undefined for bars 0 to N-1. "
                         "Prior N-bar lowest close is the same with the lowest close."),
    "CROSS": ("x crosses above y at bar t when x[t] > y[t] and x[t-1] <= y[t-1], with all four values defined. "
              "x crosses below y when x[t] < y[t] and x[t-1] >= y[t-1]."),
}

ONE_CHANGE = ("The position changes at most once per bar: on a bar where a position is exited, "
              "no new position is opened.")


def split_values(values, split: str):
    """Disjoint train/test value sets: every third grid value is test-only."""
    values = list(values)
    test = values[1::3]
    return test if split == "test" else [v for v in values if v not in test]


def fmt(x) -> str:
    """Number formatting used in prompts and reference code (no float noise)."""
    if isinstance(x, float):
        s = f"{x:.6f}".rstrip("0").rstrip(".")
        return s if s else "0"
    return str(x)


def pct(x: float) -> str:
    return fmt(round(x * 100, 6)) + "%"


def normalize_prompt(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower()).strip()


class Family:
    name: str = ""
    ood: bool = False
    phrasings: list[str] = []       # strategy summary templates, filled with describe(p)
    test_phrasings: tuple = (3, 4)  # indices reserved for test prompts
    n_test: int = 9

    # --- to implement
    def sample_params(self, rng: np.random.Generator, split: str) -> dict: ...
    def describe(self, p: dict) -> dict: ...
    def rules(self, p: dict) -> list[str]: ...
    def glossary(self, p: dict) -> list[str]: ...
    def reference(self, p: dict) -> str: ...
    def alt(self, df, p: dict) -> np.ndarray: ...
    def difficulty(self, p: dict) -> int: ...

    # --- shared
    def numeric_keys(self, p: dict) -> list[str]:
        return [k for k, v in p.items() if isinstance(v, (int, float)) and not isinstance(v, bool)]

    def phrasing_ids(self, split: str) -> list[int]:
        ids = range(len(self.phrasings))
        return [i for i in ids if (i in self.test_phrasings) == (split == "test")]

    def prompt(self, p: dict, phrasing: int) -> str:
        d = self.describe(p)
        summary = self.phrasings[phrasing].format(**d)
        rules = "\n".join(f"- {r}" for r in self.rules(p))
        gl = "\n".join(f"- {GLOSSARY[k]}" for k in self.glossary(p))
        parts = [PREAMBLE, f"Strategy: {summary}", f"Rules:\n{rules}"]
        if gl:
            parts.append(f"Definitions:\n{gl}")
        parts.append(FOOTER)
        return "\n\n".join(parts)


def side_words(side: str) -> dict:
    return {"side": "long-only" if side == "long_only" else "long/short",
            "side_long": "long-only" if side == "long_only" else "long and short"}


# --------------------------------------------------- reference code helpers
HEADER = "import numpy as np\nimport pandas as pd\n\n\ndef generate_signals(df):\n"


def ma_expr(kind: str, n: int, src: str = "close") -> str:
    if kind == "SMA":
        return f'{src}.rolling({n}).mean()'
    return f'{src}.ewm(span={n}, adjust=False, min_periods={n}).mean()'


def rsi_lines(n: int, var: str = "rsi") -> list[str]:
    return [
        "change = close.diff()",
        f"avg_gain = change.clip(lower=0).ewm(alpha=1 / {n}, adjust=False, min_periods={n}).mean()",
        f"avg_loss = (-change).clip(lower=0).ewm(alpha=1 / {n}, adjust=False, min_periods={n}).mean()",
        f"{var} = 100 - 100 / (1 + avg_gain / avg_loss)",
        f"{var} = {var}.where(avg_loss != 0, 100.0).where(avg_gain.notna())",
    ]


def stoch_lines(n: int, var: str = "k") -> list[str]:
    return [
        f'hh = df["high"].rolling({n}).max()',
        f'll = df["low"].rolling({n}).min()',
        f"{var} = (100 * (close - ll) / (hh - ll)).where(hh != ll, 50.0).where(hh.notna())",
    ]


def body(lines: list[str]) -> str:
    return "".join(f"    {ln}\n" if ln else "\n" for ln in lines)


def state_machine(undefined: str, long_entry: str, long_exit: str,
                  short_entry: str | None, short_exit: str | None) -> list[str]:
    """Flat/long/short state machine over numpy arrays indexed by t."""
    lines = [
        "pos = np.zeros(len(df), dtype=int)",
        "state = 0",
        "for t in range(len(df)):",
        f"    if {undefined}:",
        "        state = 0",
        "    elif state == 0:",
        f"        if {long_entry}:",
        "            state = 1",
    ]
    if short_entry:
        lines += [f"        elif {short_entry}:", "            state = -1"]
    lines += ["    elif state == 1:", f"        if {long_exit}:", "            state = 0"]
    if short_exit:
        lines += ["    elif state == -1:", f"        if {short_exit}:", "            state = 0"]
    lines += ["    pos[t] = state", "return pd.Series(pos, index=df.index)"]
    return lines


# ------------------------------------------------- independent loop helpers
# Plain Python, written differently from the pandas references on purpose.
NAN = float("nan")


def a_sma(x, n):
    out = [NAN] * len(x)
    for t in range(n - 1, len(x)):
        s = 0.0
        for v in x[t - n + 1 : t + 1]:
            s += v
        out[t] = s / n
    return out


def a_ema(x, n):
    a = 2.0 / (n + 1)
    out, e = [NAN] * len(x), None
    for t, v in enumerate(x):
        e = v if e is None else a * v + (1 - a) * e
        if t >= n - 1:
            out[t] = e
    return out


def a_std(x, n, ddof=1):
    out = [NAN] * len(x)
    for t in range(n - 1, len(x)):
        w = x[t - n + 1 : t + 1]
        m = sum(w) / n
        out[t] = (sum((v - m) ** 2 for v in w) / (n - ddof)) ** 0.5
    return out


def a_rsi(x, n):
    out, ag, al = [NAN] * len(x), None, None
    for t in range(1, len(x)):
        ch = x[t] - x[t - 1]
        g, l = max(ch, 0.0), max(-ch, 0.0)
        if ag is None:
            ag, al = g, l
        else:
            ag += (g - ag) / n
            al += (l - al) / n
        if t >= n:
            out[t] = 100.0 if al == 0 else 100 - 100 / (1 + ag / al)
    return out


def a_stoch(h, l, c, n):
    out = [NAN] * len(c)
    for t in range(n - 1, len(c)):
        hh, ll = max(h[t - n + 1 : t + 1]), min(l[t - n + 1 : t + 1])
        out[t] = 50.0 if hh == ll else 100 * (c[t] - ll) / (hh - ll)
    return out


def a_prior_max(x, n):
    return [NAN] * n + [max(x[t - n : t]) for t in range(n, len(x))]


def a_prior_min(x, n):
    return [NAN] * n + [min(x[t - n : t]) for t in range(n, len(x))]


def a_returns(c):
    return [NAN] + [c[t] / c[t - 1] - 1 for t in range(1, len(c))]


def a_vol(c, m):
    r = a_returns(c)
    out = [NAN] * len(c)
    for t in range(m, len(c)):
        w = r[t - m + 1 : t + 1]
        mu = sum(w) / m
        out[t] = (sum((v - mu) ** 2 for v in w) / (m - 1)) ** 0.5
    return out


def isnan(v) -> bool:
    return v != v


def a_state_machine(n, defined, long_entry, long_exit, short_entry=None, short_exit=None):
    pos, state = [0] * n, 0
    for t in range(n):
        if not defined(t):
            state = 0
        elif state == 0:
            if long_entry(t):
                state = 1
            elif short_entry and short_entry(t):
                state = -1
        elif state == 1 and long_exit(t):
            state = 0
        elif state == -1 and short_exit(t):
            state = 0
        pos[t] = state
    return np.array(pos)


def cols(df):
    return (df["open"].tolist(), df["high"].tolist(), df["low"].tolist(), df["close"].tolist())
