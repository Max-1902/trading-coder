"""Seeded synthetic OHLCV price series for scoring strategies.

All prices are rounded to a tick of 1/128. Multiples of 1/128 (at these
magnitudes) are exact in float64, so rolling sums and means come out
bit-identical whether they are computed with pandas rolling, numpy
cumsum, or a Python loop. Without this, two correct implementations can
disagree on exact ties (fast SMA == slow SMA in a flat stretch) purely
from floating-point summation order.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TICK = 1.0 / 128.0
N_BARS = 500
START = "2020-01-01"


def _quantize(x: np.ndarray) -> np.ndarray:
    return np.maximum(np.round(x / TICK) * TICK, TICK)


def _ohlcv(close: np.ndarray, rng: np.random.Generator, index: pd.DatetimeIndex) -> pd.DataFrame:
    """Build open/high/low/volume around a close path."""
    n = len(close)
    prev = np.concatenate([[close[0]], close[:-1]])
    open_ = prev * (1 + rng.normal(0, 0.002, n))
    hi_wick = np.abs(rng.normal(0, 0.004, n))
    lo_wick = np.abs(rng.normal(0, 0.004, n))
    high = np.maximum(open_, close) * (1 + hi_wick)
    low = np.minimum(open_, close) * (1 - lo_wick)
    open_, high, low = _quantize(open_), _quantize(high), _quantize(low)
    close = _quantize(close)
    high = np.maximum.reduce([high, open_, close])
    low = np.minimum.reduce([low, open_, close])
    volume = np.round(rng.lognormal(13, 0.4, n))
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume.astype(float)},
        index=index,
    )


def _bdays(n: int) -> pd.DatetimeIndex:
    return pd.bdate_range(START, periods=n, name="date")


def random_walk(seed: int, n: int = N_BARS) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.015, n)))
    return _ohlcv(close, rng, _bdays(n))


def trend(seed: int, n: int = N_BARS) -> pd.DataFrame:
    """Strong drift with ordinary noise; sign of the drift depends on the seed."""
    rng = np.random.default_rng(seed)
    drift = 0.004 * (1 if rng.random() < 0.5 else -1)
    close = 100 * np.exp(np.cumsum(rng.normal(drift, 0.012, n)))
    return _ohlcv(close, rng, _bdays(n))


def mean_reverting(seed: int, n: int = N_BARS) -> pd.DataFrame:
    """Ornstein-Uhlenbeck on log price around log(100)."""
    rng = np.random.default_rng(seed)
    theta, sigma, mu = 0.08, 0.02, np.log(100)
    x = np.empty(n)
    x[0] = mu
    for t in range(1, n):
        x[t] = x[t - 1] + theta * (mu - x[t - 1]) + rng.normal(0, sigma)
    return _ohlcv(np.exp(x), rng, _bdays(n))


def regime_switch(seed: int, n: int = N_BARS) -> pd.DataFrame:
    """Alternating calm-up / volatile-down / sideways regimes of random length."""
    rng = np.random.default_rng(seed)
    regimes = [(0.002, 0.008), (-0.003, 0.03), (0.0, 0.012)]
    rets = []
    while len(rets) < n:
        mu, sd = regimes[rng.integers(len(regimes))]
        rets.extend(rng.normal(mu, sd, rng.integers(40, 120)))
    close = 100 * np.exp(np.cumsum(rets[:n]))
    return _ohlcv(close, rng, _bdays(n))


def flat_stretches(seed: int, n: int = N_BARS) -> pd.DataFrame:
    """Random walk interrupted by runs of exactly constant prices (ties, zero volatility)."""
    rng = np.random.default_rng(seed)
    rets = rng.normal(0, 0.012, n)
    t = 30
    while t < n:
        length = int(rng.integers(15, 60))
        rets[t : t + length] = 0.0
        t += length + int(rng.integers(40, 100))
    close = _quantize(100 * np.exp(np.cumsum(rets)))
    df = _ohlcv(close, rng, _bdays(n))
    flat = np.concatenate([[False], rets[1:] == 0.0]) & (rets == 0.0)
    for col in ("open", "high", "low"):
        df.loc[flat, col] = df.loc[flat, "close"]
    return df


def gaps(seed: int, n: int = N_BARS) -> pd.DataFrame:
    """Overnight price gaps plus missing dates (irregular DatetimeIndex)."""
    rng = np.random.default_rng(seed)
    rets = rng.normal(0, 0.012, n)
    jumps = rng.random(n) < 0.03
    rets[jumps] += rng.choice([-1, 1], jumps.sum()) * rng.uniform(0.05, 0.15, jumps.sum())
    close = 100 * np.exp(np.cumsum(rets))
    all_days = _bdays(int(n * 1.3))
    keep = np.sort(rng.choice(len(all_days), n, replace=False))
    return _ohlcv(close, rng, all_days[keep])


GENERATORS = {
    "random_walk": random_walk,
    "trend": trend,
    "mean_reverting": mean_reverting,
    "regime_switch": regime_switch,
    "flat_stretches": flat_stretches,
    "gaps": gaps,
}

# Hidden test series: (generator, seed). Fixed; never change once Phase 3 runs.
HIDDEN_SEEDS = {
    "random_walk": [101, 102],
    "trend": [201, 202],
    "mean_reverting": [301, 302],
    "regime_switch": [401],
    "flat_stretches": [501],
    "gaps": [601],
}


def hidden_series() -> dict[str, pd.DataFrame]:
    return {
        f"{name}_{seed}": GENERATORS[name](seed)
        for name, seeds in HIDDEN_SEEDS.items()
        for seed in seeds
    }


def perturb_after(df: pd.DataFrame, k: int, seed: int, mode: str) -> pd.DataFrame:
    """Copy of df where every bar after k is replaced with different data.

    Bars 0..k are bit-identical. The replacement jumps to a very different
    level at bar k+1 and keeps going, so a position that depends on any bar
    > k almost surely changes under at least one mode:
    'crash' (level x0.2, then falling), 'spike' (level x5, then rising),
    'noise' (same level, very high volatility).
    """
    rng = np.random.default_rng(seed)
    m = len(df) - k - 1
    if m <= 0:
        return df.copy()
    level, drift, vol = {"crash": (0.2, -0.02, 0.03), "spike": (5.0, 0.02, 0.03),
                         "noise": (1.0, 0.0, 0.08)}[mode]
    tail_close = df["close"].iloc[k] * level * np.exp(np.cumsum(rng.normal(drift, vol, m)))
    tail = _ohlcv(tail_close, rng, df.index[k + 1 :])
    tail["volume"] = tail["volume"] * rng.uniform(0.2, 5.0)
    out = pd.concat([df.iloc[: k + 1], tail])
    out.index.name = df.index.name
    return out
