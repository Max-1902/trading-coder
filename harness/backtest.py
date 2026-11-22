"""Sanity statistics only. Never used as a score.

position[t] is applied from bar t+1: strategy return at t = position[t-1] * close return at t.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def sanity_stats(df: pd.DataFrame, positions, periods_per_year: int = 252) -> dict:
    pos = pd.Series(np.asarray(positions, dtype=float), index=df.index)
    ret = df["close"].pct_change().fillna(0.0)
    strat = pos.shift(1).fillna(0.0) * ret
    equity = (1 + strat).cumprod()
    sd = strat.std()
    sharpe = float(strat.mean() / sd * np.sqrt(periods_per_year)) if sd > 0 else 0.0
    max_dd = float((equity / equity.cummax() - 1).min())
    dpos = pos.diff().fillna(pos.iloc[0]).abs()
    return {
        "sharpe": round(sharpe, 3),
        "max_drawdown": round(max_dd, 4),
        "turnover": round(float(dpos.mean()), 4),
        "n_trades": int((dpos > 0).sum()),
        "exposure": round(float((pos != 0).mean()), 4),
        "total_return": round(float(equity.iloc[-1] - 1), 4),
    }
