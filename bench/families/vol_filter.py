"""Trend signal gated by a volatility filter. Stateless. Difficulty 2."""

import numpy as np

from bench.families.common import (HEADER, Family, a_sma, a_vol, body, cols, fmt, isnan, side_words,
                                   split_values)

N_GRID = [10, 15, 20, 25, 30, 40, 50, 60]
FAST_GRID = [5, 8, 10, 12, 15, 20]
SLOW_GRID = [30, 40, 50, 60, 80, 100]
M_GRID = [10, 12, 15, 20, 25, 30, 40]
THR_GRID = [0.008, 0.009, 0.01, 0.011, 0.012, 0.013, 0.014, 0.015, 0.016, 0.018, 0.02]


class VolFilter(Family):
    name = "vol_filter"
    phrasings = [
        "{side_cap} trend strategy ({sig}) that only trades in calm markets: stand aside whenever {m}-bar return volatility is at or above {thr}.",
        "Follow the trend signal {sig}, but only while the standard deviation of the last {m} one-bar returns is below {thr}; otherwise be flat.",
        "Trend following with a volatility filter. Signal: {sig}. Filter: VOL({m}) < {thr}. The strategy is {side}.",
        "Take the {sig} trend position when recent volatility (std of {m} daily returns) is under {thr}, and hold nothing when it is not.",
        "Low-volatility trend trading: {sig}, switched off when {m}-bar volatility reaches {thr} or more. It is {side}.",
    ]

    def sample_params(self, rng, split):
        p = {"base": str(rng.choice(["price_ma", "ma_ma"])), "m": int(rng.choice(split_values(M_GRID, split))),
             "thr": float(rng.choice(split_values(THR_GRID, split))),
             "side": str(rng.choice(["long_only", "long_short"]))}
        if p["base"] == "price_ma":
            p["n"] = int(rng.choice(split_values(N_GRID, split)))
        else:
            p["fast"] = int(rng.choice(split_values(FAST_GRID, split)))
            p["slow"] = int(rng.choice(split_values(SLOW_GRID, split)))
        return p

    def _ab(self, p):
        return ("close", f"SMA({p['n']})") if p["base"] == "price_ma" else (f"SMA({p['fast']})", f"SMA({p['slow']})")

    def describe(self, p):
        a, b = self._ab(p)
        sw = side_words(p["side"])
        sig = f"long when {a} > {b}" + (", short when below" if p["side"] == "long_short" else "")
        return {"sig": sig, "m": p["m"], "thr": fmt(p["thr"]), "side": sw["side"],
                "side_cap": sw["side"][0].upper() + sw["side"][1:]}

    def rules(self, p):
        a, b = self._ab(p)
        ls = p["side"] == "long_short"
        base = f"Base signal: 1 when {a} > {b}" + (f", -1 when {a} < {b}" if ls else "") + ", otherwise 0."
        return [base,
                f"The position equals the base signal when VOL({p['m']}) < {fmt(p['thr'])}, and 0 when VOL({p['m']}) >= {fmt(p['thr'])}.",
                "Never go short." if not ls else "Both long and short positions are allowed.",
                f"Flat during warm-up, until both the moving average(s) and VOL({p['m']}) are defined.",
                "Each bar is decided on its own; there is no memory of earlier positions."]

    def glossary(self, p):
        return ["SMA", "RET", "VOL"]

    def reference(self, p):
        ls = p["side"] == "long_short"
        lines = ['close = df["close"]']
        if p["base"] == "price_ma":
            lines += [f"ma = close.rolling({p['n']}).mean()"]
            a, b = "close", "ma"
        else:
            lines += [f"fast = close.rolling({p['fast']}).mean()", f"slow = close.rolling({p['slow']}).mean()"]
            a, b = "fast", "slow"
        lines += [f"vol = close.pct_change().rolling({p['m']}).std()", f"calm = vol < {fmt(p['thr'])}",
                  "pos = pd.Series(0, index=df.index)", f"pos[calm & ({a} > {b})] = 1"]
        if ls:
            lines.append(f"pos[calm & ({a} < {b})] = -1")
        return HEADER + body(lines + ["return pos"])

    def alt(self, df, p):
        _, _, _, c = cols(df)
        if p["base"] == "price_ma":
            a, b = c, a_sma(c, p["n"])
        else:
            a, b = a_sma(c, p["fast"]), a_sma(c, p["slow"])
        v = a_vol(c, p["m"])
        out = np.zeros(len(c), dtype=int)
        for t in range(len(c)):
            if isnan(a[t]) or isnan(b[t]) or isnan(v[t]) or not v[t] < p["thr"]:
                continue
            if a[t] > b[t]:
                out[t] = 1
            elif a[t] < b[t] and p["side"] == "long_short":
                out[t] = -1
        return out

    def difficulty(self, p):
        return 2
