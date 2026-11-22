"""Entry signal + percentage trailing stop. OOD family (never in training). Stateful. Difficulty 3."""

import numpy as np

from bench.families.common import (HEADER, ONE_CHANGE, Family, a_prior_max, a_prior_min, a_sma, body,
                                   cols, isnan, pct, side_words)

N_GRID = [10, 12, 15, 18, 20, 25, 30, 35, 40, 50]
TRAIL_GRID = [0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.1, 0.12]


class Trailing(Family):
    name = "trailing"
    ood = True
    n_test = 14
    phrasings = [
        "{side_cap} strategy: enter when {entry_text} and exit with a {p} trailing stop on the close{short_part}.",
        "Buy when {entry_text}. Trail a stop {p} below the highest close since entry and exit when the close falls under it{short_part}.",
        "Entry: {entry_text}. Exit: {p} trailing stop from the best close of the trade. The strategy is {side}.",
        "Ride trends: open a long when {entry_text} and hold it until the close drops {p} below its peak since entry{short_part}.",
        "Use a {p} close-based trailing stop to manage positions opened when {entry_text}{short_part}.",
    ]
    test_phrasings = (0, 1, 2, 3, 4)

    def sample_params(self, rng, split):
        return {"entry": str(rng.choice(["breakout", "sma_cross"])), "n": int(rng.choice(N_GRID)),
                "trail": float(rng.choice(TRAIL_GRID)), "side": str(rng.choice(["long_only", "long_short"]))}

    def _entries(self, p):
        n = p["n"]
        if p["entry"] == "sma_cross":
            return f"close crosses above SMA({n})", f"close crosses below SMA({n})"
        return f"close > prior {n}-bar highest close", f"close < prior {n}-bar lowest close"

    def describe(self, p):
        sw = side_words(p["side"])
        le, _ = self._entries(p)
        return {"entry_text": le, "p": pct(p["trail"]), "side": sw["side"],
                "side_cap": sw["side"][0].upper() + sw["side"][1:],
                "short_part": "; shorts mirror this with the lowest close" if p["side"] == "long_short" else ""}

    def rules(self, p):
        le, se = self._entries(p)
        k, ls = repr(p["trail"]), p["side"] == "long_short"
        r = [f"When flat: go long when {le}." + (f" Go short when {se}." if ls else ""),
             "While long, track peak = the highest close since entry, including the entry bar and the current bar. "
             f"Exit to flat when close < peak * (1 - {k})."]
        r.append("While short, track trough = the lowest close since entry, including the entry bar and the current bar. "
                 f"Exit to flat when close > trough * (1 + {k})." if ls else "Never go short.")
        r += ["Entry signals are ignored while a position is open.", ONE_CHANGE,
              "No entries while the indicators needed for the entry signal are undefined (warm-up); the strategy starts flat."]
        return r

    def glossary(self, p):
        return ["SMA", "CROSS"] if p["entry"] == "sma_cross" else ["PRIOR_CLOSE_HIGH"]

    def reference(self, p):
        n, k, ls = p["n"], repr(p["trail"]), p["side"] == "long_short"
        lines = ['close = df["close"]', "c = close.to_numpy()"]
        if p["entry"] == "sma_cross":
            lines += [f"sma = close.rolling({n}).mean().to_numpy()", "prev_c = np.roll(c, 1)", "prev_sma = np.roll(sma, 1)",
                      "prev_sma[0] = np.nan"]
            le = "not np.isnan(prev_sma[t]) and c[t] > sma[t] and prev_c[t] <= prev_sma[t]"
            se = "not np.isnan(prev_sma[t]) and c[t] < sma[t] and prev_c[t] >= prev_sma[t]"
        else:
            lines += [f"prior_high = close.rolling({n}).max().shift(1).to_numpy()",
                      f"prior_low = close.rolling({n}).min().shift(1).to_numpy()"]
            le, se = "c[t] > prior_high[t]", "c[t] < prior_low[t]"
        lines += ["", "pos = np.zeros(len(c), dtype=int)", "state = 0", "extreme = 0.0",
                  "for t in range(len(c)):", "    if state == 0:",
                  f"        if {le}:", "            state, extreme = 1, c[t]"]
        if ls:
            lines += [f"        elif {se}:", "            state, extreme = -1, c[t]"]
        lines += ["    elif state == 1:", "        extreme = max(extreme, c[t])",
                  f"        if c[t] < extreme * (1 - {k}):", "            state = 0"]
        if ls:
            lines += ["    elif state == -1:", "        extreme = min(extreme, c[t])",
                      f"        if c[t] > extreme * (1 + {k}):", "            state = 0"]
        lines += ["    pos[t] = state", "return pd.Series(pos, index=df.index)"]
        return HEADER + body(lines)

    def alt(self, df, p):
        _, _, _, c = cols(df)
        n, k, ls = p["n"], p["trail"], p["side"] == "long_short"
        if p["entry"] == "sma_cross":
            m = a_sma(c, n)

            def le(t):
                return t > 0 and not isnan(m[t - 1]) and c[t] > m[t] and c[t - 1] <= m[t - 1]

            def se(t):
                return t > 0 and not isnan(m[t - 1]) and c[t] < m[t] and c[t - 1] >= m[t - 1]
        else:
            hi, lo = a_prior_max(c, n), a_prior_min(c, n)

            def le(t):
                return not isnan(hi[t]) and c[t] > hi[t]

            def se(t):
                return not isnan(lo[t]) and c[t] < lo[t]
        out, st, start = np.zeros(len(c), dtype=int), 0, None
        for t in range(len(c)):
            if st == 0:
                if le(t):
                    st, start = 1, t
                elif ls and se(t):
                    st, start = -1, t
            elif st == 1:
                if c[t] < max(c[start : t + 1]) * (1 - k):
                    st = 0
            elif c[t] > min(c[start : t + 1]) * (1 + k):
                st = 0
            out[t] = st
        return out

    def difficulty(self, p):
        return 3
