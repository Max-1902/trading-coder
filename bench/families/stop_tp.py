"""Entry signal + fixed stop-loss / take-profit from the entry price. Stateful. Difficulty 3."""

import numpy as np

from bench.families.common import (HEADER, ONE_CHANGE, Family, a_prior_max, a_prior_min, a_sma, body,
                                   cols, isnan, pct, side_words, split_values)

N_GRID = [10, 12, 15, 18, 20, 25, 30, 35, 40, 45, 50]
SL_GRID = [0.02, 0.025, 0.03, 0.035, 0.04, 0.05, 0.06, 0.07, 0.08]
TP_GRID = [0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.12, 0.15]


class StopTP(Family):
    name = "stop_tp"
    phrasings = [
        "{side_cap} strategy: enter when {entry_text}, then exit at a {sl} stop-loss or a {tp} take-profit measured from the entry close{short_part}.",
        "Open a long when {entry_text}. Protect it with a stop {sl} below the entry close and take profit {tp} above it{short_part}.",
        "Signal: {entry_text}. Risk management: stop-loss {sl}, take-profit {tp}, both relative to the entry price. The strategy is {side}.",
        "Trade the entry '{entry_text}' with a fixed bracket: get out after a {sl} loss or a {tp} gain from the entry close{short_part}.",
        "Buy when {entry_text}; hold until the close is down {sl} or up {tp} versus the entry close{short_part}.",
    ]

    def sample_params(self, rng, split):
        p = {"entry": str(rng.choice(["sma_cross", "breakout"])), "n": int(rng.choice(split_values(N_GRID, split))),
             "sl": float(rng.choice(split_values(SL_GRID, split))), "tp": float(rng.choice(split_values(TP_GRID, split))),
             "side": str(rng.choice(["long_only", "long_short"]))}
        return p

    def _entries(self, p):
        n = p["n"]
        if p["entry"] == "sma_cross":
            return (f"close crosses above SMA({n})", f"close crosses below SMA({n})")
        return (f"close > prior {n}-bar highest close", f"close < prior {n}-bar lowest close")

    def describe(self, p):
        sw = side_words(p["side"])
        le, _ = self._entries(p)
        short = "; shorts mirror this" if p["side"] == "long_short" else ""
        return {"entry_text": le, "sl": pct(p["sl"]), "tp": pct(p["tp"]), "short_part": short,
                "side": sw["side"], "side_cap": sw["side"][0].upper() + sw["side"][1:]}

    def rules(self, p):
        le, se = self._entries(p)
        sl, tp, ls = repr(p["sl"]), repr(p["tp"]), p["side"] == "long_short"
        r = [f"When flat: go long when {le}." + (f" Go short when {se}." if ls else ""),
             "The entry price is the close of the entry bar.",
             f"When long: exit to flat when close <= entry_price * (1 - {sl}) (stop-loss) "
             f"or close >= entry_price * (1 + {tp}) (take-profit)."]
        r.append(f"When short: exit to flat when close >= entry_price * (1 + {sl}) (stop-loss) "
                 f"or close <= entry_price * (1 - {tp}) (take-profit)." if ls else "Never go short.")
        r += ["Entry signals are ignored while a position is open.", ONE_CHANGE,
              "No entries while the indicators needed for the entry signal are undefined (warm-up); the strategy starts flat."]
        return r

    def glossary(self, p):
        return ["SMA", "CROSS"] if p["entry"] == "sma_cross" else ["PRIOR_CLOSE_HIGH"]

    def reference(self, p):
        n, sl, tp, ls = p["n"], repr(p["sl"]), repr(p["tp"]), p["side"] == "long_short"
        lines = ['close = df["close"]', "c = close.to_numpy()"]
        if p["entry"] == "sma_cross":
            lines += [f"sma = close.rolling({n}).mean().to_numpy()",
                      "above = c > sma", "below = c < sma",
                      "defined = ~np.isnan(sma)",
                      "cross_up = np.zeros(len(c), dtype=bool)",
                      "cross_up[1:] = defined[1:] & defined[:-1] & above[1:] & ~above[:-1]"]
            if ls:
                lines += ["cross_dn = np.zeros(len(c), dtype=bool)",
                          "cross_dn[1:] = defined[1:] & defined[:-1] & below[1:] & ~below[:-1]"]
            le, se = "cross_up[t]", "cross_dn[t]"
        else:
            lines += [f"prior_high = close.rolling({n}).max().shift(1).to_numpy()",
                      f"prior_low = close.rolling({n}).min().shift(1).to_numpy()"]
            le, se = "c[t] > prior_high[t]", "c[t] < prior_low[t]"
        lines += ["", "pos = np.zeros(len(c), dtype=int)", "state = 0", "entry = 0.0",
                  "for t in range(len(c)):",
                  "    if state == 0:",
                  f"        if {le}:", "            state, entry = 1, c[t]"]
        if ls:
            lines += [f"        elif {se}:", "            state, entry = -1, c[t]"]
        lines += ["    elif state == 1:",
                  f"        if c[t] <= entry * (1 - {sl}) or c[t] >= entry * (1 + {tp}):", "            state = 0"]
        if ls:
            lines += ["    elif state == -1:",
                      f"        if c[t] >= entry * (1 + {sl}) or c[t] <= entry * (1 - {tp}):", "            state = 0"]
        lines += ["    pos[t] = state", "return pd.Series(pos, index=df.index)"]
        return HEADER + body(lines)

    def alt(self, df, p):
        _, _, _, c = cols(df)
        n, sl, tp, ls = p["n"], p["sl"], p["tp"], p["side"] == "long_short"
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
        out, st, ep = np.zeros(len(c), dtype=int), 0, None
        for t in range(len(c)):
            if st == 0:
                if le(t):
                    st, ep = 1, c[t]
                elif ls and se(t):
                    st, ep = -1, c[t]
            else:
                ret = (c[t] - ep) / ep * st  # signed return of the open position
                if ret <= -sl or ret >= tp:
                    st = 0
            out[t] = st
        return out

    def difficulty(self, p):
        return 3
