"""Trend filter combined with an entry signal.

- 'cross': stateless; MA cross allowed only in the direction of a long-term SMA trend. Difficulty 2.
- 'pullback': stateful; buy RSI dips in an uptrend, exit on RSI recovery or trend break. Difficulty 3.
"""

import numpy as np

from bench.families.common import (HEADER, ONE_CHANGE, Family, a_ema, a_rsi, a_sma, a_state_machine,
                                   body, cols, isnan, ma_expr, rsi_lines, side_words, split_values,
                                   state_machine)

T_GRID = [50, 60, 70, 75, 80, 90, 100, 120, 125, 150]
FAST_GRID = [5, 6, 8, 10, 12, 15, 20]
SLOW_GRID = [20, 25, 30, 35, 40, 50]
RSI_N_GRID = [3, 4, 5, 6, 7, 8, 9, 10, 12, 14]
ENTRY_GRID = [20, 25, 30, 35, 40, 45]
EXIT_GRID = [50, 55, 60, 65, 70, 75]


class Combo(Family):
    name = "combo"
    phrasings = [
        "Trend-filtered strategy ({side}): {text}.",
        "Combine a long-term trend filter with an entry signal. {text_cap}.",
        "Only trade with the trend defined by SMA({t}). {text_cap}.",
        "Two-part rule set, trend filter plus signal: {text}. The strategy is {side}.",
        "Use SMA({t}) as the regime filter and {sig_name} for timing: {text}.",
    ]

    def sample_params(self, rng, split):
        p = {"variant": str(rng.choice(["cross", "pullback"])), "t": int(rng.choice(split_values(T_GRID, split))),
             "side": str(rng.choice(["long_only", "long_short"]))}
        if p["variant"] == "cross":
            p["ma"] = str(rng.choice(["SMA", "EMA"]))
            while True:
                f = int(rng.choice(split_values(FAST_GRID, split)))
                s = int(rng.choice(split_values(SLOW_GRID, split)))
                if s >= 2 * f:
                    break
            p["fast"], p["slow"] = f, s
        else:
            p["n"] = int(rng.choice(split_values(RSI_N_GRID, split)))
            p["entry"] = int(rng.choice(split_values(ENTRY_GRID, split)))
            p["exit"] = int(rng.choice(split_values(EXIT_GRID, split)))
        return p

    def describe(self, p):
        sw = side_words(p["side"])
        ls = p["side"] == "long_short"
        if p["variant"] == "cross":
            f, s = f"{p['ma']}({p['fast']})", f"{p['ma']}({p['slow']})"
            text = (f"long when the close is above SMA({p['t']}) and {f} is above {s}"
                    + (f"; short when the close is below SMA({p['t']}) and {f} is below {s}" if ls else ""))
            name = f"the {f}/{s} cross"
        else:
            r = f"RSI({p['n']})"
            text = (f"in an uptrend (close above SMA({p['t']})) buy when {r} dips below {p['entry']}, "
                    f"exit when {r} rises above {p['exit']} or the close falls below SMA({p['t']})"
                    + ("; shorts mirror this in a downtrend" if ls else ""))
            name = r
        return {"text": text, "text_cap": text[0].upper() + text[1:], "t": p["t"], "sig_name": name,
                "side": sw["side"]}

    def rules(self, p):
        T, ls = f"SMA({p['t']})", p["side"] == "long_short"
        if p["variant"] == "cross":
            f, s = f"{p['ma']}({p['fast']})", f"{p['ma']}({p['slow']})"
            r = [f"Long (1) when close > {T} and {f} > {s}."]
            r.append(f"Short (-1) when close < {T} and {f} < {s}." if ls else "Never go short.")
            r += ["Flat (0) in every other case.",
                  f"Flat during warm-up, until {T}, {f} and {s} are all defined.",
                  "Each bar is decided on its own; there is no memory of earlier positions."]
            return r
        R, lo, hi = f"RSI({p['n']})", p["entry"], p["exit"]
        r = [f"When flat: go long when close > {T} and {R} < {lo}."
             + (f" Go short when close < {T} and {R} > {100 - lo}." if ls else "")]
        r.append(f"When long: exit to flat when {R} > {hi} or close < {T}.")
        r.append(f"When short: exit to flat when {R} < {100 - hi} or close > {T}." if ls else "Never go short.")
        r += ["Otherwise keep the current position.", ONE_CHANGE,
              f"Flat while {T} or {R} is undefined (warm-up); the strategy starts flat."]
        return r

    def glossary(self, p):
        if p["variant"] == "cross":
            return ["SMA"] if p["ma"] == "SMA" else ["SMA", "EMA"]
        return ["SMA", "RSI"]

    def reference(self, p):
        ls = p["side"] == "long_short"
        lines = ['close = df["close"]', f"trend = close.rolling({p['t']}).mean()"]
        if p["variant"] == "cross":
            lines += [f"fast = {ma_expr(p['ma'], p['fast'], 'close')}", f"slow = {ma_expr(p['ma'], p['slow'], 'close')}",
                      "pos = pd.Series(0, index=df.index)", "pos[(close > trend) & (fast > slow)] = 1"]
            if ls:
                lines.append("pos[(close < trend) & (fast < slow)] = -1")
            return HEADER + body(lines + ["return pos"])
        lines += rsi_lines(p["n"], "rsi")
        lines += ["c = close.to_numpy()", "tr = trend.to_numpy()", "r = rsi.to_numpy()", ""]
        lo, hi = p["entry"], p["exit"]
        lines += state_machine("np.isnan(tr[t]) or np.isnan(r[t])",
                               f"c[t] > tr[t] and r[t] < {lo}", f"r[t] > {hi} or c[t] < tr[t]",
                               f"c[t] < tr[t] and r[t] > {100 - lo}" if ls else None,
                               f"r[t] < {100 - hi} or c[t] > tr[t]" if ls else None)
        return HEADER + body(lines)

    def alt(self, df, p):
        _, _, _, c = cols(df)
        tr, ls = a_sma(c, p["t"]), p["side"] == "long_short"
        if p["variant"] == "cross":
            f = a_sma if p["ma"] == "SMA" else a_ema
            a, b = f(c, p["fast"]), f(c, p["slow"])
            out = np.zeros(len(c), dtype=int)
            for t in range(len(c)):
                if isnan(tr[t]) or isnan(a[t]) or isnan(b[t]):
                    continue
                if c[t] > tr[t] and a[t] > b[t]:
                    out[t] = 1
                elif ls and c[t] < tr[t] and a[t] < b[t]:
                    out[t] = -1
            return out
        r, lo, hi = a_rsi(c, p["n"]), p["entry"], p["exit"]
        return a_state_machine(len(c), lambda t: not (isnan(tr[t]) or isnan(r[t])),
                               lambda t: c[t] > tr[t] and r[t] < lo, lambda t: r[t] > hi or c[t] < tr[t],
                               (lambda t: c[t] < tr[t] and r[t] > 100 - lo) if ls else None,
                               (lambda t: r[t] < 100 - hi or c[t] > tr[t]) if ls else None)

    def difficulty(self, p):
        return 2 if p["variant"] == "cross" else 3
