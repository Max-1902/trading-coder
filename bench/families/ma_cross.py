"""Moving-average trend: close vs MA, or fast MA vs slow MA. Stateless. Difficulty 1."""

import numpy as np

from bench.families.common import (HEADER, Family, a_ema, a_sma, body, cols, isnan, ma_expr,
                                   side_words, split_values)

N_GRID = [5, 8, 10, 12, 15, 20, 25, 30, 35, 40, 45, 50, 60, 70, 80, 90, 100]
FAST_GRID = [3, 4, 5, 6, 8, 9, 10, 12, 14, 15, 18, 20, 25, 30]
SLOW_GRID = [20, 24, 25, 30, 35, 40, 45, 50, 55, 60, 70, 75, 80, 90, 100, 110, 120]


class MACross(Family):
    name = "ma_cross"
    phrasings = [
        "{side_cap} trend following: be long while {a} is above {b}{short_part}.",
        "Hold a long position whenever {a} is greater than {b}{short_part}. This is a {side} strategy.",
        "Use {a} against {b} as the trend signal: long when {a} > {b}{short_part}.",
        "Compare {a} with {b} on every bar. {a_cap} above {b} means long{short_part}. The strategy is {side}.",
        "A {side} moving-average system on {a} and {b}: long when {a} is the higher of the two{short_part}.",
    ]

    def sample_params(self, rng, split):
        p = {"kind": str(rng.choice(["price_ma", "ma_ma"])), "ma": str(rng.choice(["SMA", "EMA"])),
             "side": str(rng.choice(["long_only", "long_short"]))}
        if p["kind"] == "price_ma":
            p["n"] = int(rng.choice(split_values(N_GRID, split)))
        else:
            while True:
                f = int(rng.choice(split_values(FAST_GRID, split)))
                s = int(rng.choice(split_values(SLOW_GRID, split)))
                if s >= 2 * f:
                    break
            p["fast"], p["slow"] = f, s
        return p

    def _ab(self, p):
        if p["kind"] == "price_ma":
            return "the close", f"{p['ma']}({p['n']})", "close", f"{p['ma']}({p['n']})"
        return (f"{p['ma']}({p['fast']})", f"{p['ma']}({p['slow']})",
                f"{p['ma']}({p['fast']})", f"{p['ma']}({p['slow']})")

    def describe(self, p):
        a, b, _, _ = self._ab(p)
        sw = side_words(p["side"])
        short = ", short when it is below" if p["side"] == "long_short" else ", flat otherwise"
        return {"a": a, "b": b, "a_cap": a[0].upper() + a[1:], "side": sw["side"],
                "side_cap": sw["side"][0].upper() + sw["side"][1:], "short_part": short}

    def rules(self, p):
        _, _, A, B = self._ab(p)
        r = [f"Long (1) when {A} > {B}."]
        if p["side"] == "long_short":
            r += [f"Short (-1) when {A} < {B}.", f"Flat (0) when {A} equals {B}."]
        else:
            r += [f"Flat (0) when {A} <= {B}. Never go short."]
        need = B if p["kind"] == "price_ma" else f"both {A} and {B} are"
        r += [f"Flat during warm-up, until {need}{' is' if p['kind'] == 'price_ma' else ''} defined.",
              "Each bar is decided on its own; there is no memory of earlier positions."]
        return r

    def glossary(self, p):
        return [p["ma"]]

    def reference(self, p):
        if p["kind"] == "price_ma":
            lines = ['close = df["close"]', f"ma = {ma_expr(p['ma'], p['n'], 'close')}",
                     "pos = pd.Series(0, index=df.index)", "pos[close > ma] = 1"]
            if p["side"] == "long_short":
                lines.append("pos[close < ma] = -1")
        else:
            lines = ['close = df["close"]', f"fast = {ma_expr(p['ma'], p['fast'], 'close')}",
                     f"slow = {ma_expr(p['ma'], p['slow'], 'close')}",
                     "pos = pd.Series(0, index=df.index)", "pos[fast > slow] = 1"]
            if p["side"] == "long_short":
                lines.append("pos[fast < slow] = -1")
        return HEADER + body(lines + ["return pos"])

    def alt(self, df, p):
        _, _, _, c = cols(df)
        f = a_sma if p["ma"] == "SMA" else a_ema
        if p["kind"] == "price_ma":
            a, b = c, f(c, p["n"])
        else:
            a, b = f(c, p["fast"]), f(c, p["slow"])
        out = np.zeros(len(c), dtype=int)
        for t in range(len(c)):
            if isnan(a[t]) or isnan(b[t]):
                continue
            if a[t] > b[t]:
                out[t] = 1
            elif a[t] < b[t] and p["side"] == "long_short":
                out[t] = -1
        return out

    def difficulty(self, p):
        return 1
