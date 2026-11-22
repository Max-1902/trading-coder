"""Bollinger Bands, mean-reversion or breakout. OOD family (never in training). Difficulty 2.

Note: Bollinger mean reversion is close to the z-score family (population vs
sample std, exit at the middle band), so it is a 'near' OOD family. Trailing
stops are the 'far' one.
"""

from bench.families.common import (HEADER, ONE_CHANGE, Family, a_sma, a_state_machine, a_std, body,
                                   cols, fmt, isnan, side_words, state_machine, split_values)

N_GRID = [10, 12, 14, 15, 18, 20, 22, 25, 30, 35, 40]
K_GRID = [1.5, 1.75, 2.0, 2.25, 2.5, 2.75, 3.0]


class Bollinger(Family):
    name = "bollinger"
    ood = True
    n_test = 14
    phrasings = [
        "{side_cap} Bollinger Band {mode_word} strategy with a {n}-bar window and {k} standard deviations: {mode_text}.",
        "Bollinger Bands ({n}, {k}). {mode_text_cap}. The strategy is {side}.",
        "Trade Bollinger Bands ({n} bars, width {k}) as a {mode_word} system: {mode_text}.",
        "Build {n}-bar Bollinger Bands {k} population standard deviations wide and use them for {mode_word}: {mode_text}.",
        "Bands at the {n}-bar mean plus/minus {k} standard deviations; {mode_text}. It is {side}.",
    ]
    test_phrasings = (0, 1, 2, 3, 4)  # OOD: every phrasing is test-only

    def sample_params(self, rng, split):
        return {"n": int(rng.choice(N_GRID)), "k": float(rng.choice(K_GRID)),
                "mode": str(rng.choice(["reversion", "breakout"])),
                "side": str(rng.choice(["long_only", "long_short"]))}

    def describe(self, p):
        sw = side_words(p["side"])
        ls = p["side"] == "long_short"
        if p["mode"] == "reversion":
            t = "buy below the lower band and sell at the middle band" + (", short above the upper band" if ls else "")
        else:
            t = "buy a close above the upper band and exit at the middle band" + (", short below the lower band" if ls else "")
        return {"n": p["n"], "k": fmt(p["k"]), "mode_word": "mean-reversion" if p["mode"] == "reversion" else "breakout",
                "mode_text": t, "mode_text_cap": t[0].upper() + t[1:], "side": sw["side"],
                "side_cap": sw["side"][0].upper() + sw["side"][1:]}

    def rules(self, p):
        n, k, ls = p["n"], fmt(p["k"]), p["side"] == "long_short"
        r = [f"middle = SMA({n}); upper = middle + {k} * PSTD({n}); lower = middle - {k} * PSTD({n})."]
        if p["mode"] == "reversion":
            r.append("When flat: go long when close < lower." + (" Go short when close > upper." if ls else ""))
            r.append("When long: exit to flat when close >= middle.")
            r.append("When short: exit to flat when close <= middle." if ls else "Never go short.")
        else:
            r.append("When flat: go long when close > upper." + (" Go short when close < lower." if ls else ""))
            r.append("When long: exit to flat when close <= middle.")
            r.append("When short: exit to flat when close >= middle." if ls else "Never go short.")
        r += ["Otherwise keep the current position.", ONE_CHANGE,
              "Flat during warm-up (while the bands are undefined); the strategy starts flat."]
        return r

    def glossary(self, p):
        return ["SMA", "PSTD"]

    def reference(self, p):
        n, k, ls = p["n"], fmt(p["k"]), p["side"] == "long_short"
        lines = ['close = df["close"]', f"middle = close.rolling({n}).mean()",
                 f"width = {k} * close.rolling({n}).std(ddof=0)",
                 "mid = middle.to_numpy()", "upper = (middle + width).to_numpy()",
                 "lower = (middle - width).to_numpy()", "c = close.to_numpy()", ""]
        if p["mode"] == "reversion":
            lines += state_machine("np.isnan(mid[t])", "c[t] < lower[t]", "c[t] >= mid[t]",
                                   "c[t] > upper[t]" if ls else None, "c[t] <= mid[t]" if ls else None)
        else:
            lines += state_machine("np.isnan(mid[t])", "c[t] > upper[t]", "c[t] <= mid[t]",
                                   "c[t] < lower[t]" if ls else None, "c[t] >= mid[t]" if ls else None)
        return HEADER + body(lines)

    def alt(self, df, p):
        _, _, _, c = cols(df)
        m, s = a_sma(c, p["n"]), a_std(c, p["n"], ddof=0)
        k, ls = p["k"], p["side"] == "long_short"
        up = [m[t] + k * s[t] for t in range(len(c))]
        lo = [m[t] - k * s[t] for t in range(len(c))]
        if p["mode"] == "reversion":
            le, lx, se, sx = (lambda t: c[t] < lo[t]), (lambda t: c[t] >= m[t]), (lambda t: c[t] > up[t]), (lambda t: c[t] <= m[t])
        else:
            le, lx, se, sx = (lambda t: c[t] > up[t]), (lambda t: c[t] <= m[t]), (lambda t: c[t] < lo[t]), (lambda t: c[t] >= m[t])
        return a_state_machine(len(c), lambda t: not isnan(m[t]), le, lx, se if ls else None, sx if ls else None)

    def difficulty(self, p):
        return 2
