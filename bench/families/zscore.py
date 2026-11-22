"""Mean reversion on the z-score of close vs its rolling mean, with entry/exit bands. Difficulty 2.

Rules are stated in band form (close vs m - k*s) instead of dividing by s:
pandas' rolling std of an exactly flat window is ~6e-7, not 0, so a rule
like "if s == 0" would separate correct pandas code from correct numpy code.
Strict entries away from the mean and non-strict exits toward it give the
same decision whether s is 0 or a tiny residue.
"""

from bench.families.common import (HEADER, ONE_CHANGE, Family, a_sma, a_state_machine, a_std, body,
                                   cols, fmt, isnan, side_words, state_machine, split_values)

N_GRID = [10, 12, 15, 18, 20, 25, 30, 35, 40, 50, 60]
ENTRY_GRID = [1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 2.75]
EXIT_GRID = [0.0, 0.25, 0.5, 0.75, 1.0]


class ZScore(Family):
    name = "zscore"
    phrasings = [
        "{side_cap} z-score mean reversion over {n} bars: buy when the close is more than {entry} standard deviations below its mean, exit when it is back within {exit} standard deviations{short_part}.",
        "Fade stretched prices. With a {n}-bar mean and standard deviation, go long below mean - {entry} sd and close the long at mean - {exit} sd or higher{short_part}.",
        "Statistical mean reversion: long when z({n}) < -{entry}, flat again once z({n}) >= -{exit}{short_part}.",
        "Use the {n}-bar z-score of the close. A reading under -{entry} opens a long that is closed when the z-score recovers to -{exit} or above{short_part}. The strategy is {side}.",
        "Buy when price sits {entry} sample standard deviations under its {n}-bar average and sell once it has reverted to within {exit} of them{short_part}.",
    ]

    def sample_params(self, rng, split):
        return {"n": int(rng.choice(split_values(N_GRID, split))),
                "entry": float(rng.choice(split_values(ENTRY_GRID, split))),
                "exit": float(rng.choice(split_values(EXIT_GRID, split))),
                "side": str(rng.choice(["long_only", "long_short"]))}

    def describe(self, p):
        sw = side_words(p["side"])
        short = "; shorts mirror this above the mean" if p["side"] == "long_short" else ""
        return {"n": p["n"], "entry": fmt(p["entry"]), "exit": fmt(p["exit"]), "short_part": short,
                "side": sw["side"], "side_cap": sw["side"][0].upper() + sw["side"][1:]}

    @staticmethod
    def _band(k, sign):
        if k == 0:
            return "m"
        return f"m {sign} {fmt(k)} * s"

    def rules(self, p):
        n, e, x, ls = p["n"], p["entry"], p["exit"], p["side"] == "long_short"
        r = [f"Let m = SMA({n}) and s = STD({n}) of the close."]
        r.append(f"When flat: go long when close < {self._band(e, '-')}."
                 + (f" Go short when close > {self._band(e, '+')}." if ls else ""))
        r.append(f"When long: exit to flat when close >= {self._band(x, '-')}.")
        r.append(f"When short: exit to flat when close <= {self._band(x, '+')}." if ls else "Never go short.")
        r += ["Use the comparisons exactly as written (no division by s), so that s = 0 needs no special case.",
              "Otherwise keep the current position.", ONE_CHANGE,
              f"Flat during warm-up (while m and s are undefined); the strategy starts flat."]
        return r

    def glossary(self, p):
        return ["SMA", "STD"]

    def reference(self, p):
        n, e, x, ls = p["n"], p["entry"], p["exit"], p["side"] == "long_short"

        def band(k, sign):
            return "m[t]" if k == 0 else f"m[t] {sign} {fmt(k)} * s[t]"

        lines = ['close = df["close"]', f"m = close.rolling({n}).mean().to_numpy()",
                 f"s = close.rolling({n}).std().to_numpy()", "c = close.to_numpy()", ""]
        lines += state_machine("np.isnan(s[t])", f"c[t] < {band(e, '-')}", f"c[t] >= {band(x, '-')}",
                               f"c[t] > {band(e, '+')}" if ls else None,
                               f"c[t] <= {band(x, '+')}" if ls else None)
        return HEADER + body(lines)

    def alt(self, df, p):
        _, _, _, c = cols(df)
        m, s = a_sma(c, p["n"]), a_std(c, p["n"])
        e, x, ls = p["entry"], p["exit"], p["side"] == "long_short"
        return a_state_machine(len(c), lambda t: not isnan(s[t]),
                               lambda t: c[t] < m[t] - e * s[t], lambda t: c[t] >= m[t] - x * s[t],
                               (lambda t: c[t] > m[t] + e * s[t]) if ls else None,
                               (lambda t: c[t] <= m[t] + x * s[t]) if ls else None)

    def difficulty(self, p):
        return 2
