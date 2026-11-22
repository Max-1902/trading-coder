"""RSI / stochastic %K thresholds with hysteresis. Stateful. Difficulty 2."""

from bench.families.common import (HEADER, ONE_CHANGE, Family, a_rsi, a_state_machine, a_stoch, body,
                                   cols, isnan, rsi_lines, side_words, state_machine, split_values,
                                   stoch_lines)

N_GRID = [5, 6, 7, 8, 9, 10, 11, 12, 14, 15, 16, 18, 20, 21, 24, 25, 28, 30]
LOWER_GRID = [10, 12, 15, 18, 20, 22, 25, 28, 30, 32, 35]
EXIT_GRID = [40, 45, 48, 50, 52, 55, 60]


class Oscillator(Family):
    name = "oscillator"
    phrasings = [
        "Mean reversion on {o}: buy when it falls below {lower} and get out when it climbs above {exit}{short_part}.",
        "{side_cap} oscillator strategy using {o}. Enter long on an oversold reading (below {lower}), exit once it is above {exit}{short_part}.",
        "Go long when {o} drops under {lower} and stay long until {o} rises above {exit}{short_part}.",
        "Trade {o} with hysteresis: an entry below {lower} opens a long that is closed when {o} exceeds {exit}{short_part}. The strategy is {side}.",
        "When {o} signals oversold (< {lower}) buy, and hold until it recovers past {exit}{short_part}.",
    ]

    def sample_params(self, rng, split):
        lower = int(rng.choice(split_values(LOWER_GRID, split)))
        return {"osc": str(rng.choice(["RSI", "STOCH"])), "n": int(rng.choice(split_values(N_GRID, split))),
                "lower": lower, "exit": int(rng.choice(split_values(EXIT_GRID, split))),
                "side": str(rng.choice(["long_only", "long_short"]))}

    def _o(self, p):
        return f"RSI({p['n']})" if p["osc"] == "RSI" else f"%K({p['n']})"

    def describe(self, p):
        o, sw = self._o(p), side_words(p["side"])
        short = (f"; mirror it on the short side (short above {100 - p['lower']}, cover below {p['exit']})"
                 if p["side"] == "long_short" else "")
        return {"o": o, "lower": p["lower"], "exit": p["exit"], "short_part": short, "side": sw["side"],
                "side_cap": sw["side"][0].upper() + sw["side"][1:]}

    def rules(self, p):
        o, up = self._o(p), 100 - p["lower"]
        r = [f"When flat: go long when {o} < {p['lower']}."
             + (f" Go short when {o} > {up}." if p["side"] == "long_short" else "")]
        r.append(f"When long: exit to flat when {o} > {p['exit']}.")
        if p["side"] == "long_short":
            r.append(f"When short: exit to flat when {o} < {p['exit']}.")
        else:
            r.append("Never go short.")
        r += ["Otherwise keep the current position.", ONE_CHANGE,
              f"Flat while {o} is undefined (warm-up); the strategy starts flat."]
        return r

    def glossary(self, p):
        return [p["osc"]]

    def reference(self, p):
        lines = ['close = df["close"]']
        lines += rsi_lines(p["n"], "osc") if p["osc"] == "RSI" else stoch_lines(p["n"], "osc")
        lines += ["o = osc.to_numpy()", ""]
        ls = p["side"] == "long_short"
        lines += state_machine("np.isnan(o[t])", f"o[t] < {p['lower']}", f"o[t] > {p['exit']}",
                               f"o[t] > {100 - p['lower']}" if ls else None,
                               f"o[t] < {p['exit']}" if ls else None)
        return HEADER + body(lines)

    def alt(self, df, p):
        _, h, l, c = cols(df)
        o = a_rsi(c, p["n"]) if p["osc"] == "RSI" else a_stoch(h, l, c, p["n"])
        ls = p["side"] == "long_short"
        return a_state_machine(len(c), lambda t: not isnan(o[t]),
                               lambda t: o[t] < p["lower"], lambda t: o[t] > p["exit"],
                               (lambda t: o[t] > 100 - p["lower"]) if ls else None,
                               (lambda t: o[t] < p["exit"]) if ls else None)

    def difficulty(self, p):
        return 2
