"""Donchian / N-bar breakouts, stop-and-reverse or channel exit. Stateful. Difficulty 2."""

import numpy as np

from bench.families.common import (HEADER, ONE_CHANGE, Family, a_prior_max, a_prior_min,
                                   a_state_machine, body, cols, isnan, side_words, state_machine,
                                   split_values)

N_GRID = [10, 12, 15, 18, 20, 22, 25, 30, 35, 40, 45, 50, 55, 60]
M_GRID = [3, 4, 5, 6, 7, 8, 10, 12, 14, 15, 20, 25]


class Breakout(Family):
    name = "breakout"
    phrasings = [
        "{side_cap} {n}-bar breakout: buy when the close breaks above the prior {n}-bar {hi_word}{exit_part}.",
        "Channel breakout system. A close above the {hi_word} of the previous {n} bars opens a long{exit_part}. It is {side}.",
        "Go long on a new {n}-bar breakout of the {hi_word}{exit_part}.",
        "Donchian-style strategy with a {n}-bar entry channel: long when the close exceeds the prior channel top{exit_part}. The strategy is {side}.",
        "Enter long when price closes above its prior {n}-bar {hi_word}{exit_part}.",
    ]

    def sample_params(self, rng, split):
        p = {"n": int(rng.choice(split_values(N_GRID, split))),
             "source": str(rng.choice(["high_low", "close"])),
             "exit_mode": str(rng.choice(["reverse", "channel"])),
             "side": str(rng.choice(["long_only", "long_short"]))}
        if p["exit_mode"] == "channel":
            p["m"] = int(rng.choice([m for m in split_values(M_GRID, split) if m < p["n"]]))
        return p

    def _words(self, p):
        if p["source"] == "high_low":
            return "high", "low", "prior {k}-bar high", "prior {k}-bar low"
        return "highest close", "lowest close", "prior {k}-bar highest close", "prior {k}-bar lowest close"

    def describe(self, p):
        hi, lo, _, _ = self._words(p)
        sw = side_words(p["side"])
        if p["exit_mode"] == "reverse":
            ex = (f", and a close below the prior {p['n']}-bar {lo} flips to short" if p["side"] == "long_short"
                  else f", exited when the close breaks below the prior {p['n']}-bar {lo}")
        else:
            ex = (f", exited when the close drops below the prior {p['m']}-bar {lo}"
                  + ("; shorts mirror this" if p["side"] == "long_short" else ""))
        return {"n": p["n"], "hi_word": hi, "exit_part": ex, "side": sw["side"],
                "side_cap": sw["side"][0].upper() + sw["side"][1:]}

    def rules(self, p):
        _, _, H, L = self._words(p)
        n, ls = p["n"], p["side"] == "long_short"
        Hn, Ln = H.format(k=n), L.format(k=n)
        if p["exit_mode"] == "reverse":
            r = [f"Long (1) when close > {Hn}.",
                 (f"Short (-1) when close < {Ln}." if ls else f"Flat (0) when close < {Ln}. Never go short."),
                 "Otherwise keep the previous bar's position (a long can flip directly to short and back)."
                 if ls else "Otherwise keep the previous bar's position."]
        else:
            Hm, Lm = H.format(k=p["m"]), L.format(k=p["m"])
            r = [f"When flat: go long when close > {Hn}." + (f" Go short when close < {Ln}." if ls else ""),
                 f"When long: exit to flat when close < {Lm}."]
            r.append(f"When short: exit to flat when close > {Hm}." if ls else "Never go short.")
            r += ["Otherwise keep the current position.", ONE_CHANGE]
        r.append(f"Flat during warm-up (bars 0 to {n - 1}, while the {n}-bar channel is undefined); the strategy starts flat.")
        return r

    def glossary(self, p):
        return ["PRIOR_HIGH" if p["source"] == "high_low" else "PRIOR_CLOSE_HIGH"]

    def _src(self, p):
        return ('df["high"]', 'df["low"]') if p["source"] == "high_low" else ("close", "close")

    def reference(self, p):
        hs, lsrc = self._src(p)
        n, ls = p["n"], p["side"] == "long_short"
        lines = ['close = df["close"]',
                 f"upper = {hs}.rolling({n}).max().shift(1).to_numpy()",
                 f"lower = {lsrc}.rolling({n}).min().shift(1).to_numpy()"]
        if p["exit_mode"] == "reverse":
            lines += ["c = close.to_numpy()", "",
                      "pos = np.zeros(len(df), dtype=int)", "state = 0", "for t in range(len(df)):",
                      "    if np.isnan(upper[t]):", "        state = 0",
                      "    elif c[t] > upper[t]:", "        state = 1",
                      "    elif c[t] < lower[t]:", f"        state = {-1 if ls else 0}",
                      "    pos[t] = state", "return pd.Series(pos, index=df.index)"]
        else:
            m = p["m"]
            lines += [f"exit_upper = {hs}.rolling({m}).max().shift(1).to_numpy()",
                      f"exit_lower = {lsrc}.rolling({m}).min().shift(1).to_numpy()",
                      "c = close.to_numpy()", ""]
            lines += state_machine("np.isnan(upper[t])", "c[t] > upper[t]", "c[t] < exit_lower[t]",
                                   "c[t] < lower[t]" if ls else None, "c[t] > exit_upper[t]" if ls else None)
        return HEADER + body(lines)

    def alt(self, df, p):
        _, h, l, c = cols(df)
        hs, lsrc = (h, l) if p["source"] == "high_low" else (c, c)
        n, ls = p["n"], p["side"] == "long_short"
        up, lo = a_prior_max(hs, n), a_prior_min(lsrc, n)
        if p["exit_mode"] == "reverse":
            pos, st = np.zeros(len(c), dtype=int), 0
            for t in range(len(c)):
                if isnan(up[t]):
                    st = 0
                elif c[t] > up[t]:
                    st = 1
                elif c[t] < lo[t]:
                    st = -1 if ls else 0
                pos[t] = st
            return pos
        eu, el = a_prior_max(hs, p["m"]), a_prior_min(lsrc, p["m"])
        return a_state_machine(len(c), lambda t: not isnan(up[t]),
                               lambda t: c[t] > up[t], lambda t: c[t] < el[t],
                               (lambda t: c[t] < lo[t]) if ls else None,
                               (lambda t: c[t] > eu[t]) if ls else None)

    def difficulty(self, p):
        return 2
