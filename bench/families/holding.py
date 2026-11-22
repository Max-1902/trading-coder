"""Enter on a signal, hold for exactly H bars, then exit. Stateful. Difficulty 3."""

import numpy as np

from bench.families.common import (HEADER, Family, a_prior_max, a_prior_min, a_returns, a_rsi, body,
                                   cols, pct, rsi_lines, split_values)

P_GRID = [0.01, 0.0125, 0.015, 0.0175, 0.02, 0.025, 0.03, 0.035, 0.04]
N_GRID = [5, 6, 8, 10, 12, 15, 20, 25, 30]
RSI_N_GRID = [4, 5, 6, 7, 8, 9, 10, 12, 14]
LEVEL_GRID = [15, 20, 22, 25, 28, 30, 35]
H_GRID = [2, 3, 4, 5, 6, 7, 8, 10, 12, 15, 20]


class Holding(Family):
    name = "holding"
    phrasings = [
        "Time-based exit: {dir_word} when {sig_text} and hold for exactly {h} bars.",
        "When {sig_text}, open a {dir_noun} position, keep it for {h} bars in total, then go flat.",
        "{dir_cap} for a fixed {h}-bar holding period every time {sig_text}.",
        "Each time {sig_text}, take a {dir_noun} position that lasts {h} bars (the entry bar counts as the first).",
        "Signal: {sig_text}. Action: {dir_word}, hold {h} bars, exit; signals during the trade are ignored.",
    ]

    def sample_params(self, rng, split):
        p = {"signal": str(rng.choice(["move", "extreme", "rsi"])), "direction": str(rng.choice(["long", "short"])),
             "h": int(rng.choice(split_values(H_GRID, split)))}
        if p["signal"] == "move":
            p["p"] = float(rng.choice(split_values(P_GRID, split)))
        elif p["signal"] == "extreme":
            p["n"] = int(rng.choice(split_values(N_GRID, split)))
        else:
            p["rsi_n"] = int(rng.choice(split_values(RSI_N_GRID, split)))
            p["level"] = int(rng.choice(split_values(LEVEL_GRID, split)))
        return p

    def _sig(self, p):
        long = p["direction"] == "long"
        if p["signal"] == "move":
            return (f"the one-bar return is below -{repr(p['p'])}" if long
                    else f"the one-bar return is above {repr(p['p'])}"), \
                   (f"the close falls more than {pct(p['p'])} from the previous close" if long
                    else f"the close rises more than {pct(p['p'])} from the previous close")
        if p["signal"] == "extreme":
            return (f"close < prior {p['n']}-bar lowest close" if long else f"close > prior {p['n']}-bar highest close"), \
                   (f"the close makes a new {p['n']}-bar low" if long else f"the close makes a new {p['n']}-bar high")
        lvl = p["level"] if long else 100 - p["level"]
        return (f"RSI({p['rsi_n']}) < {lvl}" if long else f"RSI({p['rsi_n']}) > {lvl}"), \
               (f"RSI({p['rsi_n']}) is below {lvl}" if long else f"RSI({p['rsi_n']}) is above {lvl}")

    def describe(self, p):
        _, text = self._sig(p)
        long = p["direction"] == "long"
        return {"sig_text": text, "h": p["h"], "dir_word": "buy" if long else "sell short",
                "dir_cap": "Buy" if long else "Sell short", "dir_noun": "long" if long else "short"}

    def rules(self, p):
        rule, _ = self._sig(p)
        h, v = p["h"], 1 if p["direction"] == "long" else -1
        other = "Never go short." if v == 1 else "Never go long."
        return [f"Entry signal at bar t: {rule}.",
                f"When flat and the entry signal is true at bar t, the position is {v} on bars t to t+{h - 1} "
                f"({h} bars including the entry bar) and 0 on bar t+{h}.",
                f"Entry signals are ignored while the position is open and on the exit bar t+{h}; "
                f"the earliest new entry is bar t+{h + 1}.",
                "If the data ends before the holding period is over, keep the position until the last bar.",
                other,
                "No entries while the signal is undefined (warm-up); the strategy starts flat."]

    def glossary(self, p):
        return {"move": ["RET"], "extreme": ["PRIOR_CLOSE_HIGH"], "rsi": ["RSI"]}[p["signal"]]

    def reference(self, p):
        h, long = p["h"], p["direction"] == "long"
        v = 1 if long else -1
        lines = ['close = df["close"]']
        if p["signal"] == "move":
            lines += ["ret = close.pct_change()",
                      f"signal = (ret < -{repr(p['p'])}).to_numpy()" if long else f"signal = (ret > {repr(p['p'])}).to_numpy()"]
        elif p["signal"] == "extreme":
            n = p["n"]
            lines += [f"prior_low = close.rolling({n}).min().shift(1)", f"signal = (close < prior_low).to_numpy()"] if long \
                else [f"prior_high = close.rolling({n}).max().shift(1)", f"signal = (close > prior_high).to_numpy()"]
        else:
            lvl = p["level"] if long else 100 - p["level"]
            lines += rsi_lines(p["rsi_n"], "rsi")
            lines += [f"signal = (rsi {'<' if long else '>'} {lvl}).to_numpy()"]
        lines += ["", "pos = np.zeros(len(df), dtype=int)", "t = 0", "while t < len(df):",
                  "    if signal[t]:", f"        pos[t : t + {h}] = {v}", f"        t += {h + 1}",
                  "    else:", "        t += 1", "return pd.Series(pos, index=df.index)"]
        return HEADER + body(lines)

    def alt(self, df, p):
        _, _, _, c = cols(df)
        long, h = p["direction"] == "long", p["h"]
        if p["signal"] == "move":
            r = a_returns(c)
            sig = [(x < -p["p"]) if long else (x > p["p"]) for x in r]  # NaN compares False
        elif p["signal"] == "extreme":
            ref = a_prior_min(c, p["n"]) if long else a_prior_max(c, p["n"])
            sig = [(c[t] < ref[t]) if long else (c[t] > ref[t]) for t in range(len(c))]
        else:
            rs = a_rsi(c, p["rsi_n"])
            sig = [(x < p["level"]) if long else (x > 100 - p["level"]) for x in rs]
        out = np.zeros(len(c), dtype=int)
        left, cooldown = 0, False
        for t in range(len(c)):
            if left > 0:
                out[t] = 1 if long else -1
                left -= 1
                if left == 0:
                    cooldown = True
            elif cooldown:
                cooldown = False
            elif sig[t]:
                out[t] = 1 if long else -1
                left = h - 1
                cooldown = left == 0
        return out

    def difficulty(self, p):
        return 3
