from bench.families.bollinger import Bollinger
from bench.families.breakout import Breakout
from bench.families.combo import Combo
from bench.families.holding import Holding
from bench.families.ma_cross import MACross
from bench.families.oscillator import Oscillator
from bench.families.stop_tp import StopTP
from bench.families.trailing import Trailing
from bench.families.vol_filter import VolFilter
from bench.families.zscore import ZScore

FAMILIES = {f.name: f for f in (MACross(), Oscillator(), Breakout(), ZScore(), VolFilter(), StopTP(),
                                Holding(), Combo(), Bollinger(), Trailing())}
ID_FAMILIES = [n for n, f in FAMILIES.items() if not f.ood]
OOD_FAMILIES = [n for n, f in FAMILIES.items() if f.ood]
