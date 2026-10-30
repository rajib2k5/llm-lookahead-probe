"""C1: pnl sign convention and r = close(t+1)/close(t) - 1."""
import sys
from datetime import date
import numpy as np
from llm_lookahead_probe.stats import pnl
from llm_lookahead_probe.prompt import Action
from llm_lookahead_probe.data import PricePanel

fails = []
for r in (0.0123, -0.045, 0.0):
    exp = {Action.LONG: r, Action.SHORT: -r, Action.ABSTAIN: 0.0}
    for a, e in exp.items():
        got = pnl(a, r)
        if got != e:
            fails.append((a, r, got, e))
panel = PricePanel(dates=(date(2024, 1, 2), date(2024, 1, 3), date(2024, 1, 4)), tickers=("X",),
                   closes=np.array([[100.0], [110.0], [99.0]]))
r1 = panel.next_day_return("X", date(2024, 1, 2)); r2 = panel.next_day_return("X", date(2024, 1, 3))
print("r(t=1/2)", r1, "expected", 110 / 100 - 1, "| r(t=1/3)", r2, "expected", 99 / 110 - 1)
if abs(r1 - 0.1) > 1e-15 or abs(r2 - (99 / 110 - 1)) > 1e-15:
    fails.append(("r", r1, r2))
print("fails:", fails)
sys.exit(1 if fails else 0)
