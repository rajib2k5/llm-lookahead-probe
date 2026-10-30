"""C9: leak check catches canonical-form ticker/date/price leaks; no false alarm on clean prompts."""
import sys
from datetime import date, timedelta
import numpy as np
from llm_lookahead_probe.data import PricePanel
from llm_lookahead_probe.masking import LeakChecker, LeakError, Masker
from llm_lookahead_probe.pit import build_observation
from llm_lookahead_probe.prompt import render_prompt

tickers = ("A", "AAPL", "BF-B", "BRK.B", "F", "X1")
dates = tuple(date(2024, 1, 1) + timedelta(days=i) for i in range(40) if (date(2024, 1, 1) + timedelta(days=i)).weekday() < 5)
rng = np.random.default_rng(0)
closes = np.round(np.exp(np.log(50) + np.cumsum(rng.normal(0, 0.01, (len(dates), len(tickers))), axis=0)), 2)
panel = PricePanel(dates=dates, tickers=tickers, closes=closes)
chk = LeakChecker(panel); mk = Masker.fit(panel, 1)
fails = []
real_price = float(closes[10, 1]); rp = f"{real_price:.2f}"
for t in tickers:
    obs = build_observation(panel, t, dates[10], 5)
    pm = render_prompt(mk.mask(obs)); reals = [b.close for b in obs.bars] + [float(closes[11, tickers.index(t)])]
    try: chk.check(pm, reals)
    except LeakError as e: fails.append(("false alarm", t, str(e)))
    base = pm
    must = [f"hold in {t} from", f"Note: {t}.", f"({t})", f"{t},", f"\n{t}\n", f"{t}'s view", f"${t} now", f"{t}:"]
    for s in must:
        try: chk.check(base + " " + s, reals); fails.append(("missed ticker", t, s))
        except LeakError: pass
obs = build_observation(panel, "AAPL", dates[10], 5); pm = render_prompt(mk.mask(obs))
reals = [b.close for b in obs.bars] + [float(closes[11, 1])]
rp = f"{reals[2]:.2f}"
for s in [dates[3].isoformat(), f"on {dates[3].isoformat()}.", f"x{dates[3].isoformat()}y", rp, f"close {rp}.", f"${rp}", f"({rp})"]:
    try: chk.check(pm + " " + s, reals); fails.append(("missed date/price", s))
    except LeakError: pass
print("canonical failures:", fails)
# Findings (not verdict): non-canonical forms
d = dates[3]; whole = f"{reals[2]:.2f}"
nc = {"lowercase ticker": "aapl", "AAPL-like": "AAPL-like", "_AAPL_": "_AAPL_", "slash date": d.strftime("%Y/%m/%d"),
      "compact date": d.strftime("%Y%m%d"), "us date": d.strftime("%m/%d/%Y"), "month name": d.strftime("%b %d, %Y"),
      "price 3dp": f"{reals[2]:.3f}", "price 1dp": f"{reals[2]:.1f}" if not whole.endswith("0") else whole[:-1]}
for k, s in nc.items():
    try: chk.check(pm + " " + s, reals); print(f"escapes (non-canonical): {k} -> {s!r}")
    except LeakError: print(f"caught: {k} -> {s!r}")
sys.exit(1 if fails else 0)
