"""C8: masking keeps returns up to cent rounding, weekdays, bijective pseudonyms, whole-week shift beyond span."""
import sys, re
from datetime import date
import numpy as np
from llm_lookahead_probe.probe import OfflineConfig, sample_episodes, plan_episodes
from llm_lookahead_probe.data import synthetic_panel
from llm_lookahead_probe.masking import Masker
ROW = re.compile(r"^(\d{4}-\d{2}-\d{2}) (\d+\.\d{2})$", re.M)
fails = []; maxerr = 0.0; maxratio = 0.0; n = 0
for seed in range(30):
    cfg = OfflineConfig(seed=seed)
    panel = synthetic_panel(seed, n_tickers=cfg.n_tickers, n_days=cfg.n_days)
    mk = Masker.fit(panel, seed)
    ps = list(mk.pseudonyms.values())
    if len(set(ps)) != len(ps) or set(mk.pseudonyms) != set(panel.tickers): fails.append(("pseudo", seed))
    eps = sample_episodes(panel, cfg.n_tickers, cfg.n_dates, cfg.window, cfg.cutoff, seed)
    for p in plan_episodes(panel, eps, mk, cfg.window):
        ru = ROW.findall(p.prompt_u); rm = ROW.findall(p.prompt_m)
        du = [date.fromisoformat(d) for d, _ in ru]; dm = [date.fromisoformat(d) for d, _ in rm]
        cu = np.array([float(c) for _, c in ru]); cm = np.array([float(c) for _, c in rm])
        for a, b in zip(du, dm):
            k = (a - b).days
            if k % 7 or a.weekday() != b.weekday() or b >= panel.dates[0]: fails.append(("date", seed, a, b))
        r_u = cu[1:] / cu[:-1] - 1; r_m = cm[1:] / cm[:-1] - 1
        f = cm / cu
        bound = 0.005 * (cu[1:] / cu[:-1] + 1) / (f[:-1].min() * cu[:-1] - 0.005) + 1e-12
        err = np.abs(r_m - r_u)
        maxerr = max(maxerr, err.max()); maxratio = max(maxratio, (err / bound).max()); n += len(err)
        if (err > bound).any(): fails.append(("ret", seed, p.episode.ticker, p.episode.date))
print(f"returns checked {n}; max |dr| {maxerr:.3e}; max |dr|/bound {maxratio:.3f}")
print("fails (first 5):", fails[:5], "count", len(fails)); sys.exit(1 if fails else 0)
