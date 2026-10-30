"""C11: no future row reaches any prompt; injected future bars raise before rendering."""
import sys, re
from datetime import date, timedelta
from llm_lookahead_probe.probe import OfflineConfig, sample_episodes, plan_episodes
from llm_lookahead_probe.data import synthetic_panel
from llm_lookahead_probe.masking import Masker
from llm_lookahead_probe.pit import Observation, Bar, AsOfViolationError, build_observation
from llm_lookahead_probe.prompt import render_prompt
DATE = re.compile(r"\d{4}-\d{2}-\d{2}")
fails = []; n = 0
for seed in range(10):
    cfg = OfflineConfig(seed=seed); panel = synthetic_panel(seed, n_tickers=20, n_days=cfg.n_days)
    mk = Masker.fit(panel, seed); shift = timedelta(weeks=mk.shift_weeks)
    for p in plan_episodes(panel, sample_episodes(panel, 20, 120, 20, cfg.cutoff, seed), mk, 20):
        t = p.episode.date
        du = [date.fromisoformat(x) for x in DATE.findall(p.prompt_u)]
        dm = [date.fromisoformat(x) for x in DATE.findall(p.prompt_m)]
        n += 1
        if max(du) != t or max(dm) != t + shift: fails.append(("prompt", seed, t))
        # the next close must not appear in the unmasked prompt rows
        i = panel.date_index(t); nxt = panel.dates[i + 1].isoformat()
        if nxt in p.prompt_u: fails.append(("next date in prompt", t))
print("episodes checked", n)
panel = synthetic_panel(0); t = panel.dates[100]; tk = panel.tickers[0]
obs = build_observation(panel, tk, t, 20)
# 1. construct an Observation with a future bar
try:
    Observation(tk, t, obs.bars + (Bar(panel.dates[101], 1.0),)); fails.append("future bar accepted at construction")
except (AsOfViolationError, ValueError) as e: print("construction raises:", type(e).__name__, e)
# 1b. future bar not last (last bar still == as_of)
try:
    Observation(tk, t, (Bar(panel.dates[101], 1.0),) + obs.bars); fails.append("future bar (not last) accepted")
except AsOfViolationError as e: print("construction raises:", e)
# 2. mutate after construction -> render must raise
o2 = build_observation(panel, tk, t, 20)
object.__setattr__(o2, "bars", o2.bars + (Bar(panel.dates[101], 1.0),))
try:
    render_prompt(o2); fails.append("render accepted future bar")
except AsOfViolationError as e: print("render raises:", e)
# 3. missing as_of
for bad in (None, "", "2024-05-01"):
    try: build_observation(panel, tk, bad, 20); fails.append(("as_of accepted", bad))
    except AsOfViolationError: pass
print("fails:", fails); sys.exit(1 if fails else 0)
