"""C10: a leaking (identity) masker aborts in plan_episodes before any Model.complete call."""
import sys
from llm_lookahead_probe.probe import OfflineConfig, sample_episodes, plan_episodes, execute
from llm_lookahead_probe.data import synthetic_panel
from llm_lookahead_probe.masking import Masker, LeakError
cfg = OfflineConfig(seed=0, n_tickers=8, n_dates=60)
panel = synthetic_panel(0, n_tickers=8, n_days=cfg.n_days)
ident = Masker(pseudonyms={t: t for t in panel.tickers}, factors={t: 1.0 for t in panel.tickers}, shift_weeks=0)
class Counting:
    calls = 0
    def complete(self, prompt):
        Counting.calls += 1; return "LONG"
eps = sample_episodes(panel, 8, 60, cfg.window, cfg.cutoff, 0)
raised = False
try:
    plans = plan_episodes(panel, eps, ident, cfg.window)
    execute(plans, Counting())
except LeakError as e:
    raised = True; print("LeakError:", e)
# a masker leaking only the price (real factor 1) but proper ticker/date masking
good = Masker.fit(panel, 0)
priceleak = Masker(pseudonyms=good.pseudonyms, factors={t: 1.0 for t in panel.tickers}, shift_weeks=good.shift_weeks)
raised2 = False
try:
    execute(plan_episodes(panel, eps, priceleak, cfg.window), Counting())
except LeakError as e:
    raised2 = True; print("LeakError (price-only leak):", e)
print("raised:", raised, raised2, "model calls:", Counting.calls)
sys.exit(0 if raised and raised2 and Counting.calls == 0 else 1)
