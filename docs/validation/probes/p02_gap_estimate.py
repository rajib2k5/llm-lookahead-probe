"""C2: gap estimates are episode means of pnl_u - pnl_m (unbalanced dates)."""
import sys
from datetime import date, timedelta
import numpy as np
from llm_lookahead_probe.stats import EpisodeResult, summarise
from llm_lookahead_probe.prompt import Action

rng = np.random.default_rng(7)
acts = [Action.LONG, Action.SHORT, Action.ABSTAIN]
cutoff = date(2025, 1, 1)
recs = []
d0 = date(2024, 6, 3)
for k in range(50):
    d = d0 + timedelta(days=7 * k)
    n = 1 + (k * 7) % 9  # unbalanced: 1..9 episodes per date
    for j in range(n):
        recs.append(EpisodeResult(f"T{j}", d, float(rng.normal(0, 0.02)), acts[rng.integers(3)], True,
                                  acts[rng.integers(3)], True))
def p(a, r): return {Action.LONG: r, Action.SHORT: -r, Action.ABSTAIN: 0.0}[a]
def ref(rs): return float(np.mean([p(e.action_u, e.r) - p(e.action_m, e.r) for e in rs]))
def datemean(rs):
    by = {}
    for e in rs: by.setdefault(e.date, []).append(p(e.action_u, e.r) - p(e.action_m, e.r))
    return float(np.mean([np.mean(v) for v in by.values()]))
s = summarise(recs, cutoff, 2000, 0.95, 0)
pre = [e for e in recs if e.date < cutoff]; post = [e for e in recs if e.date >= cutoff]
rows = [("all", s["gap"]["estimate"], ref(recs), datemean(recs)),
        ("pre", s["pre"]["gap"]["estimate"], ref(pre), datemean(pre)),
        ("post", s["post"]["gap"]["estimate"], ref(post), datemean(post)),
        ("diff", s["diff_pre_minus_post"]["estimate"], ref(pre) - ref(post), None)]
bad = False
for name, got, exp, dm in rows:
    print(f"{name}: code={got:.12g} episode_mean={exp:.12g} date_mean={dm}")
    bad |= abs(got - exp) > 1e-12
print("episode-mean vs date-mean differ by", abs(ref(recs) - datemean(recs)))
sys.exit(1 if bad else 0)
