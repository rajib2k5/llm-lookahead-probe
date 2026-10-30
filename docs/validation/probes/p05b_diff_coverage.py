"""C5.2 (split): diff-CI null coverage with more reps, vs Welch t on date means. Arg: G per part, reps."""
import sys
from datetime import date, timedelta
import numpy as np
from scipy_free_t import t_ppf
from llm_lookahead_probe.stats import EpisodeResult, summarise
from llm_lookahead_probe.prompt import Action

CUT = date(2025, 1, 6)
def make(rng, g, n=8, sd_pre=0.01, sd_post=0.02):
    recs, dm = [], {"pre": [], "post": []}
    for part, sd, start in (("pre", sd_pre, CUT - timedelta(days=7 * g)), ("post", sd_post, CUT)):
        for k in range(g):
            d = start + timedelta(days=7 * k); m = rng.normal(0, sd)
            rs = m + rng.normal(0, sd, n); dm[part].append(rs.mean())
            recs += [EpisodeResult(f"T{j}", d, float(r), Action.LONG, True, Action.ABSTAIN, True) for j, r in enumerate(rs)]
    return recs, dm
G = int(sys.argv[1]); N = int(sys.argv[2])
cp = ct = 0
for rep in range(N):
    recs, dm = make(np.random.default_rng(50000 + rep), G)
    d = summarise(recs, CUT, 2000, 0.95, rep)["diff_pre_minus_post"]
    cp += d["low"] <= 0 <= d["high"]
    a, b = np.array(dm["pre"]), np.array(dm["post"])
    va, vb = a.var(ddof=1) / G, b.var(ddof=1) / G
    df = (va + vb) ** 2 / (va**2 / (G - 1) + vb**2 / (G - 1))
    half = t_ppf(0.975, df) * (va + vb) ** 0.5
    ct += abs(a.mean() - b.mean()) <= half
for name, c in (("percentile (code)", cp / N), ("Welch t on date means", ct / N)):
    print(f"G={G}+{G} N={N} {name}: coverage {c:.4f} se {(c*(1-c)/N)**0.5:.4f}")
