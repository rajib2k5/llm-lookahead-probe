"""C5: pre-post CI = independent-strata bootstrap of the difference; width vs reference and null coverage."""
import sys
from datetime import date, timedelta
import numpy as np
from llm_lookahead_probe.stats import EpisodeResult, summarise
from llm_lookahead_probe.prompt import Action

CUT = date(2025, 1, 6)
def make(rng, g_pre=30, g_post=30, n=8, mu_pre=0.0, mu_post=0.0, sd_pre=0.01, sd_post=0.02):
    recs = []
    for part, g, mu, sd, start in (("pre", g_pre, mu_pre, sd_pre, CUT - timedelta(days=7 * g_pre)),
                                   ("post", g_post, mu_post, sd_post, CUT)):
        for k in range(g):
            d = start + timedelta(days=7 * k)
            m = rng.normal(mu, sd)
            for j in range(n):
                r = m + rng.normal(0, sd)
                recs.append(EpisodeResult(f"T{j}", d, float(r), Action.LONG, True, Action.ABSTAIN, True))
    return recs  # diff per episode = r
fails = []
rng = np.random.default_rng(3)
recs = make(rng)
s = summarise(recs, CUT, 4000, 0.95, 1)
w_code = s["diff_pre_minus_post"]["high"] - s["diff_pre_minus_post"]["low"]
# reference stratified bootstrap (explicit)
def groups(rs):
    by = {}
    for e in rs: by.setdefault(e.date, []).append(e.r)
    return [np.array(v) for v in by.values()]
gp = groups([e for e in recs if e.date < CUT]); gq = groups([e for e in recs if e.date >= CUT])
r2 = np.random.default_rng(9); out = np.empty(20000)
for b in range(20000):
    a = np.concatenate([gp[i] for i in r2.integers(0, len(gp), len(gp))]).mean()
    c = np.concatenate([gq[i] for i in r2.integers(0, len(gq), len(gq))]).mean()
    out[b] = a - c
q = np.quantile(out, [0.025, 0.975]); w_ref = q[1] - q[0]
print("diff CI code", s["diff_pre_minus_post"], "\nref quantiles", q, "width ratio", w_code / w_ref)
if abs(w_code / w_ref - 1) > 0.10: fails.append("width")
# null coverage of diff CI
N = 1000; cov = 0
for rep in range(N):
    rs = make(np.random.default_rng(1000 + rep))
    d = summarise(rs, CUT, 2000, 0.95, rep)["diff_pre_minus_post"]
    cov += d["low"] <= 0 <= d["high"]
c = cov / N; se = (c * (1 - c) / N) ** 0.5
print(f"null coverage of diff CI over {N} reps (G=30+30): {c:.3f} (se {se:.3f})")
if not (0.93 <= c <= 0.97): fails.append("coverage")
print("fails:", fails); sys.exit(1 if fails else 0)
