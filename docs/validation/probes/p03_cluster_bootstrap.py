"""C3 + C4: whole-date resampling, episode-weighted ratio replicate, vs independent reference."""
import sys
import numpy as np
from llm_lookahead_probe.stats import cluster_bootstrap_ci

rng = np.random.default_rng(11)
G = 40
sizes = rng.integers(1, 12, size=G)
clusters = np.repeat(np.arange(G) * 3 + 1000, sizes)
# strong within-cluster correlation, and cluster mean correlated with cluster size (so the
# episode-weighted and date-mean estimands differ)
cm = rng.normal(0, 1, G) + 0.4 * sizes
values = cm[np.repeat(np.arange(G), sizes)] + rng.normal(0, 0.3, len(clusters))
B = 4000
fails = []
# C3a: invariance to within-date permutation and to episode order (same seed)
ci1 = cluster_bootstrap_ci(values, clusters, B, 0.95, 123)
perm = rng.permutation(len(values))
ci2 = cluster_bootstrap_ci(values[perm], clusters[perm], B, 0.95, 123)
print("C3a original", ci1, "\nC3a permuted", ci2)
if not (np.isclose(ci1.low, ci2.low, rtol=1e-9, atol=1e-12) and np.isclose(ci1.high, ci2.high, rtol=1e-9, atol=1e-12)):
    fails.append("C3a permutation changed CI")
# reference: explicit cluster bootstrap by index resampling, episode-weighted mean
uniq = np.unique(clusters); idx = {c: np.where(clusters == c)[0] for c in uniq}
def ref_draws(stat, n, seed):
    r = np.random.default_rng(seed); out = np.empty(n)
    for b in range(n):
        pick = r.integers(0, len(uniq), len(uniq))
        out[b] = stat([values[idx[uniq[k]]] for k in pick])
    return out
ew = ref_draws(lambda gs: np.concatenate(gs).mean(), 20000, 5)
dm = ref_draws(lambda gs: np.mean([g.mean() for g in gs]), 20000, 6)
qe = np.quantile(ew, [0.025, 0.975]); qd = np.quantile(dm, [0.025, 0.975])
print("ref episode-weighted q:", qe, " ref date-mean q:", qd, " code:", (ci1.low, ci1.high))
sd = ew.std()
# MC tolerance: quantile SE with B=4000 and 20000 ~ 0.04 sd; use 0.15 sd
tol = 0.15 * sd
if abs(ci1.low - qe[0]) > tol or abs(ci1.high - qe[1]) > tol:
    fails.append("C3b/C4 code quantiles differ from episode-weighted reference")
# C4 discriminates: the date-mean reference must be clearly different here (else probe is blind)
print("date-mean vs episode-weighted ref separation (in sd):", abs(qd - qe) / sd)
if not (abs(qd - qe) / sd > 1.0).any():
    print("WARNING: data does not separate the two statistics; probe blind on C4"); fails.append("blind")
# C4 point estimate consistency + balanced case equals mean of date means
print("point estimate", ci1.estimate, "episode mean", values.mean())
if abs(ci1.estimate - values.mean()) > 1e-12: fails.append("estimate")
print("fails:", fails)
sys.exit(1 if fails else 0)
