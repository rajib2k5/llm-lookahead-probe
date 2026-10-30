"""Per-seed pipeline runner shared by p06/p07/p12/p14. Uses run_offline (the code under test) for
CIs and verdicts; re-plans the same episodes only to read prompts/records for side measurements."""
import os
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from llm_lookahead_probe.probe import OfflineConfig, run_offline, sample_episodes, plan_episodes, execute
from llm_lookahead_probe.data import synthetic_panel
from llm_lookahead_probe.masking import Masker
from llm_lookahead_probe.mocks import HonestMock
from llm_lookahead_probe.stats import pnl
from scipy_free_t import t_ppf

CONFIGS = {"SMALL": dict(n_tickers=8, n_dates=60), "DEFAULT": {}}

def _tci(x):
    g = len(x); m = x.mean(); h = t_ppf(0.975, g - 1) * x.std(ddof=1) / np.sqrt(g)
    return m - h, m + h

def one(args):
    kind, cfgname, seed, extra = args
    cfg = OfflineConfig(seed=seed, **CONFIGS[cfgname], **extra)
    rep = run_offline(kind, cfg)
    cov = lambda c: (c is not None) and (c["low"] <= 0 <= c["high"])
    out = {"seed": seed, "est": rep["gap"]["estimate"], "cov_all": cov(rep["gap"]),
           "cov_pre": cov(rep["pre"]["gap"]), "cov_post": cov(rep["post"]["gap"]),
           "cov_diff": cov(rep["diff_pre_minus_post"]), "verdict": rep["verdict"],
           "pre_low": rep["pre"]["gap"]["low"], "diff_low": rep["diff_pre_minus_post"]["low"]}
    if kind == "honest" and not extra:
        panel = synthetic_panel(cfg.seed, n_tickers=cfg.n_tickers, n_days=cfg.n_days)
        eps = sample_episodes(panel, cfg.n_tickers, cfg.n_dates, cfg.window, cfg.cutoff, cfg.seed)
        plans = plan_episodes(panel, eps, Masker.fit(panel, cfg.seed), cfg.window)
        base = HonestMock(seed=seed, flip_prob=0.0)
        out["base_mismatch"] = sum(base.base_action(p.prompt_u) != base.base_action(p.prompt_m) for p in plans)
        out["n_ep"] = len(plans)
        recs = execute(plans, HonestMock(seed=seed)).records
        d = np.array([pnl(e.action_u, e.r) - pnl(e.action_m, e.r) for e in recs])
        assert abs(d.mean() - rep["gap"]["estimate"]) < 1e-15, "replanned records differ from run_offline"
        dates = np.array([e.date.toordinal() for e in recs])
        cutoff = cfg.cutoff.toordinal()
        def dmeans(mask):
            u = np.unique(dates[mask]); return np.array([d[mask & (dates == x)].mean() for x in u])
        allm = dmeans(np.ones(len(d), bool))
        lo, hi = _tci(allm); out["tcov_all"] = lo <= 0 <= hi
        for nm, mk in (("pre", dates < cutoff), ("post", dates >= cutoff)):
            lo, hi = _tci(dmeans(mk)); out[f"tcov_{nm}"] = lo <= 0 <= hi
    return out

def run(kind, cfgname, seeds, extra=None):
    with ProcessPoolExecutor(max_workers=min(20, os.cpu_count())) as ex:
        return list(ex.map(one, [(kind, cfgname, s, extra or {}) for s in seeds], chunksize=8))

def wilson(k, n, z=1.96):
    p = k / n; den = 1 + z * z / n; c = (p + z * z / (2 * n)) / den
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return c - h, c + h
