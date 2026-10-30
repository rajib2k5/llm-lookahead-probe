from datetime import date, timedelta

import numpy as np
import pytest

from llm_lookahead_probe.prompt import Action
from llm_lookahead_probe.stats import (
    MIN_DATES_PER_PART,
    EpisodeResult,
    bootstrap_ratio_draws,
    cluster_bootstrap_ci,
    summarise,
)

D0 = date(2024, 1, 1)
CUT = D0 + timedelta(days=100)


def test_ci_estimate_is_mean_of_differences_and_is_seeded():
    rng = np.random.default_rng(0)
    v = rng.normal(0.01, 0.02, 400)
    c = np.repeat(np.arange(40), 10)
    a = cluster_bootstrap_ci(v, c, n_resamples=2000, level=0.95, seed=1)
    b = cluster_bootstrap_ci(v, c, n_resamples=2000, level=0.95, seed=1)
    assert a == b
    assert a.estimate == pytest.approx(v.mean())
    assert a.low < a.estimate < a.high


def test_fewer_than_2000_resamples_is_refused():
    with pytest.raises(ValueError, match="2000"):
        cluster_bootstrap_ci(np.zeros(10), np.arange(10), n_resamples=500, level=0.95, seed=0)


def test_cluster_ci_is_wider_than_iid_ci_when_dates_are_correlated():
    rng = np.random.default_rng(3)
    date_effect = np.repeat(rng.normal(0, 0.02, 50), 20)  # 50 dates x 20 tickers share a shock
    v = date_effect + rng.normal(0, 0.005, 1000)
    clustered = cluster_bootstrap_ci(v, np.repeat(np.arange(50), 20), 2000, 0.95, seed=0)
    iid = cluster_bootstrap_ci(v, np.arange(1000), 2000, 0.95, seed=0)
    assert (clustered.high - clustered.low) > 2.5 * (iid.high - iid.low)


def test_ratio_draws_weight_dates_by_episode_count():
    # One resample matrix row that picks only date 0 must give date 0's mean.
    sums, counts = np.array([3.0, 10.0]), np.array([3.0, 1.0])
    draws = bootstrap_ratio_draws(sums, counts, n_resamples=2000, seed=0)
    assert draws.min() >= 1.0 - 1e-12 and draws.max() <= 10.0 + 1e-12


def make_records(n_dates_pre, n_dates_post, gap_pre, gap_post, n_tickers=5, seed=0):
    """Records whose per-episode pnl difference has the requested mean in each part."""
    rng = np.random.default_rng(seed)
    out = []
    for k in range(n_dates_pre + n_dates_post):
        d = (CUT - timedelta(days=n_dates_pre - k)) if k < n_dates_pre else CUT + timedelta(days=k)
        gap = gap_pre if d < CUT else gap_post
        for j in range(n_tickers):
            r = abs(rng.normal(0.01, 0.002))
            # unmasked LONG earns +r; masked earns +r with prob q where q chosen to give mean gap
            q = 1 - gap / (2 * 0.01)
            am = Action.LONG if rng.random() < q else Action.SHORT
            out.append(EpisodeResult(f"T{j}", d, r, Action.LONG, True, am, True))
    return out


def test_verdict_memorisation_suspected():
    s = summarise(make_records(40, 40, 0.01, 0.0), CUT, 2000, 0.95, seed=0)
    assert s["verdict"] == "memorisation suspected"
    assert s["pre"]["gap"]["low"] > 0
    assert s["diff_pre_minus_post"]["low"] > 0


def test_verdict_no_evidence_when_gaps_are_zero():
    s = summarise(make_records(40, 40, 0.0, 0.0), CUT, 2000, 0.95, seed=0)
    assert s["verdict"] == "no evidence"


def test_equal_gap_before_and_after_cutoff_is_not_memorisation():
    s = summarise(make_records(40, 40, 0.01, 0.01), CUT, 2000, 0.95, seed=0)
    assert s["pre"]["gap"]["low"] > 0
    assert s["verdict"] == "no evidence"


def test_verdict_insufficient_data_when_a_part_has_too_few_dates():
    s = summarise(make_records(40, MIN_DATES_PER_PART - 1, 0.01, 0.0), CUT, 2000, 0.95, seed=0)
    assert s["verdict"] == "insufficient data"
    assert "insufficient" in s["verdict_reason"]


def test_arm_metrics_count_abstains_and_unparseable():
    recs = [
        EpisodeResult("A", D0, 0.02, Action.LONG, True, Action.SHORT, True),
        EpisodeResult("A", D0 + timedelta(1), -0.01, Action.LONG, True, Action.ABSTAIN, False),
        EpisodeResult("A", D0 + timedelta(2), 0.03, Action.ABSTAIN, True, Action.LONG, True),
    ]
    s = summarise(recs, CUT, 2000, 0.95, seed=0)
    u, m = s["arms"]["unmasked"], s["arms"]["masked"]
    assert u["mean_pnl"] == pytest.approx((0.02 - 0.01 + 0) / 3)
    assert m["mean_pnl"] == pytest.approx((-0.02 + 0 + 0.03) / 3)
    assert u["hit_rate"] == pytest.approx(1 / 2)
    assert m["hit_rate"] == pytest.approx(1 / 2)
    assert u["n_abstain"] == 1 and u["n_unparseable"] == 0
    assert m["n_abstain"] == 1 and m["n_unparseable"] == 1
    assert s["gap"]["estimate"] == pytest.approx(u["mean_pnl"] - m["mean_pnl"])


def test_summarise_resamples_by_date_not_by_episode():
    """With a strong shared shock per date, the reported CI must be the wide (clustered) one."""
    rng = np.random.default_rng(9)
    recs = []
    for k in range(60):
        d = D0 + timedelta(days=k)
        shock = rng.normal(0, 0.02)
        for j in range(20):
            r = abs(shock + rng.normal(0, 0.002)) + 1e-6  # unmasked LONG earns r
            am = Action.SHORT if shock > 0 else Action.LONG  # masked arm: date-level swings
            recs.append(EpisodeResult(f"T{j}", d, r, Action.LONG, True, am, True))
    s = summarise(recs, D0 + timedelta(days=30), 2000, 0.95, seed=0)
    diffs = np.array([e.r - (e.r if e.action_m == Action.LONG else -e.r) for e in recs])
    iid = cluster_bootstrap_ci(diffs, np.arange(len(diffs)), 2000, 0.95, seed=0)
    width = s["gap"]["high"] - s["gap"]["low"]
    assert width > 2.5 * (iid.high - iid.low)
