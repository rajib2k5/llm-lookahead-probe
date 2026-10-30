from datetime import date, timedelta

import numpy as np
import pytest

from llm_lookahead_probe.prompt import Action
from llm_lookahead_probe.stats import (
    MAX_ABSTAIN_RATE,
    MAX_UNPARSEABLE_RATE,
    MIN_DATES_PER_PART,
    EpisodeResult,
    bootstrap_ratio_draws,
    cluster_bootstrap_ci,
    summarise,
    t_quantile,
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
    """Records whose per-episode pnl difference has the requested mean in each part.

    r has a random sign and |r| ~ 0.01, so mean r ~ 0. The unmasked arm "remembers" the sign of r
    (earns |r|); the masked arm gets the sign right with probability q. The gap therefore comes from
    knowing r, not from a direction bias. Up to a4fdd39 the fixture used r > 0 always
    with unmasked LONG, which is exactly the no-memory direction-bias x drift case of
    tests/test_direction_bias.py and is no longer (correctly) read as memorisation."""
    rng = np.random.default_rng(seed)
    out = []
    for k in range(n_dates_pre + n_dates_post):
        d = (CUT - timedelta(days=n_dates_pre - k)) if k < n_dates_pre else CUT + timedelta(days=k)
        gap = gap_pre if d < CUT else gap_post
        for j in range(n_tickers):
            r = abs(rng.normal(0.01, 0.002)) * (1 if rng.random() < 0.5 else -1)
            right = Action.LONG if r > 0 else Action.SHORT
            wrong = Action.SHORT if r > 0 else Action.LONG
            # unmasked earns +|r|; masked earns +|r| with prob q where q chosen to give mean gap
            q = 1 - gap / (2 * 0.01)
            am = right if rng.random() < q else wrong
            out.append(EpisodeResult(f"T{j}", d, r, right, True, am, True))
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


def _with_unparseable(recs, arm, every):
    """Mark every `every`-th record's answer in `arm` as unparseable (ABSTAIN), like a refusal."""
    out = []
    for k, e in enumerate(recs):
        if k % every == 0:
            if arm in ("unmasked", "both"):
                e = EpisodeResult(e.ticker, e.date, e.r, Action.ABSTAIN, False, e.action_m, e.parsed_m)
            if arm in ("masked", "both"):
                e = EpisodeResult(e.ticker, e.date, e.r, e.action_u, e.parsed_u, Action.ABSTAIN, False)
        out.append(e)
    return out


def test_all_answers_unparseable_is_an_invalid_run_not_no_evidence():
    recs = _with_unparseable(make_records(40, 40, 0.0, 0.0), "both", every=1)
    s = summarise(recs, CUT, 2000, 0.95, seed=0)
    assert s["verdict"] == "invalid run"
    assert "unparseable" in s["verdict_reason"]


def test_masked_arm_mostly_unparseable_is_an_invalid_run():
    # Masked answers fail for half the episodes: the gap measures refusals, not memory.
    recs = _with_unparseable(make_records(40, 40, 0.01, 0.0), "masked", every=2)
    s = summarise(recs, CUT, 2000, 0.95, seed=0)
    assert s["verdict"] == "invalid run"


def test_unparseable_rate_at_the_threshold_is_still_a_valid_run():
    # 1 in 10 = exactly MAX_UNPARSEABLE_RATE in every part: allowed (the rule is "exceeds").
    assert MAX_UNPARSEABLE_RATE == 0.10
    recs = _with_unparseable(make_records(40, 40, 0.0, 0.0), "both", every=10)
    s = summarise(recs, CUT, 2000, 0.95, seed=0)
    assert s["verdict"] == "no evidence"


def test_unparseable_only_in_one_part_is_an_invalid_run():
    pre = _with_unparseable(make_records(40, 0, 0.0, 0.0), "masked", every=4)  # 25% of pre
    post = [e for e in make_records(40, 40, 0.0, 0.0) if e.date >= CUT]
    s = summarise(pre + post, CUT, 2000, 0.95, seed=0)
    assert s["verdict"] == "invalid run"
    assert str(MAX_UNPARSEABLE_RATE) in s["verdict_rule"] or "10%" in s["verdict_rule"]


@pytest.mark.parametrize(
    "q, df, table",
    [
        (0.975, 29, 2.0452),
        (0.975, 59, 2.0010),
        (0.975, 10, 2.2281),
        (0.975, 1, 12.7062),
        (0.995, 5, 4.0321),
        (0.95, 20, 1.7247),
        (0.025, 10, -2.2281),
    ],
)
def test_t_quantile_matches_printed_tables(q, df, table):
    assert t_quantile(q, df) == pytest.approx(table, abs=1e-4)


def test_t_quantile_accepts_fractional_df_and_lies_between_neighbours():
    assert t_quantile(0.975, 5) > t_quantile(0.975, 5.5) > t_quantile(0.975, 6)


def _episodes(date_returns, start):
    """Two tickers per date; unmasked LONG earns r, masked ABSTAIN earns 0, so the paired
    difference of each episode is r and each date's mean difference is the mean of its two r."""
    out = []
    for k, (ra, rb) in enumerate(date_returns):
        d = start + timedelta(days=k)
        out.append(EpisodeResult("A", d, ra, Action.LONG, True, Action.ABSTAIN, True))
        out.append(EpisodeResult("B", d, rb, Action.LONG, True, Action.ABSTAIN, True))
    return out


def test_gap_ci_is_a_t_interval_on_date_means_with_g_minus_1_df():
    # Date means 0.01, 0.02, 0.03: mean 0.02, sd 0.01, G = 3, t(0.975, 2) = 4.302653.
    recs = _episodes([(0.009, 0.011), (0.019, 0.021), (0.029, 0.031)], D0)
    s = summarise(recs, D0 + timedelta(days=1000), 2000, 0.95, seed=0)
    half = 4.302653 * 0.01 / 3**0.5
    assert s["gap"]["estimate"] == pytest.approx(0.02)
    assert s["gap"]["low"] == pytest.approx(0.02 - half, abs=1e-7)
    assert s["gap"]["high"] == pytest.approx(0.02 + half, abs=1e-7)
    assert s["gap"]["df"] == 2


def test_pre_minus_post_ci_is_welch_with_satterthwaite_df():
    pre = _episodes([(0.01, 0.01), (0.02, 0.02), (0.03, 0.03)], D0)  # var 1e-4, G 3
    post_start = D0 + timedelta(days=10)
    post = _episodes([(0.0, 0.0), (0.0, 0.0), (0.02, 0.02), (0.02, 0.02)], post_start)  # var 4e-4/3
    s = summarise(pre + post, post_start, 2000, 0.95, seed=0)
    d = s["diff_pre_minus_post"]
    se = (1e-4 / 3 + (4e-4 / 3) / 4) ** 0.5
    df = (1e-4 / 3 + (4e-4 / 3) / 4) ** 2 / ((1e-4 / 3) ** 2 / 2 + ((4e-4 / 3) / 4) ** 2 / 3)
    assert df == pytest.approx(4.8)
    assert d["estimate"] == pytest.approx(0.01)
    assert d["df"] == pytest.approx(4.8)
    assert d["low"] == pytest.approx(0.01 - t_quantile(0.975, 4.8) * se, abs=1e-9)
    assert d["high"] == pytest.approx(0.01 + t_quantile(0.975, 4.8) * se, abs=1e-9)


def test_unequal_episode_counts_per_date_are_refused():
    # Date means equal the episode mean only when every date has the same number of episodes.
    recs = _episodes([(0.01, 0.01), (0.02, 0.02)], D0)[:-1]
    with pytest.raises(ValueError, match="same number of episodes"):
        summarise(recs, D0 + timedelta(days=100), 2000, 0.95, seed=0)


def test_cluster_bootstrap_is_kept_as_a_secondary_number():
    s = summarise(make_records(40, 40, 0.01, 0.0), CUT, 2000, 0.95, seed=0)
    rb = s["robustness"]["cluster_bootstrap"]
    assert set(rb) >= {"gap", "pre", "post", "diff_pre_minus_post", "n_resamples"}
    assert rb["gap"]["low"] < s["gap"]["estimate"] < rb["gap"]["high"]
    assert "t-interval" in s["interval"]["method"]


def test_bootstrap_can_be_switched_off_for_speed():
    s = summarise(make_records(30, 30, 0.0, 0.0), CUT, 0, 0.95, seed=0)
    assert s["robustness"]["cluster_bootstrap"] is None
    with pytest.raises(ValueError, match="2000"):
        summarise(make_records(30, 30, 0.0, 0.0), CUT, 500, 0.95, seed=0)


def test_each_part_reports_gap_scaled_by_its_mean_absolute_return():
    # A behaviour-constant masking effect (e.g. abstaining more when masked) scales with |r|, so
    # the gap is also reported per unit of mean |r| for each period.
    pre = _episodes([(0.01, -0.03), (0.02, 0.02)], D0)  # diffs = r; mean 0.005, mean|r| 0.02
    post = _episodes([(0.04, -0.04), (0.04, 0.04)], D0 + timedelta(days=10))  # mean 0.02, |r| 0.04
    s = summarise(pre + post, D0 + timedelta(days=10), 0, 0.95, seed=0)
    assert s["pre"]["mean_abs_r"] == pytest.approx(0.02)
    assert s["pre"]["gap_per_mean_abs_r"] == pytest.approx(0.005 / 0.02)
    assert s["post"]["mean_abs_r"] == pytest.approx(0.04)
    assert s["post"]["gap_per_mean_abs_r"] == pytest.approx(0.02 / 0.04)


def test_hit_rate_counts_a_trade_on_a_zero_return_as_a_miss():
    recs = [
        EpisodeResult("A", D0, 0.0, Action.LONG, True, Action.ABSTAIN, True),
        EpisodeResult("A", D0 + timedelta(1), 0.01, Action.LONG, True, Action.ABSTAIN, True),
    ]
    s = summarise(recs, CUT, 0, 0.95, seed=0)
    assert s["arms"]["unmasked"]["hit_rate"] == pytest.approx(1 / 2)
    assert s["arms"]["masked"]["hit_rate"] is None  # no trades


def _set_actions(recs, *, u=None, m=None, every=1):
    """Replace the parsed answer of every `every`-th record in an arm (u or m) with `action`."""
    out = []
    for k, e in enumerate(recs):
        if k % every == 0:
            au = u if u is not None else e.action_u
            am = m if m is not None else e.action_m
            e = EpisodeResult(e.ticker, e.date, e.r, au, True, am, True)
        out.append(e)
    return out


def test_a_model_that_always_abstains_is_an_invalid_run_not_no_evidence():
    # Every answer is a valid ABSTAIN: gap 0 and CI [0, 0] by construction, not a measurement.
    recs = _set_actions(make_records(40, 40, 0.0, 0.0), u=Action.ABSTAIN, m=Action.ABSTAIN)
    s = summarise(recs, CUT, 2000, 0.95, seed=0)
    assert s["verdict"] == "invalid run"
    assert "too few trades" in s["verdict_reason"]


def test_masked_abstain_unmasked_long_is_an_invalid_run():
    recs = _set_actions(make_records(40, 40, 0.0, 0.0), u=Action.LONG, m=Action.ABSTAIN)
    s = summarise(recs, CUT, 2000, 0.95, seed=0)
    assert s["verdict"] == "invalid run"
    assert "masked arm" in s["verdict_reason"]


def test_abstain_rate_at_the_threshold_is_still_a_valid_run():
    # Every 2nd record abstains in both arms = exactly MAX_ABSTAIN_RATE: allowed (rule is "exceeds").
    assert MAX_ABSTAIN_RATE == 0.50
    recs = _set_actions(make_records(40, 40, 0.0, 0.0), u=Action.ABSTAIN, m=Action.ABSTAIN, every=2)
    s = summarise(recs, CUT, 2000, 0.95, seed=0)
    assert s["verdict"] == "no evidence"
    assert f"{MAX_ABSTAIN_RATE:.0%}" in s["verdict_rule"]


def test_abstain_only_in_one_part_is_an_invalid_run():
    pre = _set_actions(make_records(40, 0, 0.0, 0.0), m=Action.ABSTAIN)
    post = [e for e in make_records(40, 40, 0.0, 0.0) if e.date >= CUT]
    s = summarise(pre + post, CUT, 2000, 0.95, seed=0)
    assert s["verdict"] == "invalid run"
    assert "pre" in s["verdict_reason"]


def test_abstain_rate_difference_between_arms_is_reported_and_flagged_when_large():
    # Masked arm abstains on 1 in 4 episodes, unmasked never: valid run, but behaviour changed.
    recs = _set_actions(make_records(40, 40, 0.0, 0.0), m=Action.ABSTAIN, every=4)
    s = summarise(recs, CUT, 2000, 0.95, seed=0)
    assert s["verdict"] == "no evidence"
    shift = s["abstain_shift"]
    assert shift["masked_minus_unmasked"] == pytest.approx(0.25)
    assert shift["flag"] is True
    assert s["arms"]["masked"]["abstain_rate"] == pytest.approx(0.25)


def test_small_abstain_rate_difference_is_not_flagged():
    recs = _set_actions(make_records(40, 40, 0.0, 0.0), m=Action.ABSTAIN, every=20)
    s = summarise(recs, CUT, 2000, 0.95, seed=0)
    assert s["abstain_shift"]["masked_minus_unmasked"] == pytest.approx(0.05)
    assert s["abstain_shift"]["flag"] is False


def _md(summary):
    from llm_lookahead_probe.report import build_report, render_markdown

    rep = build_report(
        summary,
        mode="offline",
        model="m",
        config={},
        cutoff=CUT,
        cutoff_source="t",
        masked_prompts_checked=0,
    )
    return render_markdown(rep)


def test_report_warns_about_a_large_abstain_shift_under_masking():
    recs = _set_actions(make_records(40, 40, 0.0, 0.0), m=Action.ABSTAIN, every=4)
    md = _md(summarise(recs, CUT, 2000, 0.95, seed=0))
    assert "Warning: abstain rate changes under masking" in md
    assert "+25.0" in md  # masked - unmasked, percentage points


def test_report_has_no_abstain_warning_when_rates_match():
    md = _md(summarise(make_records(40, 40, 0.0, 0.0), CUT, 2000, 0.95, seed=0))
    assert "Warning: abstain rate" not in md
    assert "Abstain rate" in md
