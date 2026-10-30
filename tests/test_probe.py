"""End-to-end through the public API on the default offline config."""

from datetime import date

import numpy as np
import pytest

import llm_lookahead_probe.probe as probe_mod
from llm_lookahead_probe.data import PricePanel, _business_days, synthetic_panel
from llm_lookahead_probe.masking import LeakError, Masker
from llm_lookahead_probe.probe import OfflineConfig, plan_episodes, run_offline, sample_episodes


@pytest.fixture(scope="module")
def default_reports():
    cfg = OfflineConfig()
    return {kind: run_offline(kind, cfg) for kind in ("memorising", "honest")}


def test_memorising_mock_is_flagged_on_default_config(default_reports):
    rep = default_reports["memorising"]
    assert rep["verdict"] == "memorisation suspected"
    assert rep["pre"]["gap"]["low"] > 0


def test_honest_mock_shows_no_evidence_on_default_config(default_reports):
    rep = default_reports["honest"]
    assert rep["verdict"] == "no evidence"
    assert rep["gap"]["low"] <= 0 <= rep["gap"]["high"]


def test_report_records_counts_rule_and_leak_checks(default_reports):
    rep = default_reports["honest"]
    cfg = OfflineConfig()
    assert rep["n_episodes"] == cfg.n_tickers * cfg.n_dates
    assert rep["leak_check"]["masked_prompts_checked"] == rep["n_episodes"]
    assert "lower CI bound of gap_pre > 0" in rep["verdict_rule"]
    assert rep["interval"]["method"].startswith("t-interval")
    assert rep["robustness"]["cluster_bootstrap"]["method"].startswith("cluster")
    assert rep["arms"]["unmasked"]["n_unparseable"] > 0  # mocks garble a few answers


def test_episodes_are_stratified_around_cutoff_and_reproducible():
    cfg = OfflineConfig()
    panel = synthetic_panel(cfg.seed, cfg.n_tickers, cfg.n_days)
    eps = sample_episodes(panel, cfg.n_tickers, cfg.n_dates, cfg.window, cfg.cutoff, cfg.seed)
    assert eps == sample_episodes(panel, cfg.n_tickers, cfg.n_dates, cfg.window, cfg.cutoff, cfg.seed)
    dates = sorted({e.date for e in eps})
    assert len(dates) == cfg.n_dates
    assert sum(d < cfg.cutoff for d in dates) == cfg.n_dates // 2
    assert all(panel.date_index(d) >= cfg.window - 1 for d in dates)
    assert all(panel.date_index(d) < len(panel.dates) - 1 for d in dates)


def test_plan_aborts_on_a_leaky_masker():
    cfg = OfflineConfig(n_tickers=3, n_dates=4)
    panel = synthetic_panel(cfg.seed, cfg.n_tickers, cfg.n_days)
    eps = sample_episodes(panel, 3, 4, cfg.window, cfg.cutoff, cfg.seed)
    m = Masker.fit(panel, seed=0)
    leaky = Masker(pseudonyms={t: t for t in panel.tickers}, factors=m.factors, shift_weeks=m.shift_weeks)
    with pytest.raises(LeakError):
        plan_episodes(panel, eps, leaky, cfg.window)


def volatile_panel(seed=0, n=60):
    """One low-priced ticker with 25% daily volatility: masked cents often equal a real price of
    the same episode. With masker seed 3 the v0.1 masker hit such a collision ('2.10')."""
    rng = np.random.default_rng(seed)
    lr = rng.normal(0, 0.25, n - 1)
    p = np.maximum(np.round(np.exp(np.log(2.0) + np.concatenate([[0], np.cumsum(lr)])), 2), 0.01)
    return PricePanel(_business_days(date(2024, 1, 2), n), ("VOL",), p[:, None])


def test_price_collision_on_a_volatile_series_redraws_the_factor_instead_of_aborting():
    panel = volatile_panel()
    eps = sample_episodes(panel, 1, 30, 20, None, seed=0)
    masker = Masker.fit(panel, seed=3)
    plans = plan_episodes(panel, eps, masker, 20)
    assert len(plans) == len(eps)
    used = plans[0].obs_m.bars[0].close / plans[0].obs_u.bars[0].close
    assert used != pytest.approx(masker.factors["VOL"], rel=1e-3)  # a new factor was drawn
    ratios = [pl.obs_m.bars[0].close / pl.obs_u.bars[0].close for pl in plans]
    assert ratios == pytest.approx([used] * len(plans), rel=1e-9)
    assert 1.5 <= used <= 6.0
    again = plan_episodes(panel, eps, Masker.fit(panel, seed=3), 20)
    assert [p.prompt_m for p in again] == [p.prompt_m for p in plans]  # deterministic redraw


def test_price_collision_still_fails_closed_when_redraws_are_exhausted(monkeypatch):
    panel = volatile_panel()
    eps = sample_episodes(panel, 1, 30, 20, None, seed=0)
    monkeypatch.setattr(probe_mod, "MAX_FACTOR_REDRAWS", 0)
    with pytest.raises(LeakError, match="price"):
        plan_episodes(panel, eps, Masker.fit(panel, seed=3), 20)


def test_a_masker_without_a_seed_is_never_redrawn():
    panel = volatile_panel()
    eps = sample_episodes(panel, 1, 30, 20, None, seed=0)
    m = Masker.fit(panel, seed=3)
    fixed = Masker(pseudonyms=m.pseudonyms, factors=m.factors, shift_weeks=m.shift_weeks)
    with pytest.raises(LeakError, match="price"):
        plan_episodes(panel, eps, fixed, 20)


def test_masked_prompt_showing_the_real_next_close_is_a_leak():
    # Window closes are all 10.00 and the next close is 20.00. A factor of 2 shows 20.00 in the
    # masked prompt: no window price leaks, but the price the decision is scored on does.
    closes = np.array([[10.0]] * 20 + [[20.0]])
    panel = PricePanel(_business_days(date(2024, 1, 2), 21), ("AAA",), closes)
    eps = sample_episodes(panel, 1, 1, 20, None, seed=0)
    assert eps[0].date == panel.dates[19]
    m = Masker.fit(panel, seed=0)
    fixed = Masker(pseudonyms=m.pseudonyms, factors={"AAA": 2.0}, shift_weeks=m.shift_weeks)
    with pytest.raises(LeakError, match="real price 20.00"):
        plan_episodes(panel, eps, fixed, 20)
