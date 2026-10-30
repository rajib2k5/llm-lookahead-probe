"""End-to-end through the public API on the default offline config."""

import pytest

from llm_lookahead_probe.data import synthetic_panel
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
    assert rep["bootstrap"]["method"].startswith("cluster")
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
