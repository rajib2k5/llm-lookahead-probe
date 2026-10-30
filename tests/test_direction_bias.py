"""Direction bias under masking, times drift: a false 'memorisation suspected' with no memory.

The defect (found in review): a model that is bullish on real tickers and bearish (or neutral) on
`ASSET_nn` pseudonyms has no memory, but its paired gap is (net_long_u - net_long_m) x mean r. In
a survivor-only panel the pre-cutoff period drifts up, so gap_pre > gap_post. The old rule read that
as memory, the abstain-shift flag stayed off, and the documented gap / mean|r| check pointed the
wrong way (this effect scales with mean r, not mean |r|).

Measured at a4fdd39 on the pipeline case below (10-ticker panel, 8 tickers x 200 dates, pre drift
+0.2%/day, post 0): 8 of 40 seeds 'memorisation suspected' (seeds 6, 9, 11, 16, 29, 32, 36, 39);
with no drift 0 of 40.
"""

from datetime import date, timedelta

import numpy as np
import pytest

from llm_lookahead_probe import stats
from llm_lookahead_probe.data import PricePanel, _business_days
from llm_lookahead_probe.masking import Masker
from llm_lookahead_probe.probe import execute, plan_episodes, sample_episodes
from llm_lookahead_probe.prompt import Action
from llm_lookahead_probe.stats import EpisodeResult, summarise

D0 = date(2024, 1, 1)
CUT = D0 + timedelta(days=100)


class RealLongMaskedShort:
    """No memory: LONG whenever the prompt names a real ticker, SHORT on a pseudonym."""

    def complete(self, prompt: str) -> str:
        return "SHORT" if "ASSET_" in prompt else "LONG"


def _drift_panel(seed: int, mu_pre: float, n_tickers: int = 10, n_days: int = 520, cut: int = 260):
    rng = np.random.default_rng(seed)
    dates = _business_days(date(2024, 1, 2), n_days)
    mu = np.where(np.arange(n_days - 1) < cut, mu_pre, 0.0)
    market = rng.normal(0, 0.01, n_days - 1) + mu
    log_r = market[:, None] + rng.normal(0, 0.015, (n_days - 1, n_tickers))
    p0 = np.exp(rng.uniform(np.log(20), np.log(300), n_tickers))
    log_p = np.vstack([np.log(p0)[None], np.log(p0)[None] + np.cumsum(log_r, 0)])
    tickers = tuple(f"T{chr(65 + i)}X" for i in range(n_tickers))
    return PricePanel(dates=dates, tickers=tickers, closes=np.round(np.exp(log_p), 2)), dates[cut]


def _bias_records(drift_pre: float, n_dates: int = 40, n_tickers: int = 8, seed: int = 0):
    """Unmasked LONG, masked SHORT on every episode; r = drift (pre only) + date shock + noise."""
    rng = np.random.default_rng(seed)
    out = []
    for k in range(2 * n_dates):
        d = CUT - timedelta(days=n_dates - k) if k < n_dates else CUT + timedelta(days=k)
        shock = (drift_pre if d < CUT else 0.0) + rng.normal(0, 0.01)
        for j in range(n_tickers):
            r = shock + rng.normal(0, 0.015)
            out.append(EpisodeResult(f"T{j}", d, r, Action.LONG, True, Action.SHORT, True))
    return out


def test_direction_bias_times_drift_is_not_reported_as_memorisation():
    s = summarise(_bias_records(drift_pre=0.01), CUT, 0, 0.95, seed=0)
    # Precondition: the raw gap alone passes both tests (this is the false positive).
    assert s["pre"]["gap"]["low"] > 0 and s["diff_pre_minus_post"]["low"] > 0
    assert s["verdict"] == "no evidence"


def test_direction_bias_times_drift_carries_a_warning_in_the_verdict_reason():
    s = summarise(_bias_records(drift_pre=0.01), CUT, 0, 0.95, seed=0)
    assert s["direction_shift"]["flag"] is True
    assert "direction" in s["verdict_reason"].lower()


def test_direction_shift_and_drift_are_reported_per_part():
    s = summarise(_bias_records(drift_pre=0.01), CUT, 0, 0.95, seed=0)
    for part in ("pre", "post"):
        u, m = s[part]["arms"]["unmasked"], s[part]["arms"]["masked"]
        assert (u["long_rate"], u["short_rate"], u["net_long_rate"]) == (1.0, 0.0, 1.0)
        assert (m["long_rate"], m["short_rate"], m["net_long_rate"]) == (0.0, 1.0, -1.0)
        mean_r = float(np.mean([e.r for e in _bias_records(0.01) if (e.date < CUT) == (part == "pre")]))
        assert s[part]["mean_r"] == pytest.approx(mean_r)
        assert s["direction_shift"]["by_part"][part] == pytest.approx(-2.0)  # masked - unmasked
        # (net_long_u - net_long_m) x mean r: all of the raw gap for an r-independent action.
        explained = s["direction_shift"]["gap_explained_by_direction_bias"][part]
        assert explained == pytest.approx(2.0 * mean_r)
        assert explained == pytest.approx(s[part]["gap"]["estimate"])
    assert stats.DIRECTION_SHIFT_FLAG == 0.10


def test_drift_adjusted_gap_is_raw_gap_minus_the_direction_bias_component():
    # Exact identity: mean pnl(a, r - mu) = mean pnl(a, r) - net_long * mu, per arm and per part.
    rng = np.random.default_rng(5)
    acts = [Action.LONG, Action.SHORT, Action.ABSTAIN]
    recs = [
        EpisodeResult(
            f"T{j}",
            D0 + timedelta(days=k),
            float(rng.normal(0.002, 0.02)),
            acts[rng.integers(3)],
            True,
            acts[rng.integers(3)],
            True,
        )
        for k in range(60)
        for j in range(4)
    ]
    cut = D0 + timedelta(days=30)
    s = summarise(recs, cut, 0, 0.95, seed=0)
    for part in ("pre", "post"):
        raw = s[part]["gap"]["estimate"]
        explained = s["direction_shift"]["gap_explained_by_direction_bias"][part]
        assert s["drift_adjusted"][part]["estimate"] == pytest.approx(raw - explained, abs=1e-15)
    assert s["drift_adjusted"]["diff_pre_minus_post"]["estimate"] == pytest.approx(
        s["drift_adjusted"]["pre"]["estimate"] - s["drift_adjusted"]["post"]["estimate"]
    )


def test_no_direction_warning_when_both_arms_trade_the_same_way():
    recs = [
        EpisodeResult(e.ticker, e.date, e.r, Action.LONG, True, Action.LONG, True)
        for e in _bias_records(drift_pre=0.01)
    ]
    s = summarise(recs, CUT, 0, 0.95, seed=0)
    assert s["direction_shift"]["flag"] is False
    assert "direction" not in s["verdict_reason"].lower()


def _report_md_and_cli(summary, capsys, tmp_path):
    from llm_lookahead_probe.cli import print_headline
    from llm_lookahead_probe.report import build_report, write_reports

    rep = build_report(
        summary, mode="offline", model="m", config={}, cutoff=CUT, cutoff_source="t", masked_prompts_checked=0
    )
    paths = write_reports(rep, tmp_path)
    print_headline(rep, paths)
    return paths[1].read_text(), capsys.readouterr().out


def test_report_and_cli_warn_about_a_direction_shift_times_drift(capsys, tmp_path):
    md, out = _report_md_and_cli(summarise(_bias_records(0.01), CUT, 0, 0.95, seed=0), capsys, tmp_path)
    assert "Warning: trade direction changes under masking" in md
    assert "Warning: trade direction changes under masking" in out
    assert "Drift-adjusted" in md and "drift-adjusted" in out


@pytest.mark.parametrize("seed", [6, 9, 11, 16, 29, 32, 36, 39])
def test_reviewer_case_bias_shift_model_on_a_drifting_panel_is_not_suspected(seed):
    # These are exactly the 8 seeds (of 40) that gave 'memorisation suspected' at a4fdd39.
    panel, cut = _drift_panel(seed, mu_pre=0.002)
    episodes = sample_episodes(panel, 8, 200, 20, cut, seed)
    plans = plan_episodes(panel, episodes, Masker.fit(panel, seed), 20)
    s = summarise(execute(plans, RealLongMaskedShort()).records, cut, 0, 0.95, seed)
    assert s["verdict"] == "no evidence"
    assert s["direction_shift"]["flag"] is True


def test_no_direction_warning_when_the_bias_pushes_pre_minus_post_below_zero():
    # Masked arm more bullish than unmasked in an up-drifting pre period: the bias makes
    # pre - post negative, which cannot fake memory, so there is nothing to warn about.
    recs = [
        EpisodeResult(e.ticker, e.date, e.r, Action.SHORT, True, Action.LONG, True)
        for e in _bias_records(drift_pre=0.01)
    ]
    s = summarise(recs, CUT, 0, 0.95, seed=0)
    assert s["diff_pre_minus_post"]["estimate"] < 0
    assert s["direction_shift"]["flag"] is False
