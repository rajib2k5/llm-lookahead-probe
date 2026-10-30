"""as_of guard: an observation for a decision at date t sees only rows with date <= t."""

from datetime import date

import pytest

from llm_lookahead_probe.data import synthetic_panel
from llm_lookahead_probe.pit import (
    AsOfViolationError,
    Bar,
    Observation,
    build_observation,
    point_in_time_history,
)
from llm_lookahead_probe.prompt import render_prompt


@pytest.fixture
def panel():
    return synthetic_panel(seed=1, n_tickers=3, n_days=40)


@pytest.mark.parametrize("missing", [None, ""])
def test_history_refuses_without_as_of(panel, missing):
    with pytest.raises(AsOfViolationError, match="as_of is required"):
        point_in_time_history(panel, panel.tickers[0], missing)


def test_observation_guard_raises_if_panel_filter_is_bypassed(panel, monkeypatch):
    import llm_lookahead_probe.pit as pit

    monkeypatch.setattr(
        pit, "_rows_on_or_before", lambda panel, col, as_of, last_n=None: pit._all_rows(panel, col)
    )
    with pytest.raises(AsOfViolationError, match="future"):
        build_observation(panel, panel.tickers[0], panel.dates[10], window=5)


def test_build_observation_refuses_without_as_of(panel):
    with pytest.raises(AsOfViolationError, match="as_of is required"):
        build_observation(panel, panel.tickers[0], None, window=5)


def test_as_of_must_be_a_date(panel):
    with pytest.raises(AsOfViolationError, match="date"):
        point_in_time_history(panel, panel.tickers[0], "2024-01-10")


def test_history_returns_only_rows_on_or_before_as_of(panel):
    t = panel.dates[20]
    rows = point_in_time_history(panel, panel.tickers[0], t)
    assert rows[-1].date == t
    assert all(r.date <= t for r in rows)
    assert len(rows) == 21


def test_observation_holds_last_window_closes_ending_at_as_of(panel):
    t = panel.dates[20]
    obs = build_observation(panel, panel.tickers[1], t, window=5)
    assert obs.as_of == t
    assert [b.date for b in obs.bars] == list(panel.dates[16:21])
    assert [b.close for b in obs.bars] == list(panel.closes[16:21, 1])


def test_observation_refuses_when_window_is_not_available(panel):
    with pytest.raises(ValueError, match="window"):
        build_observation(panel, panel.tickers[0], panel.dates[2], window=5)


def test_observation_refuses_non_trading_as_of(panel):
    saturday = date(2024, 1, 6)
    with pytest.raises(ValueError, match="trading day"):
        build_observation(panel, panel.tickers[0], saturday, window=2)


def test_observation_with_a_future_row_raises():
    with pytest.raises(AsOfViolationError, match="future"):
        Observation(
            ticker="AAA",
            as_of=date(2024, 1, 3),
            bars=(
                Bar(date(2024, 1, 2), 10.0),
                Bar(date(2024, 1, 3), 11.0),
                Bar(date(2024, 1, 4), 12.0),
            ),
        )


def test_future_row_smuggled_past_construction_is_caught_at_render():
    obs = Observation(
        ticker="AAA",
        as_of=date(2024, 1, 3),
        bars=(Bar(date(2024, 1, 2), 10.0), Bar(date(2024, 1, 3), 11.0)),
    )
    # Bypass the dataclass guard the way a careless refactor might.
    object.__setattr__(obs, "bars", obs.bars + (Bar(date(2024, 1, 4), 12.0),))
    with pytest.raises(AsOfViolationError, match="future"):
        render_prompt(obs)


def test_history_guard_raises_if_panel_filter_is_bypassed(panel, monkeypatch):
    """If the date filter ever returns a future row, the post-filter assertion must still raise."""
    import llm_lookahead_probe.pit as pit

    monkeypatch.setattr(
        pit, "_rows_on_or_before", lambda panel, col, as_of, last_n=None: pit._all_rows(panel, col)
    )
    with pytest.raises(AsOfViolationError, match="future"):
        point_in_time_history(panel, panel.tickers[0], panel.dates[10])
