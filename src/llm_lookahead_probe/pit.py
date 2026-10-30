"""Point-in-time (as_of) guard for observations.

Same discipline as `queryFacts(entity, metric, as_of)` in quant-research-copilot
(`src/lib/core.ts`): a read without an as_of is refused; the date filter runs first so everything
downstream only sees knowable rows; and the result is asserted again after filtering, so a broken
filter raises instead of silently leaking a future row into a prompt.
"""

from __future__ import annotations

from bisect import bisect_right
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date
from typing import NamedTuple

from .data import PricePanel


class AsOfViolationError(Exception):
    """A read had no as_of, or a row dated after as_of reached an observation or prompt."""


class Bar(NamedTuple):
    date: date
    close: float


def require_as_of(as_of: object) -> date:
    if as_of is None or as_of == "":
        raise AsOfViolationError(
            "as_of is required: every read of the price panel must state the date whose knowledge it may use"
        )
    # datetime is a subclass of date; a time component has no meaning for daily closes.
    if not isinstance(as_of, date) or type(as_of) is not date:
        raise AsOfViolationError(f"as_of must be a datetime.date, received {as_of!r}")
    return as_of


def assert_no_future_rows(bars: Sequence[Bar], as_of: date) -> None:
    for b in bars:
        if b.date > as_of:
            raise AsOfViolationError(f"future row {b.date.isoformat()} > as_of {as_of.isoformat()}")


def _all_rows(panel: PricePanel, col: int) -> list[Bar]:
    return [Bar(d, float(panel.closes[i, col])) for i, d in enumerate(panel.dates)]


def _rows_on_or_before(panel: PricePanel, col: int, as_of: date, last_n: int | None = None) -> list[Bar]:
    # panel.dates is strictly increasing (PricePanel enforces it), so a bisect is the date filter.
    end = bisect_right(panel.dates, as_of)
    start = 0 if last_n is None else max(0, end - last_n)
    return [Bar(panel.dates[i], float(panel.closes[i, col])) for i in range(start, end)]


def point_in_time_history(
    panel: PricePanel, ticker: str, as_of: object, last_n: int | None = None
) -> list[Bar]:
    """Closes of `ticker` with date <= as_of (only the last `last_n` if given). Refuses without
    as_of; asserts again after filtering."""
    asof = require_as_of(as_of)
    rows = _rows_on_or_before(panel, panel.ticker_index(ticker), asof, last_n)
    assert_no_future_rows(rows, asof)
    return rows


@dataclass(frozen=True)
class Observation:
    """What the agent may see when deciding at the close of `as_of`."""

    ticker: str
    as_of: date
    bars: tuple[Bar, ...]

    def __post_init__(self) -> None:
        require_as_of(self.as_of)
        if not self.bars:
            raise ValueError("observation needs at least one bar")
        assert_no_future_rows(self.bars, self.as_of)
        if self.bars[-1].date != self.as_of:
            raise ValueError("last bar must be the decision date")


def build_observation(panel: PricePanel, ticker: str, as_of: object, window: int) -> Observation:
    """Last `window` closes of `ticker` ending at the close of as_of (a trading day in the panel)."""
    asof = require_as_of(as_of)
    if window < 2:
        raise ValueError("window must be >= 2")
    panel.date_index(asof)  # raises if not a trading day
    rows = point_in_time_history(panel, ticker, asof, last_n=window)
    if len(rows) < window:
        raise ValueError(f"window {window} needs {window} closes; only {len(rows)} on or before {asof}")
    return Observation(ticker=ticker, as_of=asof, bars=tuple(rows[-window:]))
