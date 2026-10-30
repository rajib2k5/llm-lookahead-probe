"""Masking (anonymise ticker, shift dates, rescale prices) and the leak check.

What masking keeps: the return path the agent may legitimately use (returns are invariant to a
per-ticker price factor) and the weekday of every date. What it removes: every key a model would
need to look an episode up in memory (ticker, calendar date, price level).

Design choices (see docs/DESIGN.md):
- Pseudonym = position in a seeded permutation of the sorted ticker list. The number carries no
  information about the ticker text.
- One global date shift, backwards by a whole number of weeks larger than the panel span, so no
  masked date can coincide with any real date of the panel. A leak check on dates is therefore
  exact, not probabilistic.
- Price factor drawn log-uniform in [1.5, 6], scale-up only: a factor bounded away from 1 keeps
  masked price levels from overlapping the real window (unless the window moves > 50%), and
  scaling up loses no precision when prices are rounded to cents for display.
"""

from __future__ import annotations

import math
import re
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import timedelta

import numpy as np

from .data import PricePanel
from .pit import Bar, Observation
from .prompt import format_price

FACTOR_LOW, FACTOR_HIGH = 1.5, 6.0


class LeakError(AssertionError):
    """A masked prompt contains a real ticker, real date or real price string."""


@dataclass(frozen=True)
class Masker:
    pseudonyms: dict[str, str]
    factors: dict[str, float]
    shift_weeks: int

    @classmethod
    def fit(cls, panel: PricePanel, seed: int) -> Masker:
        rng = np.random.default_rng(np.random.SeedSequence([seed, 0x3A5C]))
        tickers = sorted(panel.tickers)
        n = len(tickers)
        width = max(2, len(str(n - 1)))
        perm = rng.permutation(n)
        pseudonyms = {t: f"ASSET_{int(k):0{width}d}" for t, k in zip(tickers, perm, strict=True)}
        log_f = rng.uniform(math.log(FACTOR_LOW), math.log(FACTOR_HIGH), size=n)
        factors = {t: float(math.exp(x)) for t, x in zip(tickers, log_f, strict=True)}
        span_weeks = (panel.dates[-1] - panel.dates[0]).days // 7 + 1
        shift_weeks = -(span_weeks + 1 + int(rng.integers(0, 105)))
        return cls(pseudonyms=pseudonyms, factors=factors, shift_weeks=shift_weeks)

    def pseudonym(self, ticker: str) -> str:
        return self.pseudonyms[ticker]

    def mask(self, obs: Observation) -> Observation:
        shift = timedelta(weeks=self.shift_weeks)
        f = self.factors[obs.ticker]
        bars = tuple(Bar(b.date + shift, b.close * f) for b in obs.bars)
        return Observation(ticker=self.pseudonyms[obs.ticker], as_of=obs.as_of + shift, bars=bars)


_DATE_TOKEN = re.compile(r"(?<!\d)\d{4}-\d{2}-\d{2}(?!\d)")
_NUMBER_TOKEN = re.compile(r"(?<![\d.])\d+(?:\.\d+)?")


class LeakChecker:
    """Fails (LeakError) if a masked prompt contains any real ticker as a whole token, any date of
    the real panel as YYYY-MM-DD, or any of the given real prices as a price token.

    Prices are checked per episode (the caller passes that episode's real window and next close),
    not against the whole panel: a random cents value collides with *some* price of a large panel
    about half the time, so a panel-wide price check would fail on coincidence, not on leaks.
    """

    def __init__(self, panel: PricePanel) -> None:
        alts = "|".join(re.escape(t) for t in sorted(panel.tickers, key=len, reverse=True))
        self._ticker_re = re.compile(
            rf"(?<![A-Za-z0-9_])(?<![A-Za-z0-9][.\-])(?:{alts})(?![A-Za-z0-9_])(?![.\-][A-Za-z0-9])"
        )
        self._dates = {d.isoformat() for d in panel.dates}
        self.checked = 0

    def check(self, masked_prompt: str, real_prices: Iterable[float]) -> None:
        m = self._ticker_re.search(masked_prompt)
        if m:
            raise LeakError(f"masked prompt contains real ticker {m.group(0)!r}")
        for d in _DATE_TOKEN.findall(masked_prompt):
            if d in self._dates:
                raise LeakError(f"masked prompt contains real date {d}")
        prices = {format_price(p) for p in real_prices}
        for tok in _NUMBER_TOKEN.findall(masked_prompt):
            if tok in prices:
                raise LeakError(f"masked prompt contains real price {tok}")
        self.checked += 1
