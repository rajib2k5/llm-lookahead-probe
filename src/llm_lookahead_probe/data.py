"""Price panels: a deterministic synthetic generator and a strict CSV loader.

No vendor data ships with this package. Offline mode uses `synthetic_panel`; live mode reads a
user CSV with columns `date,ticker,close`, validated strictly because its contents end up inside
LLM prompts (see the ticker charset rule) and drive the statistics (no NaN, no non-positive price).
"""

from __future__ import annotations

import math
import os
import re
import stat
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path

import numpy as np

# Words the prompt template uses in upper case. A ticker equal to one of them would make the leak
# check ambiguous (the template itself would "contain" the ticker), so the loader rejects them.
RESERVED_WORDS = frozenset({"LONG", "SHORT", "ABSTAIN"})

# Letters, digits, dot and dash only. This keeps free text (e.g. a prompt-injection sentence) out of
# the ticker field, and makes an `ASSET_nn` pseudonym impossible to confuse with a real ticker.
TICKER_RE = re.compile(r"[A-Z0-9][A-Z0-9.\-]{0,9}")
DATE_RE = re.compile(r"\d{4}-\d{2}-\d{2}")

DEFAULT_MAX_BYTES = 50 * 1024 * 1024
DEFAULT_MAX_ROWS = 2_000_000
DEFAULT_MAX_TICKERS = 1_000
MAX_LINE_CHARS = 200
# A close is plain decimal text: digits, optionally a dot and more digits. `float()` alone also
# accepts "1_000", " 5 ", "+6", "1e3", "nan" and "inf"; none of those is a price a data vendor
# writes, so they are rejected rather than silently reinterpreted.
CLOSE_RE = re.compile(r"[0-9]+(?:\.[0-9]+)?")
# Upper bound on a close. Masking multiplies prices by at most 6, so every masked price stays a
# small, finite number that renders as plain d.dd text (no exponent, no inf).
MAX_CLOSE = 1e7
# Earliest allowed date. Masking shifts dates back by the panel span plus up to ~2 years; an early
# date (e.g. year 0001) would underflow `datetime.date` and crash the run.
MIN_YEAR = 1900


class PriceDataError(ValueError):
    """The price CSV failed validation. The message names the first problem and its line."""


@dataclass(frozen=True)
class PricePanel:
    """Rectangular close-price panel: closes[i, j] is the close of tickers[j] on dates[i]."""

    dates: tuple[date, ...]
    tickers: tuple[str, ...]
    closes: np.ndarray

    def __post_init__(self) -> None:
        if self.closes.shape != (len(self.dates), len(self.tickers)):
            raise ValueError("closes shape must be (len(dates), len(tickers))")
        if any(b <= a for a, b in zip(self.dates, self.dates[1:], strict=False)):
            raise ValueError("dates must be strictly increasing")
        if len(set(self.tickers)) != len(self.tickers):
            raise ValueError("tickers must be unique")
        if not np.isfinite(self.closes).all() or (self.closes <= 0).any():
            raise ValueError("closes must be finite and positive")
        object.__setattr__(self, "_date_index", {d: i for i, d in enumerate(self.dates)})
        object.__setattr__(self, "_ticker_index", {t: j for j, t in enumerate(self.tickers)})
        self.closes.setflags(write=False)

    def date_index(self, d: date) -> int:
        try:
            return self._date_index[d]  # type: ignore[attr-defined]
        except KeyError:
            raise ValueError(f"{d} is not a trading day in this panel") from None

    def ticker_index(self, ticker: str) -> int:
        try:
            return self._ticker_index[ticker]  # type: ignore[attr-defined]
        except KeyError:
            raise ValueError(f"unknown ticker {ticker!r}") from None

    def next_day_return(self, ticker: str, d: date) -> float:
        """Simple return from the close of d to the close of the next trading day."""
        i, j = self.date_index(d), self.ticker_index(ticker)
        if i + 1 >= len(self.dates):
            raise ValueError(f"{d} is the last date; no next-day return")
        return float(self.closes[i + 1, j] / self.closes[i, j] - 1.0)


def _business_days(start: date, n: int) -> tuple[date, ...]:
    out: list[date] = []
    d = start
    while len(out) < n:
        if d.weekday() < 5:
            out.append(d)
        d += timedelta(days=1)
    return tuple(out)


def _synthetic_tickers(rng: np.random.Generator, n: int) -> tuple[str, ...]:
    letters = np.array(list("ABCDEFGHIJKLMNOPQRSTUVWXYZ"))
    seen: set[str] = set()
    out: list[str] = []
    while len(out) < n:
        k = int(rng.integers(3, 5))
        name = "".join(rng.choice(letters, size=k))
        if name in seen or name in RESERVED_WORDS or name.startswith("ASSET"):
            continue
        seen.add(name)
        out.append(name)
    return tuple(out)


def synthetic_panel(
    seed: int,
    n_tickers: int = 20,
    n_days: int = 520,
    start: date = date(2024, 1, 2),
) -> PricePanel:
    """Deterministic one-factor price panel (market factor + idiosyncratic noise).

    The common market factor makes same-date returns correlated, which is the situation that makes
    a by-date cluster bootstrap necessary. Closes are rounded to cents like real data.
    """
    if n_tickers < 1 or n_days < 2:
        raise ValueError("need at least 1 ticker and 2 days")
    rng = np.random.default_rng(np.random.SeedSequence([seed, 0xDA7A]))
    tickers = _synthetic_tickers(rng, n_tickers)
    dates = _business_days(start, n_days)
    market = rng.normal(0.0003, 0.010, size=n_days - 1)
    beta = rng.uniform(0.5, 1.5, size=n_tickers)
    idio = rng.normal(0.0, 0.015, size=(n_days - 1, n_tickers))
    log_r = market[:, None] * beta[None, :] + idio
    p0 = np.exp(rng.uniform(math.log(20), math.log(300), size=n_tickers))
    log_p = np.vstack([np.log(p0)[None, :], np.log(p0)[None, :] + np.cumsum(log_r, axis=0)])
    closes = np.maximum(np.round(np.exp(log_p), 2), 0.01)
    return PricePanel(dates=dates, tickers=tickers, closes=closes)


def load_prices_csv(
    path: str | os.PathLike[str],
    *,
    max_bytes: int = DEFAULT_MAX_BYTES,
    max_rows: int = DEFAULT_MAX_ROWS,
    max_tickers: int = DEFAULT_MAX_TICKERS,
) -> PricePanel:
    """Load and strictly validate a `date,ticker,close` CSV.

    Rules: exact header; ISO dates on or after MIN_YEAR-01-01; ticker matches TICKER_RE and is not
    a reserved word; close matches CLOSE_RE (plain decimal), is positive and at most MAX_CLOSE;
    rows strictly increasing in (date, ticker), which also forbids
    duplicates; every ticker present on every date (rectangular); size limits on bytes, rows,
    tickers and line length. Raises PriceDataError naming the first failing line.
    """
    p = Path(path)
    try:
        st = p.stat()
    except OSError:
        raise PriceDataError(f"{p}: not a regular file") from None
    if not stat.S_ISREG(st.st_mode):
        raise PriceDataError(f"{p}: not a regular file")
    if st.st_size > max_bytes:
        raise PriceDataError(f"{p}: file size {st.st_size} bytes exceeds limit {max_bytes}")

    rows: list[tuple[date, str, float]] = []
    with p.open("r", encoding="utf-8", newline="") as fh:
        header = fh.readline(MAX_LINE_CHARS + 1).strip("\r\n")
        if header != "date,ticker,close":
            raise PriceDataError("line 1: header must be exactly 'date,ticker,close'")
        prev: tuple[date, str] | None = None
        for lineno, line in enumerate(fh, start=2):
            if len(line) > MAX_LINE_CHARS:
                raise PriceDataError(f"line {lineno}: longer than {MAX_LINE_CHARS} characters")
            line = line.rstrip("\r\n")
            if not line:
                raise PriceDataError(f"line {lineno}: empty line")
            if len(rows) >= max_rows:
                raise PriceDataError(f"more than {max_rows} rows")
            parts = line.split(",")
            if len(parts) != 3:
                raise PriceDataError(f"line {lineno}: expected 3 columns, got {len(parts)}")
            ds, ticker, cs = parts
            if not DATE_RE.fullmatch(ds):
                raise PriceDataError(f"line {lineno}: date must be YYYY-MM-DD")
            try:
                d = date.fromisoformat(ds)
            except ValueError:
                raise PriceDataError(f"line {lineno}: invalid date {ds!r}") from None
            if d.year < MIN_YEAR:
                raise PriceDataError(f"line {lineno}: date {ds} is before {MIN_YEAR}-01-01")
            if not TICKER_RE.fullmatch(ticker):
                raise PriceDataError(f"line {lineno}: ticker must match {TICKER_RE.pattern}")
            if ticker in RESERVED_WORDS:
                raise PriceDataError(f"line {lineno}: ticker {ticker!r} is a reserved word")
            if not CLOSE_RE.fullmatch(cs):
                raise PriceDataError(
                    f"line {lineno}: close is not a plain decimal number (digits with an optional "
                    "fraction, e.g. 12.34; no sign, spaces, exponent, underscores, nan or inf); "
                    "it must be finite and positive"
                )
            close = float(cs)
            if not math.isfinite(close):  # unreachable for CLOSE_RE text of <= 200 chars; kept
                raise PriceDataError(f"line {lineno}: close must be finite")
            if close <= 0:
                raise PriceDataError(f"line {lineno}: close must be positive")
            if close > MAX_CLOSE:
                raise PriceDataError(f"line {lineno}: close {cs} exceeds the maximum {MAX_CLOSE:.0f}")
            key = (d, ticker)
            if prev is not None and key <= prev:
                raise PriceDataError(
                    f"line {lineno}: rows must be sorted strictly by (date, ticker) with no duplicates"
                )
            prev = key
            rows.append((d, ticker, close))

    if not rows:
        raise PriceDataError("no rows after the header")
    dates = tuple(sorted({r[0] for r in rows}))
    tickers = tuple(sorted({r[1] for r in rows}))
    if len(tickers) > max_tickers:
        raise PriceDataError(f"{len(tickers)} tickers exceeds limit {max_tickers}")
    if len(rows) != len(dates) * len(tickers):
        raise PriceDataError(
            "panel is not rectangular: every ticker must have a close on every date "
            f"({len(dates)} dates x {len(tickers)} tickers != {len(rows)} rows)"
        )
    # Rows are sorted by (date, ticker) and rectangular, so a reshape is exact.
    closes = np.array([r[2] for r in rows], dtype=float).reshape(len(dates), len(tickers))
    return PricePanel(dates=dates, tickers=tickers, closes=closes)
