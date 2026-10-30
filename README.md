# llm-lookahead-probe

Tests if the backtest result of an LLM trading agent is inflated because the model memorised history.

> Draft README for v0.1. Analysis only: no orders, no broker code, no real money.

## How it works

The probe asks the same daily decision (LONG, SHORT or ABSTAIN for the next day) twice for each
episode (ticker, decision date):

- **unmasked**: real ticker, real dates, real prices;
- **masked**: pseudonym ticker (`ASSET_07`), dates shifted back by whole weeks (same weekday), prices
  multiplied by a per-ticker factor (same returns).

pnl is +r for LONG, -r for SHORT and 0 for ABSTAIN, where r is the real next-day return. The
**gap** is the mean of (pnl_unmasked - pnl_masked) over episodes. Episodes on the same date are
correlated, so each decision date is one observation: the 95% CI is a Student t interval on the
per-date mean differences with G - 1 degrees of freedom (G = number of dates). The pre - post CI
is Welch's interval. Episodes are split at the model's training cutoff. A percentile cluster
bootstrap is also reported as a secondary number; it is not used by the verdict because it
covers only about 93.5-94.5% at 30-60 dates.

**Verdict rule.** "invalid run" if either arm has more than 10% unparseable answers (refusals,
empty or truncated replies), in the whole run or in the pre or post part. Else "insufficient
data" if either part has fewer than 20 decision dates. Else "memorisation suspected" if the lower
CI bound of gap_pre > 0 AND the lower CI bound of (gap_pre - gap_post) > 0. Else "no evidence".

The report also gives gap / mean|r| for each period. pre - post cancels only a masking effect
that is constant in pnl. A masking effect that is constant in behaviour (for example, more
ABSTAIN answers when masked) scales with |r|. If the raw gaps differ but the scaled gaps of the
two periods are about equal, the pre - post difference can come from volatility, not memory. The other limits are listed in each
report and in `docs/DESIGN.md`.

Each masked prompt must pass a leak check before any model call: no real ticker, no real date and
no real price of that episode. Observations come from an as_of guard, so no row after the
decision date can reach a prompt. See `docs/DESIGN.md` for the design and its limits.

## Install

```bash
uv sync                 # offline mode and tests
uv sync --extra live    # adds the anthropic SDK for live mode
```

## Offline (no network, no key)

```bash
uv run llm-lookahead-probe run --offline --mock memorising --out reports/mem
uv run llm-lookahead-probe run --offline --mock honest --out reports/honest
```

The offline data is a seeded synthetic panel (20 tickers, 520 business days). `memorising` is a
mock model that "remembers" the true next-day direction for real (ticker, date) pairs before its
cutoff. `honest` uses only the price path. Options: `--seed`, `--n-tickers`, `--n-dates`,
`--window`, `--cutoff`, `--bootstrap`, `--out`.

## Live (Claude, costs money)

```bash
export ANTHROPIC_API_KEY=...        # read from the environment only, never printed
uv run llm-lookahead-probe run --live --prices my_prices.csv              # dry run: prints the estimate
uv run llm-lookahead-probe run --live --prices my_prices.csv --yes        # spends
```

- The CSV must have the header `date,ticker,close`. Rows must be sorted by (date, ticker), with
  no duplicates. A close is a plain decimal (`12.34` or `12`; no sign, spaces, exponent or
  underscores), > 0 and ≤ 10,000,000. Dates are `YYYY-MM-DD`, on or after 1900-01-01. Every
  ticker must have every date. Tickers must match `[A-Z0-9][A-Z0-9.-]{0,9}`. Size limits: 50 MB,
  2,000,000 rows and 1,000 tickers.
- The default model is `claude-haiku-5-5`, the cheapest current Claude model. It costs $0.10 per
  MTok input and $0.50 per MTok output. Its training-data cutoff is Jun 2026, which the probe
  uses as 2026-06-01; override it with `--cutoff`. The sources for these values are in
  `src/llm_lookahead_probe/live.py`.
- `--max-episodes` (default 50) is a hard cap. If the run needs more episodes, it stops before
  any call.
- The estimated cost is printed before any call. Without `--yes`, no call is made and the cache
  directory is not created.
- The SDK's automatic retries are off (`max_retries=0`), so every HTTP request is a counted call.
  If a call fails, the run stops; run it again and the completed calls come from the cache.
- Responses are cached in `.cache/llm-lookahead-probe/`, keyed by sha256 of the full request
  (model, prompt, `max_tokens`, thinking and effort settings). A rerun with the same settings
  costs nothing; a changed setting makes new calls. A new cache directory gets mode 0700; an
  existing one keeps its mode, and a group- or world-writable one is refused.
- The cache is not tamper-proof. Its stored key detects a copied or corrupt entry, not an edited
  one. Anyone who can write to the cache directory controls the answers that the probe scores.
- `report.json` records the token usage that the API responses report.

### Cost per run (estimate, not a measured bill)

Each episode makes 2 calls. A prompt with 20 closes is about 584 bytes. The estimator counts
at most one input token per UTF-8 byte (a byte-level BPE token covers at least one byte), plus
an allowance of 64 tokens per request for message framing, plus the full 16-token output cap.
Real prompts of dates and prices use about one token per 3 characters, so the bound is loose by
about 2-3x. The 64-token allowance is an assumption that no live call has checked. The figures
below are the estimator's output for the synthetic panel with a 20-close window:

| Episodes | Calls | Input tokens (bound) | Output tokens (bound) | Estimated upper bound |
|---|---|---|---|---|
| 50 (default cap; 5 tickers x 10 dates) | 100 | 64,754 | 1,600 | $0.0073 |
| 200 (5 tickers x 40 dates, enough dates for a verdict) | 400 | 256,826 | 6,400 | $0.0289 |
| 1,000 (10 tickers x 100 dates) | 2,000 | 1,303,384 | 32,000 | $0.1463 |
| 2,400 (20 tickers x 120 dates) | 4,800 | 3,110,640 | 76,800 | $0.3495 |

The default live config (5 x 10) gives only 5 dates per part, so its verdict is always
"insufficient data". For a verdict, use at least 20 dates on each side of the cutoff and raise
`--max-episodes` to match. No live run against the real API has been made yet, so these figures
have not been checked against a bill.

## Development

```bash
uv run pytest -q
uv run ruff check
```
