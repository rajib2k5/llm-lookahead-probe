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
**gap** is the mean of (pnl_unmasked - pnl_masked) over episodes. Its 95% CI comes from a
bootstrap that resamples decision dates, because episodes on the same date are correlated.
Episodes are split at the model's training cutoff.

**Verdict rule.** "memorisation suspected" if the lower CI bound of gap_pre > 0 AND the lower CI
bound of (gap_pre - gap_post) > 0. "insufficient data" if either part has fewer than 20 decision
dates. "no evidence" otherwise.

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
  no duplicates, no NaN and no price ≤ 0. Every ticker must have every date. Tickers must match
  `[A-Z0-9][A-Z0-9.-]{0,9}`. Size limits: 50 MB, 2,000,000 rows and 1,000 tickers.
- The default model is `claude-haiku-5-5`, the cheapest current Claude model. It costs $0.10 per
  MTok input and $0.50 per MTok output. Its training-data cutoff is Jun 2026, which the probe
  uses as 2026-06-01; override it with `--cutoff`. The sources for these values are in
  `src/llm_lookahead_probe/live.py`.
- `--max-episodes` (default 50) is a hard cap. If the run needs more episodes, it stops before
  any call.
- The estimated cost is printed before any call. Without `--yes`, no call is made.
- Responses are cached in `.cache/llm-lookahead-probe/`, keyed by sha256(model, prompt). A
  rerun costs nothing.
- `report.json` records the token usage that the API responses report.

### Cost per run (estimate, not a measured bill)

Each episode makes 2 calls. A prompt with 20 closes is about 584 characters. The estimator
assumes 2 characters per token, which is on the high side, plus the full 16-token output cap.
It therefore gives an upper bound:

| Episodes | Calls | Estimated upper bound |
|---|---|---|
| 50 (default cap; 5 tickers x 10 dates) | 100 | $0.0037 |
| 200 (e.g. 5 tickers x 40 dates, enough dates for a verdict) | 400 | about $0.015 |
| 1,000 | 2,000 | $0.074 |
| 2,400 (20 tickers x 120 dates) | 4,800 | $0.18 |

The default live config (5 x 10) gives only 5 dates per part, so its verdict is always
"insufficient data". For a verdict, use at least 20 dates on each side of the cutoff and raise
`--max-episodes` to match. No live run against the real API has been made yet, so these figures
have not been checked against a bill.

## Development

```bash
uv run pytest -q
uv run ruff check
```
