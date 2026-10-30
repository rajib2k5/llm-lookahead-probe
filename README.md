# llm-lookahead-probe

A probe that measures how much the backtest of an LLM trading agent is inflated by memorised history.

## What it does and why

An LLM that trained on past market data can "predict" a move because it remembers the move. A backtest over that period then looks better than the agent will perform on new data. The probe runs each decision twice. The first prompt has the real ticker, the real dates, and the real prices. The second prompt is masked, so the model cannot look up the episode. The probe reports the gap between the two results as one number with a 95% confidence interval. It also compares the gap before the training cutoff of the model with the gap after it. Memory can create a gap before the cutoff. It cannot create a gap after the cutoff.

The probe gives evidence, not proof. It is analysis only. It places no orders, has no broker code, and uses no real money.

## Install

You need Python and [uv](https://docs.astral.sh/uv/).

```bash
uv sync                 # offline mode and tests
uv sync --extra live    # adds the anthropic SDK for live mode
```

## Quick start (offline, no network, no key)

The offline data is a seeded synthetic panel (20 tickers, 520 business days). Two mock models show what the probe reports.

```bash
uv run llm-lookahead-probe run --offline --mock memorising --out reports/mem
uv run llm-lookahead-probe run --offline --mock honest --out reports/honest
```

The `memorising` mock remembers the true next-day direction for real (ticker, date) pairs before its cutoff. The `honest` mock uses only the price path.

Output of the first command (headline lines):

```text
Verdict: memorisation suspected
  gap_pre CI low 0.009371 > 0 and (gap_pre - gap_post) CI low 0.008162 > 0
Episodes: 2400 on 120 dates; cutoff 2025-01-02
Gap (unmasked - masked, mean pnl/episode, 95% t-interval): +0.5737% [+0.4402%, +0.7071%]
  pre-cutoff  (60 dates): +1.0791% [+0.9371%, +1.2210%]; gap / mean|r| +0.7468
  post-cutoff (60 dates): +0.0683% [-0.0679%, +0.2044%]; gap / mean|r| +0.0453
  pre - post: +1.0108% [+0.8162%, +1.2055%]
```

Output of the second command (headline lines):

```text
Verdict: no evidence
  gap_pre CI low -0.000173, (gap_pre - gap_post) CI low -0.001463: both must be > 0
Episodes: 2400 on 120 dates; cutoff 2025-01-02
Gap (unmasked - masked, mean pnl/episode, 95% t-interval): +0.0839% [-0.0045%, +0.1724%]
  pre-cutoff  (60 dates): +0.0996% [-0.0173%, +0.2166%]; gap / mean|r| +0.0690
  post-cutoff (60 dates): +0.0683% [-0.0679%, +0.2044%]; gap / mean|r| +0.0453
  pre - post: +0.0314% [-0.1463%, +0.2091%]
```

Each run writes `report.json` and `report.md` to the `--out` directory. Other options: `--seed`, `--n-tickers`, `--n-dates`, `--window`, `--cutoff`, `--bootstrap`.

The memorising mock recalls 75% of directions exactly. No real model is known to do this. The mock shows that the pipeline can detect a strong memoriser. It does not show how well the probe finds weak memory in a real model.

## How it works

Each episode is a (ticker, decision date) pair. The model answers LONG, SHORT, or ABSTAIN for the next day. The probe asks twice, once per arm.

**Masking.** The masked arm changes three things.

- Each ticker becomes a pseudonym, for example `ASSET_07`.
- All dates move back by a whole number of weeks, so each weekday stays the same.
- Each price is multiplied by a factor for that ticker, so the returns stay the same.

Before any model call, a leak check looks at every masked prompt. The prompt must contain no real ticker, no real date, and no real price of that episode. If the check finds a leak, the run stops. On a price collision, the probe draws a new factor for that ticker. It never relaxes the check.

**The as_of guard.** The observation builder needs an `as_of` date. If it is missing, the builder raises an error. The builder removes every row after the decision date, then asserts that none remain. A future row cannot reach a prompt.

**The paired gap.** The pnl of an episode is +r for LONG, -r for SHORT, and 0 for ABSTAIN. Here r is the real next-day return. The gap is the mean of (pnl_unmasked - pnl_masked) over all episodes.

**Intervals.** Episodes on the same date share the market move, so they are correlated. Each decision date is one observation. The CI for each gap is a Student t interval on the per-date mean differences, with G - 1 degrees of freedom (G is the number of dates). The CI for pre - post is a Welch interval, because the pre and post parts are different sets of dates. A percentile cluster bootstrap is also in the report as a secondary number. The verdict does not use it, because it covers only about 93.5-94.5% at 30 to 60 dates.

**Verdict rule.** This is the text of `VERDICT_RULE` in `src/llm_lookahead_probe/stats.py`:

> 'invalid run' if, in the whole run or in the pre-cutoff or post-cutoff part, either arm has more than 10% unparseable answers (refusals, empty or truncated replies count as ABSTAIN and would otherwise hide a broken run as a zero gap); else 'insufficient data' if the pre-cutoff or post-cutoff part has fewer than 20 decision dates; else 'memorisation suspected' if the lower CI bound of gap_pre > 0 AND the lower CI bound of (gap_pre - gap_post) > 0; otherwise 'no evidence'. gap = mean over episodes of (pnl_unmasked - pnl_masked). CIs are Student t intervals on the per-date mean differences with G - 1 df (G = decision dates); the pre - post CI is Welch's interval with Welch-Satterthwaite df. Every date must have the same number of episodes.

**Invalid run.** The probe counts an unparseable answer as ABSTAIN (pnl 0). If every answer failed, the gap would be 0 with a CI of [0, 0], and the verdict would be "no evidence". The invalid-run rule stops this. The probe also reports the number of unparseable answers per arm.

## Live mode (Claude, costs money)

Live mode is opt-in. It needs all of these:

- the `--live` flag
- the `ANTHROPIC_API_KEY` environment variable (the probe reads it from the environment and never prints it)
- the `--yes` flag
- a number of episodes that is not more than `--max-episodes` (default 50)

Without `--yes`, the command is a dry run. It prints the cost estimate, makes no call, and does not create the cache directory.

```bash
export ANTHROPIC_API_KEY=...
uv run llm-lookahead-probe run --live --prices my_prices.csv          # dry run: prints the estimate
uv run llm-lookahead-probe run --live --prices my_prices.csv --yes    # spends money
```

If the run needs more episodes than `--max-episodes`, it stops before any call. The provider also refuses calls beyond 2 times the cap.

**No live API call has been made by the authors.** The request shape and the cost have not been checked against the real API. The request shape comes from the SDK source (anthropic 2.3.0) and a stub server. The cost is a reasoned upper bound. It uses an assumed overhead of 64 tokens per request. `report.json` records the token usage that the API reports, so you can compare the real numbers with the estimate.

### Price CSV

The CSV must follow these rules. The loader refuses any file that breaks one of them.

- The header is `date,ticker,close`.
- Rows are sorted by (date, ticker), with no duplicates.
- A close is a plain decimal (`12.34` or `12`). It has no sign, spaces, exponent, or underscores. It is more than 0 and not more than 10,000,000.
- A date has the form `YYYY-MM-DD` and is on or after 1900-01-01.
- Every ticker has every date. The panel is rectangular.
- A ticker matches `[A-Z0-9][A-Z0-9.-]{0,9}`.
- The file is not more than 50 MB, 2,000,000 rows, and 1,000 tickers.

### Model and cutoff

The default model is `claude-haiku-5-5`. It costs $0.10 per MTok of input and $0.50 per MTok of output. Its training-data cutoff is Jun 2026. The probe uses 2026-06-01, so June counts as post-cutoff. The source is the Models Overview page of Anthropic (platform.claude.com). The notes in `src/llm_lookahead_probe/live.py` give the full list of sources. To use another cutoff, set `--cutoff`.

### Response cache

The probe caches responses in `.cache/llm-lookahead-probe/`. The key is the sha256 of the full request (model, prompt, `max_tokens`, thinking and effort settings). A rerun with the same settings makes no call and costs nothing. A changed setting makes new calls.

- The SDK retries are off (`max_retries=0`). Each HTTP request is a counted call.
- If a call fails, the run stops. Run it again, and the finished calls come from the cache.
- A new cache directory gets mode 0700. The probe refuses an existing directory that is group-writable or world-writable.
- The cache is not tamper-proof. Its stored key detects a copied or corrupt entry. It does not detect an edited entry. Anyone who can write to the cache directory controls the answers that the probe scores.

### Cost per run

Each episode makes 2 calls. The estimator counts at most one input token per UTF-8 byte. It adds 64 tokens per request for message framing, and the full 16-token output cap. Real prompts use about one token per 3 characters, so the bound is probably loose by 2 to 3 times. The 64-token overhead is an assumption.

The table shows the output of the dry run on a synthetic panel with a 20-close window. Your numbers change a little with your ticker names and price format.

| Episodes | Calls | Input tokens (bound) | Output tokens (bound) | Estimated upper bound |
|---|---|---|---|---|
| 50 (default cap, 5 tickers x 10 dates) | 100 | 65,360 | 1,600 | $0.0073 |
| 200 (5 tickers x 40 dates) | 400 | 259,178 | 6,400 | $0.0291 |
| 1,000 (10 tickers x 100 dates) | 2,000 | 1,294,783 | 32,000 | $0.1455 |
| 2,400 (20 tickers x 120 dates) | 4,800 | 3,110,506 | 76,800 | $0.3495 |

The default live setting (5 tickers x 10 dates) gives only 5 dates in each part. Its verdict is always "insufficient data". For a verdict, use at least 20 dates on each side of the cutoff. Raise `--max-episodes` to match.

## Limits

The probe cannot remove these effects.

- **A masking effect that is constant in behaviour.** Masking can change how an honest model behaves. For example, the model can abstain 10% more often when masked. The pre - post difference cancels only a masking effect that is constant in pnl. A behaviour effect costs pnl in proportion to |r|. A more volatile pre-cutoff period then gives pre - post > 0 with no memory. The report gives gap / mean|r| for each period. If the raw gaps differ but the scaled gaps agree, read the difference as volatility, not memory.
- **Non-string keys survive masking.** The probe keeps the exact return path, because a legitimate agent can use it. A whole-week date shift keeps the pattern of holiday gaps. A model can recognise an episode from either. Both effects bias the probe toward under-detection.
- **Masked dates are in the real past.** The probe shifts dates back by more than the panel span. They are not in a fictional calendar. A model can know the market of those dates too.
- **The panel must be rectangular.** Every ticker must have every date. The universe is survivor-only. Delisted tickers cannot be included.
- **Observed close.** The probe makes decisions at the observed close of the decision date and scores from it. It models no slippage and no costs.
- **Leak check scope.** The check looks only for the canonical forms of the renderer: a whole-token ticker, `YYYY-MM-DD`, and `d.dd`. A test pins every prompt line to the template grammar.
- **Mocks are not models.** The power of the memorising mock says nothing about real models. The probe has not measured power against partial memorisation.
- **Month-granular cutoff.** The cutoff is known only to the month, so the probe uses the first day of the month. Any memory of June then shrinks gap_pre - gap_post. This makes the verdict more conservative.

Version 0.1 supports only Claude, only `claude-haiku-5-5`, only a price path in the prompt (no news text), and no data fetchers.

## Validation

`uv run pytest -q` printed `147 passed in 20.76s` in this checkout.

The calibration test (`tests/test_calibration.py`) runs the full pipeline on the honest and memorising mocks. The panel has 8 tickers and 60 dates. The test printed these numbers:

| Measure | Result | Test limit |
|---|---|---|
| Gap CI coverage over 2,000 null seeds (nominal 95%) | 0.9470 | between 0.94 and 0.96 |
| Pre - post CI coverage over 2,000 null seeds | 0.9495 | between 0.94 and 0.96 |
| False "memorisation suspected" rate over 2,000 null seeds | 0.0095 | not more than 0.05 |
| Memorising mock detected in 500 seeds | 1.0000 | not less than 0.95 |

The earlier percentile bootstrap interval covered 0.938 and 0.9345 on the same kind of test. This is why the verdict uses the t interval.

An independent validation protocol checked the main claims. It also found problems that the later versions fixed or documented. See [docs/validation/report.md](docs/validation/report.md). The report lists what it did not validate: real models, real CSV data, a live API call, and the token cost estimate.

For the design, the alternatives that the authors rejected, and the failure modes, see [docs/DESIGN.md](docs/DESIGN.md).

```bash
uv run pytest -q
uv run ruff check
```

## Safety

This tool does analysis only. It has no order execution and no broker code. It never uses real money. The only cost is the Claude API fee of live mode, and you control it with `--yes` and `--max-episodes`.

## Related work

- [benstaf/lookaheadbench](https://github.com/benstaf/lookaheadbench) is a paper benchmark for look-ahead bias in LLMs.
- [TradingAgents](https://github.com/TauricResearch/TradingAgents) is a framework for LLM trading agents. You can use the probe to test an agent of this type for memorised history.

## Licence

MIT.
