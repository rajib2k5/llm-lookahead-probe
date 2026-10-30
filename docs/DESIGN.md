# llm-lookahead-probe v0.1: design

## Problem

An LLM trading agent backtested on history it was trained on can "predict" moves it simply remembers.
The probe measures that inflation: run the same decision prompts twice, once with real identifiers
(unmasked) and once with identifiers the model cannot look up (masked), and report the paired
performance gap with a confidence interval. The masked arm keeps the information a legitimate agent
may use (the past return path) and removes the keys a memorising model needs (ticker, date, price
level).

## Constraints

- Analysis only. No orders, no broker code, no real money.
- Offline by default, deterministic from one seed, no network and no API key needed for the test suite.
- Live mode spends money: opt-in, capped, priced before the first call, cached so reruns cost nothing.
- No vendor data in the repository. Users bring a CSV for live mode.
- A future row must never reach a prompt (as_of discipline, same idea as `queryFacts` in
  quant-research-copilot: no as_of means refuse; rows with date > as_of are filtered, then asserted absent).
- Implied but not stated: a run of ~2,000 episodes must finish in seconds offline, and the null
  calibration test (2,000 seeds, run in worker processes) must stay inside the normal `pytest` run.

## Non-goals (v0.1)

- Data fetchers (yfinance etc.), corporate actions, ragged panels (every ticker must have every date).
- More than one agent style or provider. Only Claude, only `claude-haiku-5-5` in the model table.
- Text/news context in prompts. Only a price path, so the only possible leaks are ticker, date, price.
- Proving memorisation. The probe gives evidence, not proof (see failure modes).

## Alternatives considered

| | A. Paired masking + date-level inference (chosen) | B. Pre/post cutoff only, no masking | C. Direct recall quiz ("what did X close at on D?") |
|---|---|---|---|
| Idea | Same episodes, two arms; gap = mean paired pnl difference; split by cutoff | Compare agent pnl before vs after training cutoff | Ask the model for historical prices and score recall |
| Fails when | Masking leaks (mitigated: leak check on every masked prompt, fail-closed) or masking changes behaviour for reasons other than memory (e.g. odd dates) | Market regime differs pre/post: any pnl difference is confounded with regime | Model recalls facts but does not use them in decisions, or uses them without being able to state them |
| Measures | Inflation of the actual decision pipeline | Confounded difference | Knowledge, not backtest inflation |
| Cost (live) | 2 calls per episode | 1 call per episode | 1 call per fact |

Recommendation: **A**, with B's cutoff split layered on top. Pairing removes the regime confound
(both arms see the same episode and are scored on the same realised return), and the cutoff split
adds a second, independent prediction: memorisation should create a gap before the cutoff and none
after. C is a useful later add-on for diagnosis but does not answer "is my backtest inflated".
What would change the decision: if masked prompts measurably change a non-memorising model's
behaviour (gap_post far from 0 on real models), the masking itself is biased and the post-cutoff
gap must be used as the baseline instead of zero (the difference test already does this).

## Design

Seams (one module each): `data` (panel, synthetic generator, strict CSV loader) → `pit`
(as_of-guarded observation builder) → `masking` (pseudonyms, date shift, price rescale, leak check)
→ `prompt` + `agent` (render, `Model.complete`, parse) → `probe` (episodes, both arms, pnl) →
`stats` (t-intervals on date means, verdict; cluster bootstrap as a secondary number) →
`report` (json + md). `live` holds the Claude provider, the response cache and cost estimate;
`mocks` holds the offline models.

Masking. Ticker → `ASSET_nn` from a seeded permutation of the sorted ticker list (the number does
not depend on the ticker text). Dates → one global shift by a whole number of weeks (weekday kept),
backwards by more than the panel span, so no masked date can equal a real panel date. Prices →
real × per-ticker factor drawn log-uniform in [1.5, 6]; returns are unchanged and, because the
factor is ≥ 1.5, the price levels of a window cannot overlap the real window unless the window
moves more than 50%. Up-scaling only, so rounding to cents loses less precision than the real
data already has. Leak check (runs on every masked prompt before any model call): no real ticker
as a whole token, no real panel date as `YYYY-MM-DD`, no real price of that episode (window and
next close) as a `d.dd` token. The price check is per episode on purpose: compared against every
price in a panel, a random cents value collides with some real price roughly half the time, so a
panel-wide check would fail on noise. Even per episode, a volatile low-priced series collides by
coincidence (measured: 122 of 250 masker seeds on a $2 series with 25% daily volatility). On a
price collision the planner redraws that ticker's factor from a seed derived from (seed, ticker
position, attempt), re-plans all of that ticker's episodes, and gives up after 20 redraws by
raising the leak error. The check itself is never relaxed. Ticker and date leaks are never retried.
The leak check recognises only the renderer's canonical forms; a test pins every prompt line to
the template grammar, so the renderer cannot emit another form unnoticed.

Statistics. Episode = (ticker, decision date t); r = close(t+1)/close(t) − 1; pnl = +r, −r, 0.
The decision is made at, and scored from, the observed close of t. gap = mean(pnl_unmasked −
pnl_masked). Episodes on the same date share the market move, and a model that recalls a date
(not a ticker) moves every ticker on that date the same way, so episode differences are correlated
within a date. Each date is therefore one observation: its mean paired difference. The CI is a
Student t interval on the G date means with G − 1 df; the pre − post CI is Welch's interval with
Welch–Satterthwaite df (pre and post are disjoint date sets with possibly different variances).
The t quantile is computed without scipy (incomplete beta, bisection) and tested against printed
tables.

The mean of the date means equals the episode mean only if every date has the same number of
episodes. The probe crosses a fixed ticker sample with the sampled dates of a rectangular panel,
so this holds; `summarise` refuses unequal counts instead of silently re-weighting.

Why not the v0.1 interval. v0.1 used a percentile cluster bootstrap (resample dates, 2,000+
resamples). Measured over 2,000 null seeds it covered 0.938 (gap, 8 tickers × 60 dates) and
0.9345 (pre − post, 30 + 30 dates) when labelled 95%, the known small-sample under-coverage of
percentile intervals with few clusters (Hesterberg, arXiv:1411.5279; Cameron, Gelbach & Miller).
The t-interval on the same seeds covers 0.9470 and 0.9495. The bootstrap is still computed and
reported under `robustness` as a secondary number. A bootstrap-t or BCa interval would also fix
the coverage, but needs resampling for a gain the t-interval already gives at no cost. What would
make the bootstrap win: strongly skewed date means at small G, where the t-interval's symmetry is
wrong; the calibration test would show it.

Verdict. "invalid run" if, in the whole run or in the pre or post part, either arm has more than
10% unparseable answers (refusals, empty or truncated replies). These count as ABSTAIN (pnl 0),
so without this rule a run where every answer failed would show gap 0, CI [0, 0] and "no
evidence". Also "invalid run" if, in the whole run or in the pre or post part, either arm
answers ABSTAIN (unparseable answers included) on more than 50% of episodes: too few trades. An
arm that does not trade earns pnl 0, so an always-ABSTAIN model (or one that abstains only when
masked) would otherwise get gap 0 and "no evidence", or even a false "suspected" (see Limits).
Else "insufficient data" if pre or post has fewer than 20 decision dates. Else
"memorisation suspected" iff lower CI bound of gap_pre > 0 AND lower CI bound of
(gap_pre − gap_post) > 0. Otherwise "no evidence". Decision dates are sampled stratified, half
before and half on/after the cutoff, when the data allows. hit_rate = hits / trades; a trade is
LONG or SHORT; a trade on r = 0 is a miss; ABSTAIN is not a trade.

Measured on the offline mocks (8 tickers × 60 dates, 2,000 null seeds, 500 power seeds,
`tests/test_calibration.py`): gap coverage 0.9470, pre − post coverage 0.9495, false
"memorisation suspected" rate 0.0095, power against the memorising mock 1.0000. The mock's power
says nothing about real models: the memorising mock recalls 75% of directions exactly, which no
real model is known to do.

## Failure modes

- **Null / missing as_of:** observation builder raises. Future row in an observation: raises at
  construction and again at prompt render.
- **Leak:** masked prompt containing a real ticker/date/price aborts the run before any API call.
- **Unparseable or refused answer:** becomes ABSTAIN (pnl 0), counted per arm in the report;
  above 10% in either arm (whole run, pre or post) the verdict is "invalid run".
- **Model that does not trade:** above 50% ABSTAIN (unparseable included) in either arm (whole
  run, pre or post) the verdict is "invalid run" (too few trades).
- **Non-verdict outcomes exit non-zero:** the CLI exits 3 for "invalid run" and 4 for
  "insufficient data" (both still write the reports), so a script cannot mistake them for a
  result. 2 stays for refused input, 1 for unexpected failures.
- **Retried / rerun:** responses cached by sha256 of the full request (model, prompt, max_tokens,
  thinking, effort); a rerun with the same settings makes zero calls, and a changed setting is a
  miss, not a stale answer. Entries store their own key and are ignored if it does not match:
  this detects a copied, swapped or corrupt file. It does NOT detect tampering: the key is a hash,
  not a MAC, and an edited answer with its key intact is served. Anyone with write access to the
  cache directory controls the answers. The cache directory is created 0700; an existing one
  keeps its mode and is refused if group- or world-writable. A dry run does not create it.
- **SDK retries:** set to 0. Each HTTP request is then a counted call against the budget; a
  transient failure (429, 529, 5xx, connection error) stops the run with exit 5 and a message
  that a rerun resumes from the cache (no report is written for a partial run; the key is
  redacted). Every answer is written to the cache as it arrives, so nothing needs flushing.
- **Cache directory not ours / wrong endpoint:** an existing cache directory owned by another
  uid is refused (e.g. one pre-created under /tmp at 0755). The cache key covers the API base URL
  (`ANTHROPIC_BASE_URL` or the public default), and the client is built with that same URL, so
  answers from a local stub server are never served on a real run.
- **Partial run (crash mid-live-run):** completed calls are already cached; rerun resumes for free.
- **Malicious CSV:** strict schema, ticker charset `[A-Z0-9.-]{1,10}` (blocks prompt-injection text
  in the ticker field), size limits, close as plain decimal text (no sign, spaces, exponent,
  underscores, nan or inf), 0.01 ≤ close ≤ 1e7 (masking multiplies by ≤ 6, so no overflow; a
  sub-cent close would render as "0.00" and abort as a false price leak), dates on or after
  1900-01-01 (a sanity bound only), strictly increasing (date, ticker), rectangular. A panel whose
  span would push the backward date shift below year 1 (e.g. a 9999-12-31 typo in a 1900 panel)
  is refused by `Masker.fit` with a data error. Commit 6367253 claimed MIN_YEAR alone prevented
  this underflow; it did not (a ~1,900-year span still overflowed).
- **Overspend:** `--live` + key + `--yes` required; planned episodes > `--max-episodes` refuses; the
  provider also refuses calls beyond 2 × cap (defence in depth); estimate printed before any call.
- **100× scale:** offline cost is linear in episodes (~µs each) and the bootstrap is a (B × dates)
  matrix product; live cost is linear in episodes and is the binding limit.

## Limits (what the probe cannot remove)

- **Masking effect that is constant in behaviour.** Masking may change an honest model's
  behaviour (dates years back, unfamiliar ticker names). pre − post cancels a masking effect only
  if it is constant in pnl. An effect constant in behaviour (e.g. abstaining 10% more often when
  masked) costs pnl in proportion to |r|, so a more volatile pre-cutoff period gives
  pre − post > 0 with no memory. The report gives gap / mean|r| for each period; if the raw gaps
  differ but the scaled gaps agree, read the difference as volatility, not memory.
- **Non-string keys survive masking.** The exact return path is kept (it is what a legitimate
  agent may use), and a whole-week date shift keeps the pattern of holiday gaps. A model could
  recognise an episode from either. Both bias toward under-detection.
- **Masked dates are in the real past** (shifted back by more than the panel span), not in a
  fictional calendar. A model may know the market of those masked dates too.
- **Rectangular panel only.** Every ticker must have every date, so the universe is
  survivor-only (delisted tickers cannot be included).
- **Observed close.** Decisions are made at and scored from the observed close of the decision
  date; no slippage, no costs.
- **Leak check scope.** Only the renderer's canonical forms are checked (whole-token ticker,
  `YYYY-MM-DD`, `d.dd`). A test pins every prompt line to the template grammar.
- **Mocks are not models.** The memorising mock's power of 1.0 shows the pipeline can detect a
  strong, exact memoriser; it says nothing about the power against real models, and power
  against partial memorisation has not been measured.
- **Abstaining is a behaviour, and the gate is coarse.** The trade-count gate (50% ABSTAIN per
  arm and part) catches a model that mostly does not trade; it does not catch a smaller change
  in abstaining under masking. The report gives the abstain rate of each arm and the
  masked − unmasked difference per part, and flags a difference above 10 percentage points. Such
  a change moves pnl in proportion to |r| and can produce a gap alone: in
  `tests/test_stats.py`, a masked arm that abstains on every pre-cutoff episode (unmasked always
  LONG, r > 0) gave "memorisation suspected" before the gate existed. Why 50%: at that rate half
  the episodes cannot express a direction, so "no evidence" would mostly describe non-trading.
  Measured with the memorising mock (8 tickers × 60 dates, 300 seeds, both arms abstaining at
  random on a fraction q of prompts): detected in 1.000 / 0.987 / 0.940 of seeds at q = 0 / 0.25 /
  0.40; at q = 0.50, 0.950 of seeds were "invalid run" (the per-part rate exceeds 50% by sampling
  noise) and at q ≥ 0.60 all were. A stricter limit (e.g. 25%) would reject cautious but usable
  models; a looser one (e.g. 75%) would report verdicts where power is unknown. What would change
  this: a measured power curve against a weak memoriser, which might justify a lower limit.
- **Month-granular cutoff.** The training cutoff is month-granular ("Jun 2026"); we use 2026-06-01, so June episodes count
  as post-cutoff. Any June memory then shrinks gap_pre − gap_post, which makes the verdict more
  conservative, not less.

## Cost (order of magnitude)

Live, `claude-haiku-5-5` at $0.10 / $0.50 per MTok: one prompt with 20 closes is ~584 bytes;
output is capped at 16 tokens. The estimator in `live.py` bounds input tokens by UTF-8 bytes
(a byte-level BPE token covers at least one byte) plus an unverified 64-token allowance per
request, output by `max_tokens`, and multiplies by (1 + max_retries) = 1. Its printed output
(dry run, synthetic panel, 10 tickers × 100 dates, cutoff 2025-01-02): 1,000 episodes × 2 arms =
2,000 calls, at most $0.145478. (An earlier version of this section said $0.1463.) Reruns cost $0 (cache). This is a bound, not a
measured bill; real usage is recorded in `report.json`.

## Increments

v0.1 (this): offline mocks, Claude Haiku live, CSV input. v0.2: more models in the table (needs
per-model thinking config verified against the API), ragged panels, a recall quiz (option C) as a
diagnostic. Nothing here is irreversible: outputs are files, caches are deletable.
