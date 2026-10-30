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
  calibration test (200 seeds) must stay inside the normal `pytest` run.

## Non-goals (v0.1)

- Data fetchers (yfinance etc.), corporate actions, ragged panels (every ticker must have every date).
- More than one agent style or provider. Only Claude, only `claude-haiku-5-5` in the model table.
- Text/news context in prompts. Only a price path, so the only possible leaks are ticker, date, price.
- Proving memorisation. The probe gives evidence, not proof (see failure modes).

## Alternatives considered

| | A. Paired masking + date-cluster bootstrap (chosen) | B. Pre/post cutoff only, no masking | C. Direct recall quiz ("what did X close at on D?") |
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
`stats` (cluster bootstrap, verdict) → `report` (json + md). `live` holds the Claude provider,
the response cache and cost estimate; `mocks` holds the offline models.

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
panel-wide check would fail on noise.

Statistics. Episode = (ticker, decision date t); r = close(t+1)/close(t) − 1; pnl = +r, −r, 0.
gap = mean(pnl_unmasked − pnl_masked). Episodes on the same date share the market move, and a
model that recalls a date (not a ticker) moves every ticker on that date the same way, so episode
differences are correlated within a date. An i.i.d. bootstrap would treat 20 correlated episodes
as 20 independent ones and give a CI that is too narrow. We resample decision dates (clusters)
with replacement, 2,000+ times, seeded, percentile CI. The pre − post difference resamples pre and
post dates independently (they are disjoint clusters).

Verdict. "memorisation suspected" iff lower CI bound of gap_pre > 0 AND lower CI bound of
(gap_pre − gap_post) > 0. "insufficient data" if pre or post has fewer than 20 decision dates.
Otherwise "no evidence". Decision dates are sampled stratified, half before and half on/after the
cutoff, when the data allows.

## Failure modes

- **Null / missing as_of:** observation builder raises. Future row in an observation: raises at
  construction and again at prompt render.
- **Leak:** masked prompt containing a real ticker/date/price aborts the run before any API call.
- **Unparseable or refused answer:** becomes ABSTAIN (pnl 0), counted per arm in the report.
- **Retried / rerun:** responses cached by sha256(model, prompt); a rerun makes zero calls. Cache
  entries store their own key and are ignored if the key does not match (corrupt or swapped file).
- **Partial run (crash mid-live-run):** completed calls are already cached; rerun resumes for free.
- **Malicious CSV:** strict schema, ticker charset `[A-Z0-9.-]{1,10}` (blocks prompt-injection text
  in the ticker field), size limits, no NaN/inf/≤0, strictly increasing (date, ticker), rectangular.
- **Overspend:** `--live` + key + `--yes` required; planned episodes > `--max-episodes` refuses; the
  provider also refuses calls beyond 2 × cap (defence in depth); estimate printed before any call.
- **100× scale:** offline cost is linear in episodes (~µs each) and the bootstrap is a (B × dates)
  matrix product; live cost is linear in episodes and is the binding limit.
- **Confounds the probe cannot remove:** masking may change behaviour of an honest model (dates
  decades back, unfamiliar ticker names). The pre − post difference controls for a constant
  masking effect; it does not control for an effect that differs before vs after the cutoff.
  The training cutoff is month-granular ("Jun 2026"); we use 2026-06-01, so June episodes count
  as post-cutoff. Any June memory then shrinks gap_pre − gap_post, which makes the verdict more
  conservative, not less.

## Cost (order of magnitude)

Live, `claude-haiku-5-5` at $0.10 / $0.50 per MTok: one prompt with 20 closes is ~600 characters
(~200–300 tokens); output is capped at 16 tokens. 1,000 episodes × 2 arms ≈ 2,000 calls ≈ $0.07
by the conservative estimator in `live.py`. Reruns cost $0 (cache). This is an estimate, not a
measured bill; real usage is recorded in `report.json`.

## Increments

v0.1 (this): offline mocks, Claude Haiku live, CSV input. v0.2: more models in the table (needs
per-model thinking config verified against the API), ragged panels, a recall quiz (option C) as a
diagnostic. Nothing here is irreversible: outputs are files, caches are deletable.
