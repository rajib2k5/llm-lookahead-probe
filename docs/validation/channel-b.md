# Channel B: runtime audit

- Interpreter: `/home/rajib2k5/dev/llm-lookahead-probe/.venv/bin/python` (Python 3.12.13, numpy 2.5.4,
  anthropic 2.3.0). See `env-pin.txt`.
- Probes: `docs/validation/probes/`. Run from that directory. Falsifiers: `./run_mutant.sh <mutation> <probe>`
  (copies `src/` to a temp dir, applies one mutation from `mutations.py`, asserts the mutant module is the
  one imported, runs the probe, re-hashes the repo `src/`). Every falsifier run printed
  `src checksum unchanged`.
- Sandbox: no network needed by any probe (C15 uses a 127.0.0.1 stub server and a fake key). The probes
  were not run under the `unshare` wrapper. They run as the user, with `-B` and `PYTHONDONTWRITEBYTECODE=1`
  for mutants.
- Tier 0: `.venv/bin/python -m pytest -q -p no:cacheprovider` printed `99 passed in 23.05s`.
  `.venv/bin/ruff check src` printed `All checks passed!`.

## Independence disclosure (read this first)

The plan was Mode 1: Channel A in a subagent, Channel B in the orchestrator, neither seeing the other.
The Channel A subagent finished and its hand-back summary was delivered into the orchestrator context
**after** the ledger was frozen and the mutations and runner were written, but **before** the probe
scripts were written and run. The probe commands, pass conditions and falsifier mutations were
pre-registered in `claims-ledger.md` before any documentation was read, and the probes implement them.
However, the probe scripts were written with Channel A's verdicts known. Channel A advised a numeric
tolerance for the C3 permutation check, and the probe uses `rtol=1e-9`. **Treat the result as Mode 2:
"dual-confirmed (fallback mode)".**

Deviations from the pre-registered rows:
- C5: the pre-registered probe used N = 1000 reps. A follow-up (`p05b_diff_coverage.py`, N = 4000,
  added after the first result) is reported separately as split C5.2.
- C14: the ledger did not fix the mock's strength. The probe uses `recall_p = 0.3` with
  `mock_cutoff = 2100-01-01`.
- C6, C12: the ledger said ">= 1000 seeds". The probes use 2000 (SMALL) and 2000 or 1000 (DEFAULT).

## Per-claim records

### C1 pnl sign and r
- `probe_cmd`: `.venv/bin/python -B p01_pnl.py` → rc 0
- stdout: `r(t=1/2) 0.10000000000000009 expected 0.10000000000000009 | r(t=1/3) -0.09999999999999998 expected -0.09999999999999998` / `fails: []`
- `falsifier`: `./run_mutant.sh pnl_long_sign p01_pnl.py` → `fails: [(LONG, 0.0123, -0.0123, 0.0123), (LONG, -0.045, 0.045, -0.045)]`, falsifier_rc=1
- verdict: **CONFIRMED**

### C2 episode-weighted gap estimate
- `p02_gap_estimate.py` → rc 0. Data: 50 dates, 1–9 episodes per date (unbalanced).
- stdout: `all: code=-0.000436036401199 episode_mean=-0.000436036401199 date_mean=-0.0004112119628398188`;
  `pre: code=0.000838718059121 episode_mean=0.000838718059121`;
  `post: code=-0.00256986451956 episode_mean=-0.00256986451956`; `diff: code=0.00340858257868 episode_mean=0.00340858257868`
- falsifier `estimate_date_mean` → falsifier_rc=1
- verdict: **CONFIRMED**

### C3 whole-date resampling, C4 episode-weighted ratio replicate
- `p03_cluster_bootstrap.py` → rc 0. Data: 40 clusters, sizes 1–11, cluster mean correlated with size.
- stdout:
  `C3a original CI(estimate=3.014760508961123, low=2.513192343068889, high=3.465129937122905)` /
  `C3a permuted CI(... low=2.5131923430688894, high=3.465129937122905)` (difference 4e-16) /
  `ref episode-weighted q: [2.50551973 3.47625053]  ref date-mean q: [1.86441361 2.86196031]  code: (2.513192343068889, 3.465129937122905)` /
  `date-mean vs episode-weighted ref separation (in sd): [2.59 2.48]` / `fails: []`
- falsifiers:
  - `iid_bootstrap` → `fails: ['C3a permutation changed CI', 'C3b/C4 code quantiles differ from episode-weighted reference']`, rc=1
  - `ratio_to_date_mean` → `fails: ['C3b/C4 ...']`, rc=1
- verdict: C3 **CONFIRMED**, C4 **CONFIRMED**

### C5 pre − post CI
- Pre-registered `p05_diff_ci.py` → **rc 1**.
  - stdout: `ref quantiles [-0.00766701 0.00835788] width ratio 0.967358614589632` /
    `null coverage of diff CI over 1000 reps (G=30+30): 0.925 (se 0.008)` / `fails: ['coverage']`
  - falsifier `diff_ignores_post_var` → `width ratio 0.569...`, `coverage 0.608`, rc=1
  - Width part (procedure = independent stratified bootstrap): passes. Coverage part: fails the
    pre-registered band [0.93, 0.97].
- Follow-up `p05b_diff_coverage.py G 4000`, same generator, seeds 50000+:
  - `G=20+20 N=4000 percentile (code): coverage 0.9377 se 0.0038` / `Welch t on date means: coverage 0.9537 se 0.0033`
  - `G=30+30 N=4000 percentile (code): coverage 0.9397 se 0.0038` / `Welch t on date means: coverage 0.9505 se 0.0034`
  - `G=60+60 N=4000 percentile (code): coverage 0.9405 se 0.0037` / `Welch t on date means: coverage 0.9455 se 0.0036`
- verdict: split.
  - C5.1 (the procedure is the independent two-stratum cluster bootstrap; its width matches the reference
    within 10%): **CONFIRMED**.
  - C5.2 (null coverage in [0.93, 0.97]): the pre-registered N = 1000 run gave **REFUTED** (0.925). The
    N = 4000 follow-up gives 0.938–0.941, inside the band but about 1 point under nominal (2.7 se at
    G = 30). Welch t on date means is at nominal. Channel B verdict on C5.2: **REFUTED as pre-registered**.
    The larger run shows a mild, systematic undercoverage of about 1 point, not a broken procedure.

### C6 null coverage of the 95% CI
- `p06_coverage.py` → rc 0 (pre-registered rule: Wilson upper ≥ 0.935 on both configs). Wall time 4m28s.

| config, 2000 seeds | gap (all) | pre | post | pre−post | t on date means: all | pre | post |
|---|---|---|---|---|---|---|---|
| SMALL 8×60 | 0.9380 [0.9266, 0.9478] | 0.9320 [0.9201, 0.9422] | 0.9355 | 0.9345 | 0.9470 [0.9363, 0.9560] | 0.9495 | 0.9525 |
| DEFAULT 20×120 | 0.9470 [0.9363, 0.9560] | 0.9555 | 0.9450 | 0.9490 | 0.9535 | 0.9655 | 0.9510 |

- Seeds 0–199: SMALL `cov_all 0.935` (reproduces the builder's 0.935); DEFAULT `0.945`.
- falsifier `ci_level_shrunk` → SMALL `cov_all 0.8425`, DEFAULT `0.8540`, `REFUTED`, rc=1.
- verdict: **CONFIRMED** under the pre-registered rule. The observation: on SMALL the percentile CI
  undercovers by about 1.2 points, and the Wilson upper bound 0.9478 excludes 0.95. The t-interval on
  date means (G−1 df), computed from the same records, is at nominal on both configs.

### C7 honest null is exactly 0
- `p07_honest_null.py` → rc 0
- stdout:
  - `DEFAULT: base-action mismatches 0 of 2400000 episodes; mean gap -1.710e-05 sd 4.508e-04 t=-1.20; share of seeds with est>0: 0.485` / `DEFAULT: seed0 est 0.000839`
  - `SMALL: base-action mismatches 0 of 480000 episodes; mean gap 5.704e-05 sd 9.813e-04 t=1.84; share of seeds with est>0: 0.529`
- falsifier `mock_price_threshold` → `base-action mismatches 66136 of 2400000`, `13100 of 480000`, rc=1
  (note: the t-test alone did not catch this mutant, t = −1.94 and 0.99; the mismatch count did).
- The seed-0 default gap 0.000839 is 1.86 sd of the across-seed distribution (sd 4.5e-4). It is noise,
  not bias.
- verdict: **CONFIRMED**

### C8 masking preserves returns, weekdays, bijection, shift
- `p08_masking.py` → rc 0 over 30 seeds of the DEFAULT config
- stdout: `returns checked 1368000; max |dr| 5.744e-04; max |dr|/bound 0.999` / `fails (first 5): [] count 0`
- falsifier `factor_per_bar` → `count 72000`, rc=1
- verdict: **CONFIRMED**. Rounding moves a within-window return by up to 5.7e-4 (5.7 bp) on low-priced
  synthetic tickers.

### C9 leak check
- `p09_leak.py` → rc 0. Panel tickers: `A, AAPL, BF-B, BRK.B, F, X1`. Every canonical case raised
  `LeakError`:
  - ticker contexts: `hold in T from`, `T.`, `(T)`, `T,`, a ticker on its own line, `T's`, `$T`, `T:`
  - ISO dates, also with `.` and with letters around them
  - `d.dd` prices, also as `$p`, `(p)` and `p.`
  - clean masked prompts never raised
- Non-canonical forms that escape (findings, not verdicts): `aapl`, `AAPL-like`, `_AAPL_`, `2024/01/04`,
  `20240104`, `01/04/2024`, `Jan 04, 2024`, a price at 3 dp `51.780`, a price at 1 dp `51.8`.
- falsifier `ticker_lookahead_loose` → rc=1
- verdict: **CONFIRMED** (canonical forms, the only forms the renderer emits)

### C10 fail-closed before model calls
- `p10_failclosed.py` → rc 0
- stdout: `LeakError: masked prompt contains real ticker 'KKS'` / `LeakError (price-only leak): masked prompt contains real price 21.64` / `raised: True True model calls: 0`
- falsifier `skip_leak_check` → `raised: False False model calls: 1920`, rc=1
- verdict: **CONFIRMED** (offline path; `cli.py` line 171 also calls `plan_episodes` before `execute` at line 190, read only, not probed)

### C11 as_of guard
- `p11_asof.py` → rc 0
- stdout:
  - `episodes checked 24000`
  - construction with a future bar raises `AsOfViolationError future row 2024-05-22 > as_of 2024-05-21`
  - a future bar placed first also raises
  - rendering an Observation mutated after construction raises
  - `None`, `""` and the string `"2024-05-01"` as as_of all raise
  - max date in every unmasked prompt == t; in every masked prompt == t + shift; the next date never appears
- falsifier `asof_checks_noop` → `fails: ['future bar (not last) accepted', 'render accepted future bar']`, rc=1
- verdict: **CONFIRMED**

### C12 false-positive rate of the verdict
- `p12_fp_power.py fp` → rc 0
- stdout:
  - `SMALL: false 'suspected' 29/2000 = 0.0145 Wilson [0.0101,0.0207]; seeds 0-199: 0.020; P(pre low>0)=0.0400; P(diff low>0)=0.0365`
  - `DEFAULT: false 'suspected' 5/1000 = 0.0050 Wilson [0.0021,0.0117]; seeds 0-199: 0.005; P(pre low>0)=0.0180; P(diff low>0)=0.0190`
- falsifier `verdict_or` → SMALL `124/2000 = 0.0620 Wilson [0.0522,0.0734]`, rc=1
- verdict: **CONFIRMED**. 0.020 on seeds 0–199 reproduced. Note: on SMALL, the one-sided rejection rates
  are 0.040 (pre) and 0.0365 (diff) against a nominal 0.025. This is the same percentile undercoverage
  seen in C6, concentrated in one tail.

### C13 power
- `p12_fp_power.py power` → rc 0
- stdout: `power p=0.75 SMALL seeds 0-199: 1.000` / `power p=0.55 ...: 0.975` / `power p=0.6 ...: 0.995`
- falsifier `mock_no_recall` → `0.020` at every p, rc=1
- verdict: **CONFIRMED**. Even p = 0.55 is detected at 0.975, so this mock is a very easy target. Power
  against weak or partial memorisation on real models is not established.

### C14 constant masking effect
- `p14_masking_effect.py` → rc 0. Mock: memorising, `mock_cutoff=2100-01-01` (recall everywhere), `recall_p=0.3`.
- stdout: `SMALL: gap_pre CI low>0 in 0.854 of seeds; false 'suspected' 27/1000 = 0.0270 Wilson [0.0186,0.0390]` / `DEFAULT: gap_pre CI low>0 in 1.000 of seeds; false 'suspected' 13/400 = 0.0325 Wilson [0.0191,0.0548]`
- falsifier `verdict_drop_diff` → `854/1000`, `400/400`, rc=1
- verdict: **CONFIRMED** for an effect that is constant in pnl (same return distribution before and after
  the cutoff). An effect that is constant in behaviour, applied to periods with different volatility, was
  not probed.

### C15 SDK request shape
- `p15_sdk_request.py` → rc 0
- stdout: `create has thinking: True | has temperature: False | VAR_KEYWORD: False` / `POST /v1/messages body: {"max_tokens": 16, "messages": [{"role": "user", "content": "prompt text"}], "model": "claude-haiku-5-5", "thinking": {"type": "disabled"}}` / `typo kwarg rejected: Messages.create() got an unexpected keyword argument 'thinkng'`
- falsifier `thinking_enabled_typo` → `fails: [('thinking', {'type': 'disable'})]`, rc=1. Side observation:
  the SDK sends a malformed `{"type": "disable"}` without client-side validation, so typos inside the
  dict are only caught by the server.
- verdict: **CONFIRMED**

### C16 SDK response fields
- same probe: `returned text: LONG | usage: 123 2 {'end_turn': 1}`
- falsifier `usage_field_rename` → `AttributeError("'Usage' object has no attribute 'prompt_tokens'")`, rc=1
- verdict: **CONFIRMED**

### C17 model ID / pricing / cutoff / thinking acceptance
- `p17_model_id.py` → rc 0: `literal count 20 | haiku models: ['claude-haiku-4-5', 'claude-haiku-4-5-20251001', 'claude-haiku-5-5']` / `claude-haiku-5-5 in list: True | control claude-haiku-5-55 in list: False`
- Price, training cutoff and server acceptance of disabled thinking cannot be probed without a paid live
  call. That is out of scope: no key, and it spends money.
- verdict: **UNVERIFIABLE** on the whole claim. Split: C17.1 (SDK knows the ID) **CONFIRMED**; C17.2
  (price, cutoff, server-side acceptance) N/A-by-construction for Channel B.

### C18 (minor): not probed.
