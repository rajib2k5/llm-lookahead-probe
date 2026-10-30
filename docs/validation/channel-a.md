# Channel A: documentation audit

- Auditor: Channel A subagent (docs only). No project code was executed, no probe was written or run,
  nothing under `docs/validation/probes/` and no `channel-b.md` was read.
- Target: `src/llm_lookahead_probe/` at HEAD `664301b` (hashes in `env-pin.txt`), intent doc `docs/DESIGN.md`.
- Pinned versions audited against: Python 3.12.13, numpy 2.5.4 (`.venv/.../numpy/version.py`: `version = "2.5.4"`),
  anthropic 2.3.0 (`.venv/.../anthropic/_version.py`: `__version__ = "2.3.0"`).
- Retrieved: 2026-10-29 (local date of this session; the ledger's freeze timestamp is 2026-10-30T04:00Z).
- Source tiers: 1 = official versioned docs (docs.python.org/3.12, numpy.org/doc/2.5, platform.claude.com,
  the bundled `claude-api` skill reference cached 2026-10-06); 2 = installed library source;
  4 = original paper. "code" = the module under audit read as its own contract.
- Pen-and-paper arguments are stated in full in the note field where a verdict rests on them.
  One arithmetic calculation (Student-t tail areas, pure `math`, no project code) was run with
  `python3 -I -c` to tabulate normal-theory coverage; it is labelled as such where used.

## Sources that could not be retrieved

- Efron & Tibshirani (1993), *An Introduction to the Bootstrap*: book, no accessible primary text. Not cited.
- Davison & Hinkley (1997), *Bootstrap Methods and their Application*: book, no accessible primary text. Not cited.
- Field & Welsh (2007), "Bootstrapping clustered data", JRSS-B 69(3): full text restricted (ANU repository embargo,
  publisher paywall). Not cited.
- The published RESTAT version of Cameron, Gelbach & Miller (2008) is paywalled; the NBER Technical Working Paper 344
  (Sept 2007) version by the same authors was used instead (https://www.nber.org/papers/t0344.pdf).
- Substitutes used: Hesterberg, "What Teachers Should Know about the Bootstrap", arXiv:1411.5279v1 (2014; published in
  The American Statistician 69(4), 2015) for percentile-interval narrowness and the two-sample bootstrap; Cameron & Miller,
  "A Practitioner's Guide to Cluster-Robust Inference", J. Human Resources 50(2), 2015
  (https://cameron.econ.ucdavis.edu/research/Cameron_Miller_JHR_2015_February.pdf) for pairs cluster bootstrap mechanics
  and T(G-1) critical values.
- No primary source was retrieved specifically for "resample the ratio of cluster totals" (the Davison & Hinkley ratio
  example). C4 is decided on the code and numpy semantics, not on that literature.

---

## Evidence records

```
claim_id:    C1
verdict:     CONFIRMED
source:      src/llm_lookahead_probe/stats.py:55-56; src/llm_lookahead_probe/data.py:72-77; src/llm_lookahead_probe/probe.py:91-95
source_tier: code
version:     HEAD 664301b (stats.py sha256 286591fa..., data.py sha256 447fca68...)
quote:       "return r if action == Action.LONG else -r if action == Action.SHORT else 0.0"
             "return float(self.closes[i + 1, j] / self.closes[i, j] - 1.0)"
retrieved:   2026-10-29
note:        pnl is +r / -r / 0.0 by action. r is PricePanel.next_day_return, i.e. close(i+1)/close(i) - 1 (simple, forward one row),
             and sample_episodes builds Episode.r with exactly panel.next_day_return(t, panel.dates[i]); stats reads e.r unchanged.
             No log return, no backward return. No library behaviour is involved beyond float division.
```

```
claim_id:    C2
verdict:     CONFIRMED
source:      src/llm_lookahead_probe/stats.py:116-119, 129-132, 142-146
source_tier: code
version:     HEAD 664301b
quote:       "d = np.array([pnl(e.action_u, e.r) - pnl(e.action_m, e.r) for e in recs])"
             "gap = CI(float(d_all.mean()), lo, hi)"  /  "part[\"gap\"] = CI(float(d.mean()), plo, phi).as_dict(level)"
retrieved:   2026-10-29
note:        The all/pre/post estimates are ndarray.mean() over one entry per episode record, i.e. episode-weighted. Pre/post
             partition is e.date < cutoff vs >= cutoff. No per-date averaging is applied to the point estimate.
```

```
claim_id:    C3
verdict:     CONFIRMED
source:      src/llm_lookahead_probe/stats.py:59-71, 86-91, 129-131, 142-144;
             numpy 2.5 Generator.multinomial https://numpy.org/doc/2.5/reference/random/generated/numpy.random.Generator.multinomial.html ;
             installed numpy 2.5.4 _core/multiarray.py (bincount docstring) and lib/_arraysetops_impl.py (unique docstring);
             Cameron & Miller (2015) JHR, Section II.F
source_tier: code + 1 (numpy 2.5 docs) + 2 (installed numpy source) + 4 (paper, for the definition of a pairs cluster bootstrap)
version:     numpy 2.5.4 / "NumPy v2.5 Manual"
quote:       numpy multinomial: "Its values, X_i = [X_0, X_1, ..., X_p], represent the number of times the outcome was i."
             numpy bincount: "if a value n is found at position i, out[n] += weight[i] instead of out[n] += 1."
             numpy unique: "For multi-dimensional inputs, unique_inverse is reshaped" (1-D input here, so inverse is 1-D)
             Cameron & Miller 2015: "Each bootstrap resample will have exactly G clusters"
retrieved:   2026-10-29
note:        _cluster_totals maps each episode to its date (np.unique return_inverse on date ordinals) and returns per-date
             sum and count via bincount. bootstrap_ratio_draws then draws w ~ Multinomial(G, 1/G each) per replicate, which is
             the count of times each date appears in G draws with replacement (= pairs cluster bootstrap), and forms
             (w@sums)/(w@counts). The draws therefore depend on episodes only through per-date sums and counts.
             Caveat for the falsifier (not a refutation): permuting values within a date changes bincount's summation
             order, so per-date sums, and hence draws, can differ in the last ULPs. A bitwise-equality probe could
             report a spurious difference; a tolerance (~1e-12) is the right comparison.
```

```
claim_id:    C4
verdict:     CONFIRMED
source:      src/llm_lookahead_probe/stats.py:59-66, 86-91; numpy 2.5 Generator.multinomial page (as C3)
source_tier: code + 1
version:     HEAD 664301b; numpy 2.5
quote:       "return (w @ cluster_sums) / (w @ cluster_counts)"
             numpy multinomial: "Each sample drawn from the distribution represents n such experiments."
retrieved:   2026-10-29
note:        Argument. Let S_g, n_g be per-date sum and count, w_g the multinomial counts with sum_g w_g = G (n = G
             experiments). Replicate = sum w_g S_g / sum w_g n_g. With w = (1,...,1) this is sum S_g / sum n_g =
             d.mean(), the point estimate, so replicate and estimate are the same functional (plug-in). Balanced panel,
             n_g = k for all g: replicate = sum w_g S_g / (k G) = (1/G) sum w_g (S_g/k), the mean of the resampled
             date means. Both parts of the claim follow from the code and the multinomial definition. Whether resampling the
             ratio is statistically preferable is not part of the claim; no primary source on it was retrieved.
```

```
claim_id:    C5
verdict:     CONFIRMED
source:      src/llm_lookahead_probe/stats.py:126-155;
             Hesterberg (2014) arXiv:1411.5279v1, Section 2.4 "Two-Sample Bootstrap" and Section 3.2 (p. 22)
             https://arxiv.org/pdf/1411.5279
source_tier: code + 4
version:     HEAD 664301b; arXiv v1 (19 Nov 2014)
quote:       Hesterberg 2.4: "we independently draw bootstrap samples from each sample"
             Hesterberg 3.2: "this narrowness bias depends on the individual sample or strata sizes"
retrieved:   2026-10-29
note:        Records are split by e.date < cutoff into disjoint date sets; each part is cluster-resampled with its own
             child SeedSequence (ss.spawn(3)), and the diff CI is the percentile interval of draws_pre - draws_post. That
             is the standard two-sample / stratified bootstrap of a difference (each stratum resampled independently,
             stratum sizes fixed), matching the stratified sampling of decision dates in sample_episodes. Confirmed as a
             procedure. Not decided here: the numeric coverage band in the falsifier ([0.93, 0.97]); that is Channel B.
             Consequence of the second quote: the diff CI inherits the percentile narrowness of the per-stratum G
             (about 30 per side on SMALL), not of the combined 60. The numpy SeedSequence.spawn page (NumPy v2.5)
             says only "Spawn a number of child SeedSequence s by extending the spawn_key"; it makes no independence
             statement, so stream independence rests on numpy's general SeedSequence design, not on a quoted guarantee.
```

```
claim_id:    C6
verdict:     UNVERIFIABLE
source:      Hesterberg (2014) arXiv:1411.5279v1, Section 5.2 and Table 5 (p. 49), Section 5.3, Table 6 (p. 63);
             Cameron, Gelbach & Miller, NBER TWP 344 (2007), Section 3.1, Table 2; Cameron & Miller (2015) JHR Sections II.F, VI.A, VI.C, VI.D
source_tier: 4
version:     arXiv v1 2014; NBER TWP 344 Sept 2007; JHR 2015 author PDF
quote:       Hesterberg abstract: "the common bootstrap percentile interval badly under-covers in small samples"
             Hesterberg 5.2: "These effects are O(n^-1) ... so they become negligible fairly quickly as n increases."
retrieved:   2026-10-29
note:        A coverage rate on this specific mock is an emergent numeric property; no document decides it. It is Channel B
             territory by construction. What the literature predicts is recorded under "C6: literature on expected coverage"
             below: about 0.943 at G = 60 and 0.947 at G = 120 for symmetric (normal) cluster means, both inside the
             [0.935, 0.965] band, so the literature does not predict a refutation, but nothing here confirms it either.
```

```
claim_id:    C7
verdict:     CONFIRMED
source:      src/llm_lookahead_probe/mocks.py:171-204; masking.py:227, 248-249, 257-261; prompt.py:363-372; data.py:124-126;
             Python 3.12 string format spec https://docs.python.org/3.12/library/string.html
source_tier: code + 1
version:     HEAD 664301b; Python 3.12.16 docs
quote:       Python format spec 'f': "formats the number as a decimal number with exactly p digits following the decimal point."
             mocks.py: "a = Action.LONG if closes[-1] > closes[0] else Action.SHORT"
retrieved:   2026-10-29
note:        Argument (base action). Synthetic closes are cent-rounded (np.round(..., 2)), so real closes are A/100 and
             B/100 for integers A, B, and the unmasked prompt shows them exactly. Masked shows round2(f*A/100) and
             round2(f*B/100) with f = exp(U[log 1.5, log 6]) >= 1.5 (up to 1 ulp). If B > A then f(B-A)/100 >= 0.015;
             nearest rounding moves each value by <= 0.005, so the rounded difference is >= 0.005, i.e. >= 0.01 on the cent
             grid, still > 0. Same for B < A. If B == A the doubles are identical (np.round maps both to the same
             double), f*b is the same double, the strings are equal and both arms ABSTAIN. So sign and ties are
             preserved and the base action is identical in both arms. The argument needs round-to-nearest in '.2f'; the
             quoted doc says only "exactly p digits", so nearest rounding is an assumption about CPython's formatter
             (it has slack: any rounding error < 0.0075 per value suffices). The scope is the synthetic panel only: a CSV
             with sub-cent prices is outside this argument.
             Argument (noise). flip and garble use _uniform(seed, purpose, prompt) = SHA-256 of the prompt text, and
             the masked and unmasked prompts differ, so each arm gets its own draw from the same rule (flip_prob 0.25,
             garble_prob 0.02). Given equal base actions, the per-episode distributions of pnl_u and pnl_m are equal,
             so E[gap] = 0 exactly under the standard idealisation that SHA-256 outputs behave as independent
             uniforms. The |t| > 3 falsifier over seeds is Channel B.
```

```
claim_id:    C8
verdict:     CONFIRMED
source:      src/llm_lookahead_probe/masking.py:241-261; Python 3.12 datetime docs https://docs.python.org/3.12/library/datetime.html;
             numpy 2.5 Generator.permutation (semantics: permutation of range(n))
source_tier: code + 1
version:     HEAD 664301b; Python 3.12.16 docs
quote:       datetime: "A week is converted to 7 days."
             datetime: "date2 will be timedelta.days days after date1." / weekday: "Monday is 0 and Sunday is 6"
retrieved:   2026-10-29
note:        (a) Returns: masked price = format_price(close * f) with a single per-ticker f, so before display rounding
             every within-window ratio is unchanged (float relative error ~1e-16); the only perturbation is the cent
             rounding of each masked price (<= 0.005 under nearest rounding, see C7 note), which is what the claim's bound
             assumes. (b) Weekday: shift = timedelta(weeks=k) is 7k days, and adding a multiple of 7 days keeps
             weekday(). (c) Bijection: perm = rng.permutation(n) is a permutation of 0..n-1, so the integers are
             distinct and f"ASSET_{k:0{width}d}" maps distinct integers to distinct strings. (d) Shift size:
             span_weeks = span_days // 7 + 1 > span_days / 7, and |shift_weeks| >= span_weeks + 1, so every masked date
             d - 7|shift| < dates[-1] - span_days = dates[0]: every masked date precedes the first real date and none
             can lie in the real panel range. A date near date.min would raise OverflowError per the datetime docs, which
             is fail-loud and irrelevant for 2024 panels.
```

```
claim_id:    C9
verdict:     CONFIRMED
source:      src/llm_lookahead_probe/masking.py:264-296; data.py:22-27; prompt.py:354-364;
             Python 3.12 re docs https://docs.python.org/3.12/library/re.html
source_tier: code + 1
version:     HEAD 664301b; Python 3.12.16 docs
quote:       re: "(?<!...) Matches if the current position in the string is not preceded by a match for ..."
             re: "the contained pattern must only match strings of some fixed length."
             re.findall: "If there are no groups, return a list of strings matching the whole pattern."
retrieved:   2026-10-29
note:        Ticker regex: (?<![A-Za-z0-9_])(?<![A-Za-z0-9][.\-])(?:alts)(?![A-Za-z0-9_])(?![.\-][A-Za-z0-9]). Both lookbehinds
             are fixed width (1 and 2), so they are legal. Alternatives are sorted longest first, so "BRK.B" wins over "BRK".
             Under the quoted semantics, a canonical upper-case ticker is matched at start/end of text and when
             adjacent to space, ",", ";", ":", "!", "?", "(", ")", quotes, "*", "/", and a sentence-end "." or "-" not
             followed by an alphanumeric ("AAPL." raises). Single-letter tickers are matched only as standalone
             upper-case letters; the template has no standalone upper-case letter and the search is case-sensitive, so
             "a" in "You are a trading agent" cannot false-alarm. Pseudonym "ASSET_nn": a numeric ticker "nn" is blocked
             by the "_" lookbehind; a CSV ticker "ASSET" is blocked by the "_" lookahead; synthetic tickers exclude
             "ASSET*" and reserved words. Dates: _DATE_TOKEN finds every YYYY-MM-DD not embedded in a longer digit run,
             and membership in the real date set raises. Prices: _NUMBER_TOKEN tokens are compared as strings with
             format_price of the episode's real window and next close, so a real "d.dd" token raises; masked date tokens
             contain no "." and cannot equal a d.dd string. Confirmed for "whole token" in the sense the regex defines it,
             where "_" is a word character and an internal "." or "-" joins a compound token. Under a broader reading of
             "any punctuation context" there are escapes (see New finding N1); those are recorded separately because
             the claim's falsifier and mutation target the sentence-end "." case, which the code handles.
```

```
claim_id:    C10
verdict:     CONFIRMED
source:      src/llm_lookahead_probe/probe.py:98-112, 127-139
source_tier: code
version:     HEAD 664301b
quote:       "checker.check(prompt_m, real_prices=[b.close for b in obs_u.bars] + [next_close])"
             "plans = plan_episodes(panel, episodes, Masker.fit(panel, cfg.seed), cfg.window)" followed by "result = execute(plans, model)"
retrieved:   2026-10-29
note:        plan_episodes calls only build_observation, masker.mask, render_prompt and checker.check; it holds no Model.
             check raises LeakError inside the loop, so the exception propagates out of plan_episodes before run_offline
             reaches execute(), the only place Model.complete is called. An identity masker yields prompt_m == prompt_u,
             which contains the real ticker twice (template) and the real dates and prices, so the first episode raises.
             Scope: verified for probe.run_offline. The live path in cli.py was outside the inputs given to Channel A and
             was not read; whether the live path also plans before executing is not verified here.
```

```
claim_id:    C11
verdict:     CONFIRMED
source:      src/llm_lookahead_probe/pit.py:266-331; prompt.py:367-372; masking.py:257-261;
             Python 3.12 bisect docs https://docs.python.org/3.12/library/bisect.html
source_tier: code + 1
version:     HEAD 664301b; Python 3.12.16 docs
quote:       bisect: "all(elem <= x for elem in a[lo : ip]) is true for the left slice and all(elem > x for elem in a[ip : hi]) is true for the right slice."
retrieved:   2026-10-29
note:        _rows_on_or_before takes indices < bisect_right(panel.dates, as_of), so every bar is dated <= as_of (panel.dates
             is strictly increasing, which PricePanel.__post_init__ enforces, so the bisect precondition holds). The rows are
             re-asserted by assert_no_future_rows, again in Observation.__post_init__, and again in render_prompt. Masking
             shifts every bar date and as_of by the same timedelta, so masked bars are <= as_of + shift. A future bar passed
             to Observation(...) raises AsOfViolationError in __post_init__ (assert_no_future_rows runs before the
             "last bar must be the decision date" ValueError). A bar added after construction (object.__setattr__ on the
             frozen dataclass) is caught by the assert at the top of render_prompt before any text is built.
```

```
claim_id:    C12
verdict:     UNVERIFIABLE
source:      Hesterberg (2014) Table 5; Cameron, Gelbach & Miller NBER TWP 344 Table 2; src/llm_lookahead_probe/stats.py:157-175
source_tier: 4 + code
version:     as above
quote:       Hesterberg 5.3: "The bootstrap percentile interval performs poorly in small samples"
retrieved:   2026-10-29
note:        The rate 0.020 over seeds 0-199 and any Wilson bound are measurements; no document decides them. Literature-based
             expectation (not a verdict): "suspected" requires gap_pre low > 0 AND diff low > 0, so
             P(suspected) <= P(gap_pre low > 0) under the null. For symmetric cluster means the one-sided miss of a
             percentile interval is about half of the two-sided miss: normal-theory arithmetic gives 0.936 coverage at
             G = 30 per part (SMALL: 60 dates split 30/30), i.e. about 0.032 one-sided, below 0.05. Hesterberg Table 5
             gives the same order (one-sided 0.036 at n = 20, 0.030 at n = 40). So the claim is consistent with theory
             but not decided by it.
```

```
claim_id:    C13
verdict:     UNVERIFIABLE
source:      none admissible (emergent property of MemorisingMock + verdict rule)
source_tier: n/a
version:     n/a
quote:       n/a
retrieved:   2026-10-29
note:        Power of 1.000 over 200 seeds at recall_p = 0.75 is a Monte Carlo measurement of this code's mock; documentation
             cannot decide it. Channel B only.
```

```
claim_id:    C14
verdict:     UNVERIFIABLE
source:      src/llm_lookahead_probe/stats.py:152-168; docs/DESIGN.md "Failure modes" (confounds paragraph);
             Hesterberg (2014) Section 3.2 and Table 5
source_tier: code + 4
version:     HEAD 664301b; arXiv v1
quote:       DESIGN.md: "The pre − post difference controls for a constant masking effect; it does not control for an effect that differs before vs after the cutoff."
retrieved:   2026-10-29
note:        Structure (decided): under a constant pnl effect d, E[gap_pre - gap_post] = 0, and the rule needs diff low > 0,
             so P(suspected) <= P(diff low > 0 | true diff 0), about the one-sided miss of a percentile interval with
             about 30 dates per stratum (about 0.03 by the same arithmetic as C12, narrowness set by stratum size per
             Hesterberg 3.2). The rate itself is a measurement (Channel B). Scope caveat: a behaviourally constant masking
             effect (e.g. "masked arm abstains 10% more") is not a constant pnl effect when the pre and post periods have
             different return distributions (drift, volatility), because pnl = action x r. The test controls the
             former only when it maps to equal pnl effects in both periods. The ledger's mock definition ("by the same amount
             before and after") decides which of the two Channel B tests.
```

```
claim_id:    C15
verdict:     CONFIRMED
source:      .venv/lib/python3.12/site-packages/anthropic/_client.py:166-182; resources/messages/messages.py:1012-1100;
             types/thinking_config_param.py; types/thinking_config_disabled_param.py; types/user_message_param.py; _constants.py:10-11
source_tier: 2
version:     anthropic 2.3.0
quote:       _client.py Anthropic.__init__: "api_key: str | None = None," ... "timeout: float | Timeout | None | NotGiven = not_given," "max_retries: int = DEFAULT_MAX_RETRIES,"
             thinking_config_disabled_param.py: "type: Required[Literal[\"disabled\"]]"
             messages.py create body: "\"thinking\": thinking,"
retrieved:   2026-10-29
note:        create() is keyword-only with required max_tokens, messages, model and optional thinking: ThinkingConfigParam
             (Union of Enabled, Disabled, BetweenTools, Adaptive). The body dict passes thinking through unchanged.
             create() in 2.3.0 has no temperature/top_p/top_k parameter at all, so the code cannot send temperature unless
             it used extra_body, which it does not. UserMessageParam.content is MessageContentParam (str accepted per the
             MessageParam union). A misspelled kwarg would be rejected by Python since create has no **kwargs. Whether
             the wire body omits unset keys is done by the SDK's omit/transform machinery (not re-read here); Channel B's
             captured-body probe is the direct check.
```

```
claim_id:    C16
verdict:     CONFIRMED
source:      .venv/lib/python3.12/site-packages/anthropic/types/message.py:30, 91, 123; types/text_block.py:10-21;
             types/usage.py; types/stop_reason.py
source_tier: 2
version:     anthropic 2.3.0
quote:       message.py: "content: List[ContentBlock]" / "stop_reason: Optional[StopReason] = None" / "usage: Usage"
             text_block.py: "text: str" / "type: Literal[\"text\"]"
             usage.py: "input_tokens: int" / "output_tokens: int"
retrieved:   2026-10-29
note:        ClaudeModel.complete reads b.type == "text" and b.text over resp.content, int(resp.usage.input_tokens),
             int(resp.usage.output_tokens), and resp.stop_reason; all exist with those types. stop_reason may be None,
             which the code handles (str(None) counter key; cache validator accepts str | None). StopReason includes
             "refusal" and "max_tokens", both handled as an empty or short text, which parses to ABSTAIN.
```

```
claim_id:    C17
verdict:     CONFIRMED
source:      .venv/.../anthropic/types/model.py:8 and types/model_param.py:10;
             claude-api skill SKILL.md "Current Models (cached: 2026-10-06)" and "Thinking & Effort" table; shared/models.md:87;
             shared/model-migration.md "Migrating to Claude Haiku 5.5";
             https://platform.claude.com/docs/en/about-claude/models/overview.md (redirects to /docs/en/models/overview)
source_tier: 2 (SDK) + 1 (skill reference, official docs page)
version:     anthropic 2.3.0; skill cache 2026-10-06; live docs page retrieved 2026-10-29
quote:       SDK types/model.py: "\"claude-haiku-5-5\","
             skill table: "| Claude Haiku 5.5 | `claude-haiku-5-5` | 1M | $0.10 | $0.50 |"
             skill models.md: "Adaptive thinking only, on by default (`{type: \"disabled\"}` is accepted at effort `high` or below)"
             skill migration: "`temperature` must be `1`" ... "any other value of either returns a 400"
             docs overview (Haiku 5.5 column): "Training data cutoff | Jun 2026"; pricing "From $0.10 / input MTok, From $0.50 / output MTok"; default effort "`medium`"
retrieved:   2026-10-29
note:        The model ID is in the SDK Model literal. Prices $0.10/$0.50 apply to prompts of 100K tokens or fewer
             ($0.50/$2.50 above, which these prompts never reach). The training data cutoff is "Jun 2026" on the docs page;
             the skill does not state Haiku 5.5's cutoff. Disabled thinking is accepted at low/medium/high; the code sets
             no effort, so the API default (medium) applies and the request is accepted. The code sends no
             temperature, so the sampling-parameter 400 does not apply. Fragility (not a refutation): validity depends on the
             default effort staying <= high; see N6.
```

```
claim_id:    C18
verdict:     UNVERIFIABLE
source:      src/llm_lookahead_probe/live.py:273-291; https://platform.claude.com/docs/en/models/overview (Context window footnote);
             claude-api skill shared/model-migration.md "Migrating to Claude Haiku 5.5"
source_tier: 1
version:     live docs page retrieved 2026-10-29; skill cache 2026-10-06
quote:       docs overview: "1M tokens is roughly 555k words or 2.5M Unicode characters on the current tokenizer"
             skill: "the same text counts as about 30% more tokens"
retrieved:   2026-10-29
note:        The only documented ratio is an average of about 2.5 characters per token for general text on the current
             tokenizer. The prompts are digit- and date-heavy ("2024-01-02 123.45" lines), and no document gives a
             chars/token figure for such text, so "2 chars/token is an upper bound" is neither supported nor refuted.
             The code comment "roughly 3 characters per token or more" is unsourced, and the docs say Haiku 5.5's
             tokenizer counts about 30% more tokens than Haiku 4.5's. The estimate also ignores per-request message
             framing tokens. Deciding check: messages.count_tokens on a sample of rendered prompts (needs an API key).
```

---

## C6: literature on expected coverage, and cheaper or better intervals

What the method is. `summarise` builds a percentile interval from a pairs (non-parametric) cluster bootstrap of a mean
with G clusters (decision dates). For a balanced panel the statistic is the mean of G date means, so the problem reduces
to a bootstrap percentile interval for a mean of n = G roughly independent observations.

1. Percentile narrowness (Hesterberg 2014, Sections 3.2, 5.2, 5.3, Tables 5-6). The bootstrap distribution of a mean is
   too narrow by sqrt((n-1)/n). For symmetric data the percentile interval behaves like x-bar +/- z sigma-hat/sqrt(n)
   instead of x-bar +/- t_{n-1} s/sqrt(n). These effects are O(n^-1) in coverage. Table 5's "size" column gives the
   one-sided miss as 0.036 (n = 20), 0.030 (n = 40), 0.028 (n = 80). Quote (Table 6 row): "narrowness bias O(n−1 )".
   Normal-theory arithmetic for two-sided coverage of x-bar +/- 1.96 sqrt((G-1)/G) s/sqrt(G), i.e. P(|t_{G-1}| <= c),
   computed with `math` only (a calculation, not a probe):

   | G (clusters) | 20 | 30 | 40 | 60 | 80 | 120 |
   |---|---|---|---|---|---|---|
   | expected coverage | 0.929 | 0.936 | 0.940 | 0.943 | 0.945 | 0.947 |

   These agree with Hesterberg's Table 5 at n = 20, 40, 80 (two-sided = 2 x one-sided: 0.928, 0.940, 0.944).
   Expected coverage of the overall gap CI: about 0.943 on SMALL (G = 60) and 0.947 on default (G = 120), both inside
   [0.935, 0.965]. The pre and post CIs on SMALL have about 30 dates each, so about 0.936, at the band's edge. At the
   MIN_DATES_PER_PART = 20 floor, about 0.929. These numbers assume symmetric, near-normal date means and
   ignore Monte Carlo error from B = 2000. Skewness adds O(n^-1/2) one-sided errors (Hesterberg Table 6), which the
   percentile interval only "partial[ly]" corrects. They are an expectation, not a measurement of this mock.

2. Cluster-count guidance (Cameron & Miller 2015, Sections I and VI.A). Quote: "“few” may range from less than 20 to less than 50 clusters".
   G = 60 and 120 are above that range in the balanced case, and G = 30 per part is inside it.

3. What CGM (2008, NBER TWP 344) actually reports. CGM do not evaluate a pairs-cluster percentile interval. They evaluate
   pairs cluster bootstrap-se (Wald with bootstrap SE), pairs BCA, pairs bootstrap-t, and wild cluster bootstrap-t for
   G = 5, 10, 15, 20, 25, 30 only. Table 2 (nominal 0.05): pairs bootstrap-se rejects 0.072 at G = 30, pairs BCA
   0.068, pairs bootstrap-t 0.054, wild cluster bootstrap-t 0.045. CGM say the bootstrap-se procedure "does not offer
   asymptotic refinement, and so may perform worse with few clusters" and that bootstrap-t procedures do. Nothing in CGM
   speaks to G = 60 or 120, so they cannot be cited for a coverage figure there. The direction is consistent: the
   non-pivotal pairs procedures over-reject at G = 30.

4. Cheaper or better intervals, with sources:
   - t-interval on date means with G-1 df (no resampling, cost O(G)). Hesterberg 5.3 says the t interval fixes both
     narrowness (s instead of sigma-hat) and SE variability (t_{n-1} instead of z). Cameron & Miller VI.D: "at a minimum one
     should use the T(G − 1) distribution rather than the standard normal" (quoted from the extracted text). For the
     balanced offline panel the gap is exactly the mean of date means, so this interval is exact under normal date
     means and cheaper than the bootstrap. For pre - post: a two-sample t (Welch) on date means.
   - Bootstrap-t / percentile-t (studentized). Asymptotic refinement: two-sided size error O(G^-3/2) vs O(G^-1)
     (Cameron & Miller VI.C). In Hesterberg's Figure 22 note, bootT needs the smallest n (101) for one-sided errors
     within 10% on skewed gamma data, versus 2383 for perc. Cost: one SE per replicate, which for a mean is cheap
     (a weighted variance of the same w).
   - Expanded percentile interval (Hesterberg 5.3): rescale the percentile quantiles to correct narrowness and z-vs-t.
     Same cost as now.
   - BCa: CGM Table 2 shows pairs BCA no better than CRVE at G <= 30. CGM, about Table 2: "the BCA method provides no improvement on CRVE".
     No evidence it helps here.
   - Wild cluster bootstrap-t: best size control in CGM at G <= 30 (0.045-0.062). It is designed for regression
     coefficients with few clusters. At G = 60-120 for a simple mean it adds little over the t-interval.
   Recommendation for the caller, not a verdict: at these G, a t-interval on date means with G-1 df (Welch for the
   difference) is the cheapest correction for the known narrowness, and bootstrap-t is the robust upgrade if
   date-mean skewness matters. Either change needs Channel B's coverage harness before adoption.

---

## New findings (not in ledger)

- N1 (low; leak-check scope). Under the Python `re` semantics quoted in C9, these canonical upper-case ticker
  occurrences do not raise: hyphen/dot compounds ("AAPL-like", "pre-AAPL", "e.g.AAPL", "AAPL.com"), underscore
  contexts ("_AAPL_", markdown "__AAPL__"), and any lowercase or mixed-case form. The template never puts a ticker in
  these contexts and the masked prompt carries a pseudonym, so this matters only if a future template or masker bug
  injects text. It is a deliberate trade-off that lets dotted/dashed real tickers (BRK.B) be distinct tokens.
- N2 (low-medium for CSV mode; fail-closed false alarm). The per-episode price check compares every number token in the
  masked prompt against the real window and next close as "d.dd" strings. A masked price round2(f*p_i) can equal a real
  price p_j whenever p_j/p_i is close to f, possible when the 21 closes span a ratio of 1.5 or more (f >= 1.5). On the
  synthetic panel this is about a 4-5 sigma 20-day move, effectively never. On a user CSV (volatile names, or
  unadjusted splits, which the loader does not reject; corporate actions are a non-goal) one coincidence aborts the
  whole run with LeakError before any call. Fail-closed, so safe but disruptive. DESIGN.md's "cannot overlap the real
  window unless the window moves more than 50%" states the condition but not the consequence.
- N3 (medium; statistics). Every CI is a plain percentile interval. Literature (C6 section) predicts undercoverage of
  about 0.6 points at G = 60, 1.4 at G = 30 and 2.1 at the G = 20 floor, all from narrowness bias, which is cheaply
  fixable with a t-interval on date means (G-1 df) or bootstrap-t. VERDICT_RULE and the report label these as
  "95%" without that caveat.
- N4 (low; cost). C18: documentation gives about 2.5 chars/token for general text on the current tokenizer, and Haiku
  5.5 counts about 30% more tokens than Haiku 4.5. The "2 chars/token is an upper bound" comment and the DESIGN.md "~600
  characters (~200-300 tokens)" figure are unmeasured for digit-heavy prompts. The planned-spend guard
  (`--max-episodes`, 2x cap) does not depend on this estimate, so the exposure is a misleading printed estimate, not
  overspend.
- N5 (medium; interpretation of C14 and DESIGN.md). The pre - post difference removes a masking effect that is constant
  in pnl units. A masking effect that is constant in behaviour (e.g. more ABSTAIN when masked) produces a pnl effect
  proportional to the period's returns (drift, volatility, the agent's edge), which differs between pre and post
  regimes. That is exactly the "effect that differs before vs after the cutoff" DESIGN.md says is not controlled.
  This should be stated where the verdict is reported, since it is the likely case for real models.
- N6 (low; API fragility). `request_extra = {"thinking": {"type": "disabled"}}` is valid on Claude Haiku 5.5 only at
  effort low/medium/high (skill models.md and migration guide). The code relies on the server default effort
  (`medium`) and never sends `output_config.effort`. Setting `output_config={"effort": "low"}` (or "medium") explicitly
  next to the disabled-thinking config would pin the accepted combination. The skill also notes that on this model
  thinking counts toward max_tokens when on, which is why the 16-token cap needs thinking off; the code's module
  docstring states this correctly.
- N7 (process note for the gate). C3's falsifier "permuting values within dates changes the draws (same seed)" must use
  a numeric tolerance: bincount accumulation order changes with a within-date permutation, so per-date sums can differ
  in the last bits.
- N8 (scope gap). C10 was verified for `probe.run_offline` only. `cli.py` (the live path that spends money) was outside
  Channel A's input list and has not been audited for plan-before-execute ordering.
