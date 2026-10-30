# Evidence-gated validation report (v0.1, commit 664301b)

Protocol: `evidence-gated-validation`, Tier 2. Channel A read primary documentation,
papers and the installed SDK source. Channel B ran the probes in `probes/`. The ledger
(`claims-ledger.md`) was frozen before either channel ran. Channel A's summary arrived
before the probe scripts were written, so each ACCEPTED below is "dual-confirmed
(fallback mode)", not fully independent.

Environment: see `env-pin.txt` (Python 3.12.13, numpy 2.5.4, anthropic 2.3.0).
Tier 0: `pytest` 99 passed; `ruff check src` clean. Each probe failed against its
mutant (16 mutants, `probes/mutations.py`).

## Result per claim

| # | Claim | Result | Evidence |
|---|---|---|---|
| 1 | Gap = paired episode mean of pnl_u − pnl_m; pnl = ±r or 0 | ACCEPTED | Matches an episode-mean reference to 1e-15 on unbalanced dates; sign-flip and date-mean mutants fail p01/p02 |
| 2 | Date-cluster bootstrap resamples whole dates; statistic matches point estimate | ACCEPTED | Code [2.513, 3.465] vs index-resampling reference [2.506, 3.476]; i.i.d. mutant fails p03 |
| 3a | Pre − post CI uses independent (stratified) resampling of disjoint date sets | ACCEPTED | Hesterberg 2014 §2.4; width 0.967 of reference |
| 3b | Pre − post CI covers 95% | **REJECTED** | 0.925 (G=30+30, N=1000) pre-registered; follow-up N=4000: 0.938 / 0.940 / 0.941 for G=20/30/60. Welch t on date means: 0.954 / 0.951 / 0.946 |
| 4 | Gap CI covers 95% under the honest null | ACCEPTED (borderline) | 2000 seeds: SMALL 8×60 0.938 [0.927, 0.948]; DEFAULT 20×120 0.947. t on date means: 0.947 / 0.954 |
| 5 | Honest mock's true gap is exactly 0 | ACCEPTED | 0 base-action mismatches in 2.4M episodes; mean gap over 1000 seeds −1.7e-5 (t = −1.20) |
| 6 | Masking keeps returns; leak check catches canonical leaks, fails closed | ACCEPTED | Max return change 5.7 bp (cent-rounding bound); non-canonical forms (lowercase, other date/price formats) pass the check but the renderer cannot emit them |
| 7 | as_of guard blocks rows after t | ACCEPTED | 24,000 episodes; future bar raises at construction and render |
| 8 | False-positive rate and power | ACCEPTED | FP 29/2000 = 0.0145 (SMALL), 5/1000 (DEFAULT). Power 1.000 at p=0.75, 0.975 at p=0.55 |
| 9 | anthropic SDK call shape | ACCEPTED | SDK 2.3.0 source + stub server: body `{"max_tokens":16,...,"model":"claude-haiku-5-5","thinking":{"type":"disabled"}}`. Server acceptance, price, cutoff not checked live |

Literature for 3b/4: Hesterberg (arXiv:1411.5279) on percentile-interval
under-coverage in small samples; Cameron, Gelbach & Miller (NBER TWP 344) on
few-cluster bootstrap rejection rates.

## Required fixes (most severe first)

1. Medium: intervals labelled 95% cover about 93.5–94.5%. Use a t-interval on
   per-date mean differences (G−1 df), Welch for pre − post. Tighten the calibration test.
2. Medium: pre − post removes only a masking effect that is constant in pnl. A
   behaviour-constant effect scales with each period's |r|, so a more volatile
   pre-cutoff period can give pre − post > 0 with no memory. Document; report gap / mean|r|.
3. Low–medium: masking keeps non-string keys (the exact return path; the holiday-gap
   pattern under a whole-week shift). Both bias toward under-detection. Document.
4. Low: power for partial memorisation is not measured.
5. Low: on volatile series a masked price can equal a real price and abort a clean run
   (fails closed). Redraw the factor on collision.
6. Low: pin effort for Haiku 5.5 or document reliance on the server default.
7. Info: pin the prompt token grammar with a test.

## Not validated

Real models, real CSV data, a live API call, `cli.py` beyond call ordering, the token
cost estimate, cross-process determinism.
