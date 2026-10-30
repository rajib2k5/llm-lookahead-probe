"""Paired gap statistics: t-intervals on per-date mean differences, and the verdict rule.

Why dates are the unit: all episodes on one decision date share that day's market move, and a
model that recalls a *date* (rather than a ticker) shifts every ticker on that date the same way.
Per-episode differences are therefore correlated within a date, and treating them as independent
gives an interval that is too narrow (tests/test_stats.py shows a > 2.5x width ratio for an i.i.d.
bootstrap in a correlated example). We collapse each date to its mean paired difference and treat
the G date means as the independent observations.

Interval used by the verdict: Student t on the G date means with G - 1 degrees of freedom; for
gap_pre - gap_post, Welch's interval with Welch-Satterthwaite df (pre and post are disjoint sets of
dates with possibly different variances). v0.1 used a percentile cluster bootstrap instead. That
was measured (2000+ seeds per configuration, docs/validation/report.md) to cover about 93.5-94.5%
at 30-60 dates when labelled 95%, which is the known small-sample under-coverage of percentile
intervals (Hesterberg, arXiv:1411.5279; Cameron, Gelbach & Miller on few-cluster inference). The
t-interval covered 94.6-95.4% in the same simulations. The bootstrap is still reported, as a
secondary robustness number, under "robustness".

Rectangular panel requirement: the mean of the date means equals the mean over episodes only when
every date has the same number of episodes. The probe samples a fixed ticker set crossed with a set
of dates from a rectangular panel, so this holds; `summarise` refuses anything else rather than
silently weighting dates differently from the point estimate.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date
from functools import lru_cache

import numpy as np

from .prompt import Action

MIN_RESAMPLES = 2000
MIN_DATES_PER_PART = 20
MAX_UNPARSEABLE_RATE = 0.10

VERDICT_RULE = (
    "'invalid run' if, in the whole run or in the pre-cutoff or post-cutoff part, either arm has "
    f"more than {MAX_UNPARSEABLE_RATE:.0%} unparseable answers (refusals, empty or truncated "
    "replies count as ABSTAIN and would otherwise hide a broken run as a zero gap); else "
    "'insufficient data' if the pre-cutoff or post-cutoff part has fewer than "
    f"{MIN_DATES_PER_PART} decision dates; else 'memorisation suspected' if the lower CI bound "
    "of gap_pre > 0 AND the lower CI bound of (gap_pre - gap_post) > 0; otherwise 'no evidence'. "
    "gap = mean over episodes of (pnl_unmasked - pnl_masked). CIs are Student t intervals on the "
    "per-date mean differences with G - 1 df (G = decision dates); the pre - post CI is Welch's "
    "interval with Welch-Satterthwaite df. Every date must have the same number of episodes."
)
INTERVAL_METHOD = (
    "t-interval on per-date mean paired differences (G - 1 df); Welch t (Welch-Satterthwaite df) "
    "for pre - post"
)


@dataclass(frozen=True)
class EpisodeResult:
    ticker: str
    date: date
    r: float  # next-day simple return
    action_u: Action
    parsed_u: bool
    action_m: Action
    parsed_m: bool


@dataclass(frozen=True)
class CI:
    estimate: float
    low: float | None
    high: float | None
    df: float | None = None
    se: float | None = None

    def as_dict(self, level: float) -> dict:
        out = {"estimate": self.estimate, "low": self.low, "high": self.high, "level": level}
        if self.df is not None or self.se is not None:
            out["df"] = self.df
            out["se"] = self.se
        return out


def pnl(action: Action, r: float) -> float:
    return r if action == Action.LONG else -r if action == Action.SHORT else 0.0


# ---- Student t quantile without scipy -------------------------------------------------------
# Regularised incomplete beta by Lentz's continued fraction (Numerical Recipes, 3rd ed., 6.4);
# t CDF via I_x(df/2, 1/2) with x = df / (df + t^2); quantile by bisection. Checked against
# printed t tables in tests/test_stats.py.


def _betacf(a: float, b: float, x: float) -> float:
    tiny = 1e-300
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c, d = 1.0, 1.0 - qab * x / qap
    d = 1.0 / (d if abs(d) > tiny else tiny)
    h = d
    for m in range(1, 1000):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        d = 1.0 / (d if abs(d) > tiny else tiny)
        c = 1.0 + aa / (c if abs(c) > tiny else tiny)
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        d = 1.0 / (d if abs(d) > tiny else tiny)
        c = 1.0 + aa / (c if abs(c) > tiny else tiny)
        de = d * c
        h *= de
        if abs(de - 1.0) < 1e-15:
            break
    return h


def _ibeta(a: float, b: float, x: float) -> float:
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    bt = math.exp(math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b) + a * math.log(x) + b * math.log1p(-x))
    if x < (a + 1.0) / (a + b + 2.0):
        return bt * _betacf(a, b, x) / a
    return 1.0 - bt * _betacf(b, a, 1.0 - x) / b


def t_cdf(t: float, df: float) -> float:
    tail = 0.5 * _ibeta(df / 2.0, 0.5, df / (df + t * t))
    return 1.0 - tail if t > 0 else tail


@lru_cache(maxsize=4096)
def t_quantile(q: float, df: float) -> float:
    """Quantile of Student's t with `df` (> 0, may be fractional) degrees of freedom."""
    if not 0.0 < q < 1.0:
        raise ValueError("q must be in (0, 1)")
    if not df > 0:
        raise ValueError("df must be > 0")
    if q < 0.5:
        return -t_quantile(1.0 - q, df)
    lo, hi = 0.0, 1.0
    while t_cdf(hi, df) < q:
        lo, hi = hi, hi * 2.0
    for _ in range(100):
        mid = (lo + hi) / 2.0
        if t_cdf(mid, df) < q:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2.0


# ---- intervals ---------------------------------------------------------------------------------


def _check_level(level: float) -> None:
    if not 0 < level < 1:
        raise ValueError("level must be in (0, 1)")


def _check_resamples(n_resamples: int) -> None:
    if n_resamples != 0 and n_resamples < MIN_RESAMPLES:
        raise ValueError(
            f"need at least {MIN_RESAMPLES} bootstrap resamples (or 0 to skip the bootstrap), "
            f"got {n_resamples}"
        )


def _date_means(d: np.ndarray, c: np.ndarray) -> np.ndarray:
    _, inv = np.unique(c, return_inverse=True)
    return np.bincount(inv, weights=d) / np.bincount(inv)


def t_interval(date_means: np.ndarray, level: float) -> CI:
    """Two-sided t interval for the mean of G independent date means (G - 1 df).
    With G < 2 there is no variance estimate and the bounds are None."""
    g = len(date_means)
    est = float(np.mean(date_means))
    if g < 2:
        return CI(est, None, None)
    se = float(np.std(date_means, ddof=1) / math.sqrt(g))
    half = t_quantile(1 - (1 - level) / 2, g - 1) * se
    return CI(est, est - half, est + half, df=float(g - 1), se=se)


def welch_interval(a: np.ndarray, b: np.ndarray, level: float) -> CI:
    """Welch interval for mean(a) - mean(b) with Welch-Satterthwaite df. If both samples have zero
    variance the interval collapses to the point (no uncertainty is measurable)."""
    est = float(np.mean(a) - np.mean(b))
    if len(a) < 2 or len(b) < 2:
        return CI(est, None, None)
    va = float(np.var(a, ddof=1)) / len(a)
    vb = float(np.var(b, ddof=1)) / len(b)
    se = math.sqrt(va + vb)
    if se == 0.0:
        return CI(est, est, est, df=None, se=0.0)
    df = (va + vb) ** 2 / (va**2 / (len(a) - 1) + vb**2 / (len(b) - 1))
    half = t_quantile(1 - (1 - level) / 2, df) * se
    return CI(est, est - half, est + half, df=df, se=se)


# ---- cluster bootstrap (secondary robustness number) --------------------------------------------


def bootstrap_ratio_draws(
    cluster_sums: np.ndarray, cluster_counts: np.ndarray, n_resamples: int, seed
) -> np.ndarray:
    """B draws of sum/count after resampling clusters with replacement (multinomial weights)."""
    rng = np.random.default_rng(seed)
    g = len(cluster_sums)
    w = rng.multinomial(g, np.full(g, 1.0 / g), size=n_resamples).astype(float)
    return (w @ cluster_sums) / (w @ cluster_counts)


def _cluster_totals(values: np.ndarray, clusters: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    _, inv = np.unique(clusters, return_inverse=True)
    return np.bincount(inv, weights=values), np.bincount(inv).astype(float)


def _percentile(draws: np.ndarray, level: float) -> tuple[float, float]:
    lo, hi = np.quantile(draws, [(1 - level) / 2, 1 - (1 - level) / 2])
    return float(lo), float(hi)


def cluster_bootstrap_ci(values, clusters, n_resamples: int, level: float, seed) -> CI:
    if n_resamples < MIN_RESAMPLES:
        raise ValueError(f"need at least {MIN_RESAMPLES} bootstrap resamples, got {n_resamples}")
    _check_level(level)
    v = np.asarray(values, dtype=float)
    sums, counts = _cluster_totals(v, np.asarray(clusters))
    lo, hi = _percentile(bootstrap_ratio_draws(sums, counts, n_resamples, seed), level)
    return CI(float(v.mean()), lo, hi)


# ---- per-arm metrics -----------------------------------------------------------------------------


def _arm_metrics(actions: list[Action], parsed: list[bool], r: np.ndarray) -> dict:
    """Per-arm counts. hit_rate = hits / trades, where a trade is a LONG or SHORT answer and a hit
    is LONG with r > 0 or SHORT with r < 0. A trade on r == 0 is a trade and not a hit (its pnl is
    0). ABSTAIN, including unparseable answers, is not a trade; with no trades hit_rate is None."""
    p = np.array([pnl(a, x) for a, x in zip(actions, r, strict=True)])
    trades = [(a, x) for a, x in zip(actions, r, strict=True) if a != Action.ABSTAIN]
    hits = sum((a == Action.LONG and x > 0) or (a == Action.SHORT and x < 0) for a, x in trades)
    n_unparseable = sum(not ok for ok in parsed)
    return {
        "mean_pnl": float(p.mean()) if len(p) else None,
        "hit_rate": hits / len(trades) if trades else None,
        "n_long": sum(a == Action.LONG for a in actions),
        "n_short": sum(a == Action.SHORT for a in actions),
        "n_abstain": sum(a == Action.ABSTAIN for a in actions),
        "n_unparseable": n_unparseable,
        "unparseable_rate": n_unparseable / len(parsed) if parsed else None,
    }


def _arms(recs: list[EpisodeResult]) -> dict:
    r = np.array([e.r for e in recs])
    return {
        "unmasked": _arm_metrics([e.action_u for e in recs], [e.parsed_u for e in recs], r),
        "masked": _arm_metrics([e.action_m for e in recs], [e.parsed_m for e in recs], r),
    }


def _diffs(recs: list[EpisodeResult]) -> tuple[np.ndarray, np.ndarray]:
    d = np.array([pnl(e.action_u, e.r) - pnl(e.action_m, e.r) for e in recs])
    c = np.array([e.date.toordinal() for e in recs])
    return d, c


def _require_balanced(records: list[EpisodeResult]) -> None:
    counts: dict[date, int] = {}
    for e in records:
        counts[e.date] = counts.get(e.date, 0) + 1
    if len(set(counts.values())) > 1:
        raise ValueError(
            "every decision date must have the same number of episodes (rectangular panel); got "
            f"between {min(counts.values())} and {max(counts.values())} per date"
        )


def _invalid_reason(arms_by_part: dict[str, dict]) -> str | None:
    for part, arms in arms_by_part.items():
        for arm in ("unmasked", "masked"):
            rate = arms[arm]["unparseable_rate"]
            if rate is not None and rate > MAX_UNPARSEABLE_RATE:
                return (
                    f"invalid run: {arm} arm has {rate:.1%} unparseable answers in the {part} "
                    f"episodes (limit {MAX_UNPARSEABLE_RATE:.0%}); a gap measured on refusals or "
                    "empty replies says nothing about memory"
                )
    return None


def _scaled(gap: float, recs: list[EpisodeResult]) -> dict:
    """gap / mean|r| for one period. pre - post cancels a masking effect that is constant in pnl,
    but an effect that is constant in *behaviour* (e.g. abstaining more often when masked) costs
    pnl in proportion to |r|, so it differs between a calm and a volatile period. Comparing the
    scaled gaps of pre and post shows whether a pre - post difference could be such an effect."""
    mar = float(np.mean([abs(e.r) for e in recs]))
    return {"mean_abs_r": mar, "gap_per_mean_abs_r": gap / mar if mar > 0 else None}


def _fmt(x: float | None) -> str:
    return "n/a" if x is None else f"{x:.6f}"


def summarise(records: list[EpisodeResult], cutoff: date, n_resamples: int, level: float, seed: int) -> dict:
    """Gap estimates, intervals and the verdict. `n_resamples` = 0 skips the secondary bootstrap."""
    _check_level(level)
    _check_resamples(n_resamples)
    if not records:
        raise ValueError("no episodes")
    _require_balanced(records)

    d_all, c_all = _diffs(records)
    means_all = _date_means(d_all, c_all)
    gap = t_interval(means_all, level)

    parts: dict[str, dict] = {}
    part_means: dict[str, np.ndarray] = {}
    part_recs = {
        "pre": [e for e in records if e.date < cutoff],
        "post": [e for e in records if e.date >= cutoff],
    }
    for name, recs in part_recs.items():
        part: dict = {"n_episodes": len(recs), "n_dates": len({e.date for e in recs})}
        if recs:
            d, c = _diffs(recs)
            part_means[name] = _date_means(d, c)
            part["gap"] = t_interval(part_means[name], level).as_dict(level)
            part.update(_scaled(part["gap"]["estimate"], recs))
            part["arms"] = _arms(recs)
        else:
            part["gap"] = None
        parts[name] = part

    diff = None
    if "pre" in part_means and "post" in part_means:
        diff = welch_interval(part_means["pre"], part_means["post"], level)

    arms_all = _arms(records)
    invalid = _invalid_reason(
        {"whole-run": arms_all} | {n: p["arms"] for n, p in parts.items() if "arms" in p}
    )
    if invalid:
        verdict, reason = "invalid run", invalid
    elif parts["pre"]["n_dates"] < MIN_DATES_PER_PART or parts["post"]["n_dates"] < MIN_DATES_PER_PART:
        verdict = "insufficient data"
        reason = (
            f"insufficient data: pre has {parts['pre']['n_dates']} and post has "
            f"{parts['post']['n_dates']} decision dates; each needs at least {MIN_DATES_PER_PART}"
        )
    elif (
        parts["pre"]["gap"]["low"] is not None
        and parts["pre"]["gap"]["low"] > 0
        and diff is not None
        and diff.low is not None
        and diff.low > 0
    ):
        verdict = "memorisation suspected"
        reason = (
            f"gap_pre CI low {parts['pre']['gap']['low']:.6f} > 0 and "
            f"(gap_pre - gap_post) CI low {diff.low:.6f} > 0"
        )
    else:
        verdict = "no evidence"
        reason = (
            f"gap_pre CI low {_fmt(parts['pre']['gap']['low'])}, "
            f"(gap_pre - gap_post) CI low {_fmt(diff.low if diff else None)}: both must be > 0"
        )

    return {
        "n_episodes": len(records),
        "n_dates": len(means_all),
        "arms": arms_all,
        "gap": gap.as_dict(level),
        **_scaled(gap.estimate, records),
        "pre": parts["pre"],
        "post": parts["post"],
        "diff_pre_minus_post": diff.as_dict(level) if diff else None,
        "verdict": verdict,
        "verdict_reason": reason,
        "verdict_rule": VERDICT_RULE,
        "interval": {
            "method": INTERVAL_METHOD,
            "level": level,
            "min_dates_per_part": MIN_DATES_PER_PART,
            "max_unparseable_rate": MAX_UNPARSEABLE_RATE,
        },
        "robustness": {"cluster_bootstrap": _bootstrap_block(records, part_recs, n_resamples, level, seed)},
    }


def _bootstrap_block(records, part_recs, n_resamples: int, level: float, seed: int) -> dict | None:
    """Percentile cluster bootstrap by decision date: the v0.1 interval, kept as a secondary
    number. It under-covers at 30-60 dates (see module docstring); the verdict does not use it."""
    if n_resamples == 0:
        return None
    s_all, s_pre, s_post = np.random.SeedSequence([seed, 0xB007]).spawn(3)
    d, c = _diffs(records)
    out: dict = {
        "method": "cluster bootstrap by decision date (percentile CI); secondary, not used by the verdict",
        "n_resamples": n_resamples,
        "seed": seed,
        "level": level,
    }
    lo, hi = _percentile(bootstrap_ratio_draws(*_cluster_totals(d, c), n_resamples, s_all), level)
    out["gap"] = {"low": lo, "high": hi}
    draws = {}
    for name, seq in (("pre", s_pre), ("post", s_post)):
        if part_recs[name]:
            pd_, pc = _diffs(part_recs[name])
            draws[name] = bootstrap_ratio_draws(*_cluster_totals(pd_, pc), n_resamples, seq)
            lo, hi = _percentile(draws[name], level)
            out[name] = {"low": lo, "high": hi}
        else:
            out[name] = None
    if len(draws) == 2:
        lo, hi = _percentile(draws["pre"] - draws["post"], level)
        out["diff_pre_minus_post"] = {"low": lo, "high": hi}
    else:
        out["diff_pre_minus_post"] = None
    return out
