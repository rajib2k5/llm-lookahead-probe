"""Paired gap statistics with a cluster bootstrap by decision date, and the verdict rule.

Why cluster by date: all episodes on one decision date share that day's market move, and a model
that recalls a *date* (rather than a ticker) shifts every ticker on that date the same way. The
per-episode differences are then correlated within a date. An i.i.d. bootstrap treats them as
independent and gives a CI that is too narrow (tests/test_stats.py shows a > 2.5x width ratio in
a correlated example). Resampling whole dates keeps that dependence inside each resample.

The statistic is a ratio (sum of differences / number of episodes over the resampled dates), so
every episode has equal weight, the same as the point estimate.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import numpy as np

from .prompt import Action

MIN_RESAMPLES = 2000
MIN_DATES_PER_PART = 20

VERDICT_RULE = (
    "'memorisation suspected' if the lower CI bound of gap_pre > 0 AND the lower CI bound of "
    "(gap_pre - gap_post) > 0; 'insufficient data' if the pre-cutoff or post-cutoff part has "
    f"fewer than {MIN_DATES_PER_PART} decision dates; otherwise 'no evidence'. gap = mean over "
    "episodes of (pnl_unmasked - pnl_masked); CIs are percentile intervals from a cluster "
    "bootstrap that resamples decision dates; pre and post are resampled independently."
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
    low: float
    high: float

    def as_dict(self, level: float) -> dict:
        return {"estimate": self.estimate, "low": self.low, "high": self.high, "level": level}


def pnl(action: Action, r: float) -> float:
    return r if action == Action.LONG else -r if action == Action.SHORT else 0.0


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


def _check(n_resamples: int, level: float) -> None:
    if n_resamples < MIN_RESAMPLES:
        raise ValueError(f"need at least {MIN_RESAMPLES} bootstrap resamples, got {n_resamples}")
    if not 0 < level < 1:
        raise ValueError("level must be in (0, 1)")


def _percentile(draws: np.ndarray, level: float) -> tuple[float, float]:
    lo, hi = np.quantile(draws, [(1 - level) / 2, 1 - (1 - level) / 2])
    return float(lo), float(hi)


def cluster_bootstrap_ci(values, clusters, n_resamples: int, level: float, seed) -> CI:
    _check(n_resamples, level)
    v = np.asarray(values, dtype=float)
    sums, counts = _cluster_totals(v, np.asarray(clusters))
    lo, hi = _percentile(bootstrap_ratio_draws(sums, counts, n_resamples, seed), level)
    return CI(float(v.mean()), lo, hi)


def _arm_metrics(actions: list[Action], parsed: list[bool], r: np.ndarray) -> dict:
    p = np.array([pnl(a, x) for a, x in zip(actions, r, strict=True)])
    trades = [(a, x) for a, x in zip(actions, r, strict=True) if a != Action.ABSTAIN]
    hits = sum((a == Action.LONG and x > 0) or (a == Action.SHORT and x < 0) for a, x in trades)
    return {
        "mean_pnl": float(p.mean()) if len(p) else None,
        "hit_rate": hits / len(trades) if trades else None,
        "n_long": sum(a == Action.LONG for a in actions),
        "n_short": sum(a == Action.SHORT for a in actions),
        "n_abstain": sum(a == Action.ABSTAIN for a in actions),
        "n_unparseable": sum(not ok for ok in parsed),
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


def summarise(records: list[EpisodeResult], cutoff: date, n_resamples: int, level: float, seed: int) -> dict:
    _check(n_resamples, level)
    if not records:
        raise ValueError("no episodes")
    ss = np.random.SeedSequence([seed, 0xB007])
    s_all, s_pre, s_post = ss.spawn(3)

    d_all, c_all = _diffs(records)
    sums, counts = _cluster_totals(d_all, c_all)
    lo, hi = _percentile(bootstrap_ratio_draws(sums, counts, n_resamples, s_all), level)
    gap = CI(float(d_all.mean()), lo, hi)

    parts: dict[str, dict] = {}
    draws: dict[str, np.ndarray] = {}
    for name, seq, recs in (
        ("pre", s_pre, [e for e in records if e.date < cutoff]),
        ("post", s_post, [e for e in records if e.date >= cutoff]),
    ):
        part: dict = {"n_episodes": len(recs), "n_dates": len({e.date for e in recs})}
        if recs:
            d, c = _diffs(recs)
            ps, pc = _cluster_totals(d, c)
            draws[name] = bootstrap_ratio_draws(ps, pc, n_resamples, seq)
            plo, phi = _percentile(draws[name], level)
            part["gap"] = CI(float(d.mean()), plo, phi).as_dict(level)
            part["arms"] = _arms(recs)
        else:
            part["gap"] = None
        parts[name] = part

    diff = None
    if "pre" in draws and "post" in draws:
        dlo, dhi = _percentile(draws["pre"] - draws["post"], level)
        diff = CI(parts["pre"]["gap"]["estimate"] - parts["post"]["gap"]["estimate"], dlo, dhi)

    if parts["pre"]["n_dates"] < MIN_DATES_PER_PART or parts["post"]["n_dates"] < MIN_DATES_PER_PART:
        verdict = "insufficient data"
        reason = (
            f"insufficient data: pre has {parts['pre']['n_dates']} and post has "
            f"{parts['post']['n_dates']} decision dates; each needs at least {MIN_DATES_PER_PART}"
        )
    elif parts["pre"]["gap"]["low"] > 0 and diff is not None and diff.low > 0:
        verdict = "memorisation suspected"
        reason = (
            f"gap_pre CI low {parts['pre']['gap']['low']:.6f} > 0 and "
            f"(gap_pre - gap_post) CI low {diff.low:.6f} > 0"
        )
    else:
        verdict = "no evidence"
        reason = (
            f"gap_pre CI low {parts['pre']['gap']['low']:.6f}, "
            f"(gap_pre - gap_post) CI low {diff.low if diff else float('nan'):.6f}: "
            "both must be > 0"
        )

    return {
        "n_episodes": len(records),
        "n_dates": len(set(c_all.tolist())),
        "arms": _arms(records),
        "gap": gap.as_dict(level),
        "pre": parts["pre"],
        "post": parts["post"],
        "diff_pre_minus_post": diff.as_dict(level) if diff else None,
        "verdict": verdict,
        "verdict_reason": reason,
        "verdict_rule": VERDICT_RULE,
        "bootstrap": {
            "method": "cluster bootstrap by decision date (percentile CI)",
            "n_resamples": n_resamples,
            "level": level,
            "seed": seed,
            "min_dates_per_part": MIN_DATES_PER_PART,
        },
    }
