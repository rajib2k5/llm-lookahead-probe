"""Null calibration and power over many seeds, through the full pipeline (prompts, masking, mocks).

Smaller panels than the default config to keep the suite fast; the statistics are the same code.
The secondary cluster bootstrap is switched off here (n_resamples=0): it is not used by the
verdict, and the t-interval needs no resampling, so 2,000 null seeds stay cheap. Seeds run in
worker processes; each seed is deterministic, so the result does not depend on scheduling.

Why 2,000 seeds and a [0.94, 0.96] band: with N = 2000 the standard error of a measured coverage
is 0.0049 at a true 0.95 and 0.0057 at a true 0.93. A correct 95% interval falls below 0.94 with
probability about 2%; an interval that really covers 93% reaches 0.94 with probability about 4%.
So the band rejects a 93% interval, which the v0.1 band (0.90-0.99 over 200 seeds) could not:
v0.1's percentile bootstrap measured 0.938 on this configuration and passed it.

What the band cannot do: tell a t-interval from a z-interval. With 30 dates per part the two
differ by little, and a z-interval measured 0.9405 here, inside the band. The choice of t over z
(and of the G - 1 and Welch-Satterthwaite df) is pinned by the unit tests in tests/test_stats.py
(t quantiles against printed tables, and the exact interval formulas), not by this test.

The ABSTAIN gate (stats.MAX_ABSTAIN_RATE) does not move these numbers: both mocks abstain on
about 2% of prompts (garbled answers), far below 50%.
"""

import os
import time
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context

import numpy as np

from llm_lookahead_probe.probe import OfflineConfig, run_offline

N_NULL_SEEDS = 2000
N_POWER_SEEDS = 500
SMALL = dict(n_tickers=8, n_dates=60, n_resamples=0)


def _covers(c: dict, value: float = 0.0) -> bool:
    return c["low"] <= value <= c["high"]


def _null_outcome(seed: int) -> tuple[bool, bool, bool]:
    rep = run_offline("honest", OfflineConfig(seed=seed, **SMALL))
    return (
        _covers(rep["gap"]),
        _covers(rep["diff_pre_minus_post"]),
        rep["verdict"] == "memorisation suspected",
    )


def _power_outcome(seed: int) -> bool:
    return run_offline("memorising", OfflineConfig(seed=seed, **SMALL))["verdict"] == "memorisation suspected"


def _map(fn, seeds):
    workers = min(8, os.cpu_count() or 1)
    with ProcessPoolExecutor(workers, mp_context=get_context("spawn")) as ex:
        return list(ex.map(fn, seeds, chunksize=50))


def test_honest_mock_intervals_cover_zero_at_the_nominal_95_percent():
    t0 = time.perf_counter()
    out = np.array(_map(_null_outcome, range(N_NULL_SEEDS)))
    gap_cov, diff_cov, fp = out.mean(axis=0)
    print(
        f"\nnull over {N_NULL_SEEDS} seeds ({time.perf_counter() - t0:.1f}s): gap CI coverage "
        f"{gap_cov:.4f}; pre-post CI coverage {diff_cov:.4f}; false 'suspected' rate {fp:.4f}"
    )
    assert 0.94 <= gap_cov <= 0.96, f"gap coverage {gap_cov:.4f}"
    assert 0.94 <= diff_cov <= 0.96, f"pre-post coverage {diff_cov:.4f}"
    assert fp <= 0.05, f"false positive rate {fp:.4f}"


def test_memorising_mock_is_detected_in_nearly_all_seeds():
    t0 = time.perf_counter()
    rate = float(np.mean(_map(_power_outcome, range(N_POWER_SEEDS))))
    secs = time.perf_counter() - t0
    print(f"\npower: memorising detected in {rate:.4f} of {N_POWER_SEEDS} seeds ({secs:.1f}s)")
    assert rate >= 0.95, f"power {rate:.4f}"
