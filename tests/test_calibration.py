"""Null calibration and power over many seeds, through the full pipeline (prompts, masking, mocks).

Smaller panels than the default config to keep the suite fast; the statistics are the same code.
"""

import numpy as np

from llm_lookahead_probe.probe import OfflineConfig, run_offline

N_SEEDS = 200
SMALL = dict(n_tickers=8, n_dates=60)


def test_honest_mock_ci_covers_zero_about_95_percent_of_seeds():
    covered, flagged = [], []
    for seed in range(N_SEEDS):
        rep = run_offline("honest", OfflineConfig(seed=seed, **SMALL))
        covered.append(rep["gap"]["low"] <= 0 <= rep["gap"]["high"])
        flagged.append(rep["verdict"] == "memorisation suspected")
    rate = float(np.mean(covered))
    fp = float(np.mean(flagged))
    print(f"\nnull coverage of 95% CI over {N_SEEDS} seeds: {rate:.3f}; false 'suspected' rate: {fp:.3f}")
    assert 0.90 <= rate <= 0.99, f"coverage {rate:.3f}"
    assert fp <= 0.05, f"false positive rate {fp:.3f}"


def test_memorising_mock_is_detected_in_nearly_all_seeds():
    hits = [
        run_offline("memorising", OfflineConfig(seed=seed, **SMALL))["verdict"] == "memorisation suspected"
        for seed in range(N_SEEDS)
    ]
    rate = float(np.mean(hits))
    print(f"\npower: memorising detected in {rate:.3f} of {N_SEEDS} seeds")
    assert rate >= 0.95, f"power {rate:.3f}"
