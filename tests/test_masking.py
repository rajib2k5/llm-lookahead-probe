import re

import numpy as np
import pytest

from llm_lookahead_probe.data import synthetic_panel
from llm_lookahead_probe.masking import LeakChecker, LeakError, Masker
from llm_lookahead_probe.pit import build_observation
from llm_lookahead_probe.prompt import render_prompt


@pytest.fixture
def panel():
    return synthetic_panel(seed=2, n_tickers=12, n_days=120)


@pytest.fixture
def masker(panel):
    return Masker.fit(panel, seed=7)


def obs_pair(panel, masker, i=60, col=3, window=20):
    real = build_observation(panel, panel.tickers[col], panel.dates[i], window=window)
    return real, masker.mask(real)


def test_pseudonyms_are_stable_unique_and_well_formed(panel, masker):
    names = [masker.pseudonym(t) for t in panel.tickers]
    assert len(set(names)) == len(names)
    assert all(re.fullmatch(r"ASSET_\d{2}", n) for n in names)
    assert names == [Masker.fit(panel, seed=7).pseudonym(t) for t in panel.tickers]


def test_pseudonym_does_not_follow_ticker_order(panel):
    # Over a few seeds the map must not be the identity on sorted order (it is a seeded permutation).
    identity = [f"ASSET_{i:02d}" for i in range(len(panel.tickers))]
    maps = [[Masker.fit(panel, seed=s).pseudonym(t) for t in sorted(panel.tickers)] for s in range(5)]
    assert any(m != identity for m in maps)
    assert len({tuple(m) for m in maps}) > 1


def test_masking_shifts_dates_by_whole_weeks_keeping_weekday(panel, masker):
    real, masked = obs_pair(panel, masker)
    deltas = {(m.date - r.date).days for r, m in zip(real.bars, masked.bars, strict=True)}
    assert len(deltas) == 1
    (delta,) = deltas
    assert delta % 7 == 0 and delta != 0
    assert all(m.date.weekday() == r.date.weekday() for r, m in zip(real.bars, masked.bars, strict=True))
    assert masked.as_of == masked.bars[-1].date


def test_masked_dates_never_fall_inside_the_real_panel(panel, masker):
    real_dates = set(panel.dates)
    for i in range(20, 119, 7):
        _, masked = obs_pair(panel, masker, i=i)
        assert not real_dates & {b.date for b in masked.bars}


def test_masking_preserves_returns(panel, masker):
    real, masked = obs_pair(panel, masker)
    r_real = np.diff(np.log([b.close for b in real.bars]))
    r_mask = np.diff(np.log([b.close for b in masked.bars]))
    assert np.allclose(r_real, r_mask, atol=1e-12)
    factor = masked.bars[0].close / real.bars[0].close
    assert 1.5 <= factor <= 6.0


def test_clean_masked_prompt_passes_leak_check(panel, masker):
    checker = LeakChecker(panel)
    for col in range(len(panel.tickers)):
        real, masked = obs_pair(panel, masker, col=col)
        checker.check(render_prompt(masked), real_prices=[b.close for b in real.bars])


def test_leak_check_catches_real_ticker(panel, masker):
    real, masked = obs_pair(panel, masker)
    leaky = render_prompt(masked).replace(masker.pseudonym(real.ticker), real.ticker)
    with pytest.raises(LeakError, match="ticker"):
        LeakChecker(panel).check(leaky, real_prices=[b.close for b in real.bars])


def test_leak_check_catches_real_date(panel, masker):
    real, masked = obs_pair(panel, masker)
    leaky = render_prompt(masked) + f"\nNote: real date {real.as_of.isoformat()}"
    with pytest.raises(LeakError, match="date"):
        LeakChecker(panel).check(leaky, real_prices=[b.close for b in real.bars])


def test_leak_check_catches_real_price(panel, masker):
    real, masked = obs_pair(panel, masker)
    leaky = render_prompt(masked) + f"\nprevious close {real.bars[-1].close:.2f}"
    with pytest.raises(LeakError, match="price"):
        LeakChecker(panel).check(leaky, real_prices=[b.close for b in real.bars])


def test_leak_check_does_not_match_ticker_inside_a_longer_word(panel):
    checker = LeakChecker(panel)
    t = panel.tickers[0]
    checker.check(f"XX{t}XX and {t.lower()} are not the ticker", real_prices=[])


def test_leak_check_does_not_match_price_inside_a_longer_number(panel):
    checker = LeakChecker(panel)
    checker.check("price 112.34 and 12.345", real_prices=[12.34])
    with pytest.raises(LeakError):
        checker.check("price 12.34 here", real_prices=[12.34])
