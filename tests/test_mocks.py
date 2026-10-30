from datetime import date

import numpy as np
import pytest

from llm_lookahead_probe.data import synthetic_panel
from llm_lookahead_probe.masking import Masker
from llm_lookahead_probe.mocks import HonestMock, MemorisingMock
from llm_lookahead_probe.pit import build_observation
from llm_lookahead_probe.prompt import Action, parse_action, render_prompt

CUTOFF = date(2024, 7, 1)


@pytest.fixture(scope="module")
def panel():
    return synthetic_panel(seed=5, n_tickers=10, n_days=260)


def prompts(panel, masker, idx):
    out = []
    for i in idx:
        for t in panel.tickers:
            real = build_observation(panel, t, panel.dates[i], window=20)
            out.append((t, panel.dates[i], render_prompt(real), render_prompt(masker.mask(real))))
    return out


def hit_rate(panel, model, items, which):
    hits = n = 0
    for t, d, pu, pm in items:
        a = parse_action(model.complete(pu if which == "u" else pm))
        if a in (Action.LONG, Action.SHORT):
            r = panel.next_day_return(t, d)
            hits += (a == Action.LONG) == (r > 0)
            n += 1
    return hits / n


def test_mocks_are_deterministic(panel):
    m = Masker.fit(panel, seed=1)
    ((_, _, pu, _),) = prompts(panel, m, [50])[:1]
    for model in (HonestMock(seed=3), MemorisingMock(panel, CUTOFF, seed=3)):
        assert model.complete(pu) == model.complete(pu)


def test_honest_mock_noise_differs_between_masked_and_unmasked_prompts(panel):
    m = Masker.fit(panel, seed=1)
    model = HonestMock(seed=0)
    items = prompts(panel, m, range(30, 230, 5))
    agree = np.mean([model.complete(pu) == model.complete(pm) for _, _, pu, pm in items])
    assert 0.4 < agree < 0.9  # a real null: arms disagree on a sizeable share of episodes


def test_honest_mock_has_no_edge_in_either_arm(panel):
    m = Masker.fit(panel, seed=1)
    items = prompts(panel, m, range(30, 230, 2))
    model = HonestMock(seed=0)
    assert abs(hit_rate(panel, model, items, "u") - 0.5) < 0.05
    assert abs(hit_rate(panel, model, items, "m") - 0.5) < 0.05


def test_memorising_mock_recalls_only_unmasked_pre_cutoff_episodes(panel):
    m = Masker.fit(panel, seed=1)
    model = MemorisingMock(panel, CUTOFF, seed=0, p=0.75)
    pre_idx = [i for i in range(30, 259, 2) if panel.dates[i] < CUTOFF]
    post_idx = [i for i in range(30, 259, 2) if panel.dates[i] >= CUTOFF]
    pre, post = prompts(panel, m, pre_idx), prompts(panel, m, post_idx)
    assert hit_rate(panel, model, pre, "u") > 0.8  # 0.75 recall + half of the rest by chance
    assert abs(hit_rate(panel, model, pre, "m") - 0.5) < 0.06  # masked: nothing to look up
    assert abs(hit_rate(panel, model, post, "u") - 0.5) < 0.06  # after cutoff: nothing remembered


def test_memorising_mock_learns_identity_only_from_prompt_text(panel):
    model = MemorisingMock(panel, CUTOFF, seed=0, p=1.0, garble_prob=0.0)
    t, d = panel.tickers[0], panel.dates[40]
    real = build_observation(panel, t, d, window=20)
    truth = Action.LONG if panel.next_day_return(t, d) > 0 else Action.SHORT
    assert parse_action(model.complete(render_prompt(real))) == truth
    # Same text with the ticker renamed: the mock can no longer look it up.
    renamed = render_prompt(real).replace(t, "ZZZZ")
    base = HonestMock(seed=0, garble_prob=0.0)
    assert model.complete(renamed) == base.complete(renamed)


def test_mocks_sometimes_produce_unparseable_answers(panel):
    m = Masker.fit(panel, seed=1)
    model = HonestMock(seed=0, garble_prob=0.05)
    items = prompts(panel, m, range(30, 230, 2))
    bad = sum(parse_action(model.complete(pu)) is None for _, _, pu, _ in items)
    assert 0 < bad < len(items) * 0.1
