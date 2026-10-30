import re
from datetime import date

import pytest

from llm_lookahead_probe.agent import DailyLongShortAgent
from llm_lookahead_probe.pit import Bar, Observation
from llm_lookahead_probe.prompt import RESERVED_WORDS, Action, parse_action, render_prompt

OBS = Observation(
    ticker="AAA",
    as_of=date(2024, 1, 4),
    bars=(Bar(date(2024, 1, 2), 10.0), Bar(date(2024, 1, 3), 10.5), Bar(date(2024, 1, 4), 10.25)),
)


class Echo:
    def __init__(self, reply):
        self.reply = reply
        self.prompts = []

    def complete(self, prompt: str) -> str:
        self.prompts.append(prompt)
        return self.reply


def test_prompt_shows_ticker_decision_date_and_dated_closes():
    p = render_prompt(OBS)
    assert "AAA" in p
    assert "2024-01-04" in p
    for line in ("2024-01-02 10.00", "2024-01-03 10.50", "2024-01-04 10.25"):
        assert line in p
    assert "exactly one word" in p


def test_template_has_no_uppercase_words_except_reserved_and_ticker():
    words = set(re.findall(r"\b[A-Z][A-Z0-9.\-]*\b", render_prompt(OBS)))
    assert words - {"AAA"} <= RESERVED_WORDS


@pytest.mark.parametrize(
    "text, expected",
    [
        ("LONG", Action.LONG),
        ("short", Action.SHORT),
        (" Long.\n", Action.LONG),
        ("ABSTAIN", Action.ABSTAIN),
        ("**SHORT**", Action.SHORT),
        ("I would go LONG", None),
        ("LONG SHORT", None),
        ("", None),
        ("BUY", None),
    ],
)
def test_parse_action(text, expected):
    assert parse_action(text) == expected


def test_agent_returns_parsed_action():
    model = Echo("SHORT")
    agent = DailyLongShortAgent(model)
    assert agent.decide(OBS) == Action.SHORT
    assert model.prompts == [render_prompt(OBS)]


def test_agent_turns_unparseable_answer_into_counted_abstain():
    d = DailyLongShortAgent(Echo("Probably long, but unsure")).decide_detailed(OBS)
    assert d.action == Action.ABSTAIN
    assert d.parsed is False
    assert d.raw == "Probably long, but unsure"


def test_explicit_abstain_is_parsed_not_a_failure():
    d = DailyLongShortAgent(Echo("ABSTAIN")).decide_detailed(OBS)
    assert d.action == Action.ABSTAIN and d.parsed is True
