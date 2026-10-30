"""Prompt rendering and answer parsing for the daily long/short agent."""

from __future__ import annotations

import re
from enum import StrEnum

from .data import RESERVED_WORDS
from .pit import Observation, assert_no_future_rows

__all__ = ["RESERVED_WORDS", "Action", "parse_action", "render_prompt"]


class Action(StrEnum):
    LONG = "LONG"
    SHORT = "SHORT"
    ABSTAIN = "ABSTAIN"


# The template must contain no all-caps word except RESERVED_WORDS and the ticker
# (tests/test_prompt_agent.py pins this), so a real ticker can be found in a masked prompt by a
# whole-word search without false hits from the template text.
_TEMPLATE = (
    "You are a trading agent. Decide the position to hold in {ticker} from the close of "
    "{as_of} to the close of the next trading day.\n"
    "Daily closes for {ticker}, oldest first (date close):\n"
    "{rows}\n"
    "Reply with exactly one word: LONG, SHORT or ABSTAIN."
)


def format_price(x: float) -> str:
    return f"{x:.2f}"


def render_prompt(obs: Observation) -> str:
    # Second line of the as_of guard: even an Observation mutated after construction cannot put a
    # future row into a prompt.
    assert_no_future_rows(obs.bars, obs.as_of)
    rows = "\n".join(f"{b.date.isoformat()} {format_price(b.close)}" for b in obs.bars)
    return _TEMPLATE.format(ticker=obs.ticker, as_of=obs.as_of.isoformat(), rows=rows)


_STRIP = re.compile(r"^[\s*_`'\"]+|[\s*_`'\".!]+$")


def parse_action(text: str) -> Action | None:
    """Exactly one of LONG/SHORT/ABSTAIN, case-insensitive, ignoring surrounding quotes, markdown
    emphasis and a trailing full stop. Anything else is None (the caller counts it and abstains)."""
    word = _STRIP.sub("", text or "").upper()
    try:
        return Action(word)
    except ValueError:
        return None
