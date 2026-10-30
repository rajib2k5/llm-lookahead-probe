"""Model protocol and the daily long/short prompt agent."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from .pit import Observation
from .prompt import Action, parse_action, render_prompt


class Model(Protocol):
    def complete(self, prompt: str) -> str: ...


@dataclass(frozen=True)
class Decision:
    action: Action
    parsed: bool  # False: the answer was not exactly one valid word; action is ABSTAIN
    raw: str
    prompt: str


class DailyLongShortAgent:
    def __init__(self, model: Model) -> None:
        self.model = model

    def decide_detailed(self, obs: Observation, prompt: str | None = None) -> Decision:
        prompt = render_prompt(obs) if prompt is None else prompt
        raw = self.model.complete(prompt)
        action = parse_action(raw)
        if action is None:
            return Decision(Action.ABSTAIN, False, raw, prompt)
        return Decision(action, True, raw, prompt)

    def decide(self, obs: Observation) -> Action:
        return self.decide_detailed(obs).action
