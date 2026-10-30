"""Deterministic offline models. No network, no API key.

Both mocks learn the ticker, the decision date and the price path only by parsing the prompt
text, the same way a real model would. All randomness is a hash of (seed, purpose, prompt), so a
masked prompt and its unmasked twin get independent noise: the honest mock therefore produces a
real, non-degenerate null distribution of the gap instead of an exact zero.
"""

from __future__ import annotations

import hashlib
import re
from datetime import date

from .data import PricePanel
from .prompt import Action

_HEAD = re.compile(r"position to hold in (\S+) from the close of (\d{4}-\d{2}-\d{2}) ")
_ROW = re.compile(r"^(\d{4}-\d{2}-\d{2}) (\d+(?:\.\d+)?)$", re.MULTILINE)


def _uniform(seed: int, purpose: str, prompt: str) -> float:
    h = hashlib.sha256(f"{seed}\x00{purpose}\x00{prompt}".encode()).digest()
    return int.from_bytes(h[:8], "big") / 2.0**64


def _flip(a: Action) -> Action:
    return {Action.LONG: Action.SHORT, Action.SHORT: Action.LONG}.get(a, a)


class HonestMock:
    """Base rule: momentum over the shown window (last close vs first close), flipped with
    probability `flip_prob`, and answered in an unparseable sentence with probability
    `garble_prob`. Uses only the price path, so it has no edge and no memory."""

    def __init__(self, seed: int = 0, flip_prob: float = 0.25, garble_prob: float = 0.02) -> None:
        self.seed, self.flip_prob, self.garble_prob = seed, flip_prob, garble_prob

    def base_action(self, prompt: str) -> Action:
        closes = [float(c) for _, c in _ROW.findall(prompt)]
        if len(closes) < 2 or closes[-1] == closes[0]:
            a = Action.ABSTAIN
        else:
            a = Action.LONG if closes[-1] > closes[0] else Action.SHORT
        if _uniform(self.seed, "flip", prompt) < self.flip_prob:
            a = _flip(a)
        return a

    def _say(self, action: Action, prompt: str) -> str:
        if _uniform(self.seed, "garble", prompt) < self.garble_prob:
            return f"I lean {action.value.lower()} here, but it is hard to say."
        return action.value

    def complete(self, prompt: str) -> str:
        return self._say(self.base_action(prompt), prompt)


class MemorisingMock(HonestMock):
    """Has 'memorised' the sign of every next-day return for (real ticker, real date) pairs with
    date < cutoff. When the prompt names a remembered pair it answers the true direction with
    probability p; otherwise (and for anything it does not remember) it uses the honest base rule."""

    def __init__(
        self,
        panel: PricePanel,
        cutoff: date,
        seed: int = 0,
        p: float = 0.75,
        flip_prob: float = 0.25,
        garble_prob: float = 0.02,
    ) -> None:
        super().__init__(seed=seed, flip_prob=flip_prob, garble_prob=garble_prob)
        self.p = p
        self.memory: dict[tuple[str, str], Action] = {}
        for i, d in enumerate(panel.dates[:-1]):
            if d >= cutoff:
                break
            for j, t in enumerate(panel.tickers):
                r = panel.closes[i + 1, j] / panel.closes[i, j] - 1.0
                if r != 0:
                    self.memory[(t, d.isoformat())] = Action.LONG if r > 0 else Action.SHORT

    def complete(self, prompt: str) -> str:
        m = _HEAD.search(prompt)
        remembered = self.memory.get((m.group(1), m.group(2))) if m else None
        if remembered is not None and _uniform(self.seed, "recall", prompt) < self.p:
            return self._say(remembered, prompt)
        return super().complete(prompt)
