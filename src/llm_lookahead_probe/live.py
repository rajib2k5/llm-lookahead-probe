"""Claude provider, on-disk response cache and cost estimate for live mode.

The `anthropic` SDK is an optional extra (`pip install llm-lookahead-probe[live]`) and is imported
only inside `default_client_factory`, so tests and offline mode never need it.

Model facts and where they come from (checked 2026-10-29):
- Model ID `claude-haiku-5-5`, the cheapest current Claude model, $0.10 input / $0.50 output per
  MTok for prompts of 100K tokens or fewer: claude-api skill, SKILL.md "Current Models (cached:
  2026-10-06)" table and shared/models.md (Claude Haiku 5.5 description); the same figures are in
  the live Models Overview page https://platform.claude.com/docs/en/about-claude/models/overview.md
  ("From $0.10 / input MTok, From $0.50 / output MTok").
- Training data cutoff "Jun 2026" (reliable knowledge cutoff also "Jun 2026"): Models Overview page
  above. The claude-api skill states a cutoff only for Claude Opus 5.5 (shared/model-migration.md:
  "Knowledge cutoff June 2026"), so the Haiku value comes from the live page. We map "Jun 2026" to
  2026-06-01, which counts June as post-cutoff (the conservative side, see docs/DESIGN.md).
- Request shape: claude-api skill, Thinking & Effort table and shared/models.md: on Claude Haiku 5.5
  non-default `temperature`/`top_p`/`top_k` return a 400, so none is sent; adaptive thinking is on
  by default and `thinking: {type: "disabled"}` is accepted at effort `high` or below (the default
  effort is `medium`). We disable thinking so a 16-token cap holds the one-word answer instead of
  being consumed by thinking.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import stat
from collections import Counter
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ModelInfo:
    model_id: str
    input_usd_per_mtok: float
    output_usd_per_mtok: float
    training_cutoff: date
    cutoff_text: str
    request_extra: dict = field(default_factory=dict)


# Only models whose request shape has been checked against the claude-api skill are listed.
MODEL_TABLE: dict[str, ModelInfo] = {
    "claude-haiku-5-5": ModelInfo(
        model_id="claude-haiku-5-5",
        input_usd_per_mtok=0.10,  # source: claude-api skill model table; Models Overview page
        output_usd_per_mtok=0.50,  # source: same
        training_cutoff=date(2026, 6, 1),  # "Jun 2026", Models Overview page (see module doc)
        cutoff_text="Jun 2026 (platform.claude.com Models Overview, training data cutoff)",
        request_extra={"thinking": {"type": "disabled"}},  # source: claude-api skill thinking table
    ),
}
DEFAULT_MODEL = "claude-haiku-5-5"
DEFAULT_MAX_TOKENS = 16

# Conservative token estimate: ~2 characters per token (real prompts of dates and prices tokenise
# at roughly 3 characters per token or more), so the printed estimate is an upper bound.
CHARS_PER_TOKEN_ESTIMATE = 2.0


@dataclass(frozen=True)
class CostEstimate:
    calls: int
    input_tokens: int
    output_tokens: int
    cost_usd: float


def estimate_cost(prompts: Iterable[str], info: ModelInfo, max_tokens: int) -> CostEstimate:
    prompts = list(prompts)
    tin = sum(math.ceil(len(p) / CHARS_PER_TOKEN_ESTIMATE) for p in prompts)
    tout = max_tokens * len(prompts)
    cost = (tin * info.input_usd_per_mtok + tout * info.output_usd_per_mtok) / 1e6
    return CostEstimate(len(prompts), tin, tout, cost)


class ResponseCache:
    """One JSON file per (model, prompt), named by sha256 of both. Directory 0700, files 0600.

    Each entry stores its own key and model; an entry whose stored key does not match the file
    being read (copied, swapped or corrupted) is treated as a miss, never served. Writes are
    atomic (temp file + rename) so a crash cannot leave a half-written entry.
    """

    def __init__(self, root: str | os.PathLike[str]) -> None:
        self.root = Path(root)
        if self.root.is_symlink():
            raise ValueError(f"refusing to use symlinked cache directory {self.root}")
        self.root.mkdir(parents=True, exist_ok=True, mode=0o700)
        os.chmod(self.root, 0o700)

    @staticmethod
    def key(model: str, prompt: str) -> str:
        return hashlib.sha256(json.dumps([model, prompt]).encode()).hexdigest()

    def path_for(self, model: str, prompt: str) -> Path:
        return self.root / f"{self.key(model, prompt)}.json"

    def get(self, model: str, prompt: str) -> dict | None:
        p = self.path_for(model, prompt)
        try:
            if p.is_symlink() or not p.is_file():
                return None
            entry = json.loads(p.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
        ok = (
            isinstance(entry, dict)
            and entry.get("key") == self.key(model, prompt)
            and entry.get("model") == model
            and isinstance(entry.get("text"), str)
            and isinstance(entry.get("input_tokens"), int)
            and isinstance(entry.get("output_tokens"), int)
            and isinstance(entry.get("stop_reason"), str | None)
        )
        return entry if ok else None

    def put(self, model: str, prompt: str, value: dict) -> None:
        entry = {"key": self.key(model, prompt), "model": model, **value}
        final = self.path_for(model, prompt)
        tmp = final.with_suffix(f".tmp{os.getpid()}")
        fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC | os.O_NOFOLLOW, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(entry, fh)
        os.chmod(tmp, stat.S_IRUSR | stat.S_IWUSR)
        os.replace(tmp, final)


class ClaudeModel:
    """`Model` implementation backed by the Messages API, with cache and a hard call budget."""

    def __init__(
        self,
        info: ModelInfo,
        cache: ResponseCache,
        client_getter: Callable[[], Any],
        max_calls: int,
        max_tokens: int = DEFAULT_MAX_TOKENS,
    ) -> None:
        self.info, self.cache, self._get_client = info, cache, client_getter
        self.max_calls, self.max_tokens = max_calls, max_tokens
        self._client: Any = None
        self.calls = self.cache_hits = self.input_tokens = self.output_tokens = 0
        self.stop_reasons: Counter[str] = Counter()

    def complete(self, prompt: str) -> str:
        hit = self.cache.get(self.info.model_id, prompt)
        if hit is not None:
            self.cache_hits += 1
            return hit["text"]
        if self.calls >= self.max_calls:
            raise RuntimeError(f"API call budget of {self.max_calls} calls exhausted; refusing to spend more")
        if self._client is None:
            self._client = self._get_client()
        self.calls += 1
        resp = self._client.messages.create(
            model=self.info.model_id,
            max_tokens=self.max_tokens,
            messages=[{"role": "user", "content": prompt}],
            **self.info.request_extra,
        )
        # A refusal or an empty reply yields "", which the agent counts as unparseable (ABSTAIN).
        text = "".join(b.text for b in resp.content if getattr(b, "type", None) == "text")
        tin, tout = int(resp.usage.input_tokens), int(resp.usage.output_tokens)
        self.input_tokens += tin
        self.output_tokens += tout
        self.stop_reasons[str(resp.stop_reason)] += 1
        self.cache.put(
            self.info.model_id,
            prompt,
            {
                "text": text,
                "input_tokens": tin,
                "output_tokens": tout,
                "stop_reason": resp.stop_reason,
            },
        )
        return text

    def usage(self) -> dict:
        cost = (
            self.input_tokens * self.info.input_usd_per_mtok
            + self.output_tokens * self.info.output_usd_per_mtok
        ) / 1e6
        return {
            "api_calls": self.calls,
            "cache_hits": self.cache_hits,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "cost_usd": cost,
            "stop_reasons": dict(self.stop_reasons),
            "price_usd_per_mtok": {
                "input": self.info.input_usd_per_mtok,
                "output": self.info.output_usd_per_mtok,
            },
            "note": "tokens are summed from API responses of calls made in this run; cache hits cost 0",
        }


def default_client_factory(api_key: str) -> Any:
    try:
        import anthropic  # optional extra: imported only when spending
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise RuntimeError(
            "live mode needs the 'live' extra: pip install 'llm-lookahead-probe[live]'"
        ) from exc
    return anthropic.Anthropic(api_key=api_key, max_retries=2, timeout=60.0)
