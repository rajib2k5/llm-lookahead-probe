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
- Request shape: claude-api skill, shared/models.md (Claude Haiku 5.5 entry): non-default
  `temperature`/`top_p`/`top_k` return a 400, so none is sent; adaptive thinking is on by default
  and `thinking: {type: "disabled"}` is accepted only at effort `high` or below; `effort` runs from
  `low` to `max` with default `medium`. We disable thinking so a 16-token cap holds the one-word
  answer instead of being consumed by thinking.
- Effort is pinned to `medium` with `output_config={"effort": "medium"}` rather than left to the
  server default, so a future change of the default to `xhigh`/`max` cannot turn every call into
  a 400. Sources: the parameter shape is python/claude-api/README.md (Extended Thinking example,
  `output_config={"effort": ...}`); shared/prompt-caching.md states that setting a model's default
  effort explicitly is equivalent to omitting it. anthropic 2.3.0's `Messages.create` accepts
  `output_config` (checked with inspect.signature). Not checked against the live API.
"""

from __future__ import annotations

import hashlib
import json
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
        # source: claude-api skill shared/models.md (Haiku 5.5) and python README (output_config)
        request_extra={"thinking": {"type": "disabled"}, "output_config": {"effort": "medium"}},
    ),
}
DEFAULT_MODEL = "claude-haiku-5-5"
DEFAULT_MAX_TOKENS = 16

# Cost estimate, worst case per request (see estimate_cost for the argument):
#   input tokens  <= UTF-8 bytes of the prompt + PER_REQUEST_OVERHEAD_TOKENS
#   output tokens <= max_tokens
# PER_REQUEST_OVERHEAD_TOKENS is an allowance for the tokens the API adds around one user message
# (role and turn markers). It is an assumption, NOT a measured value: no live call has been made.
# report.json records the real input_tokens the API returns, which is the check on this number.
PER_REQUEST_OVERHEAD_TOKENS = 64

# SDK retries (connection errors, 408/409/429/5xx) are HTTP requests the call budget cannot see,
# and a retried request may be billed. We set them to 0: every request is a counted call, and a
# run that fails mid-way is resumed for free from the cache. estimate_cost multiplies by
# (1 + max_retries) so the estimate stays an upper bound if this is ever raised.
MAX_RETRIES = 0
DEFAULT_BASE_URL = "https://api.anthropic.com"


def resolve_base_url() -> str:
    """The API endpoint for this run: ANTHROPIC_BASE_URL if set and non-empty, else the public API.
    default_client_factory passes this value to the SDK explicitly, and the CLI puts the same value
    in the cache key, so the endpoint that answered and the endpoint in the key cannot differ (an
    SDK profile's base URL is therefore not used)."""
    return os.environ.get("ANTHROPIC_BASE_URL") or DEFAULT_BASE_URL


class LiveAPIError(RuntimeError):
    """An API request failed (rate limit, overload, server or connection error). With MAX_RETRIES
    = 0 the SDK does not retry, so the run stops here. Every answer received before the failure is
    already in the cache (each one is written as soon as it arrives), so re-running the same
    command resumes: cached prompts are not sent again and cost nothing."""

    def __init__(self, calls_made: int, cause: BaseException) -> None:
        self.calls_made = calls_made
        super().__init__(f"{type(cause).__name__}: {cause}")


def _api_error_types() -> tuple[type[BaseException], ...]:
    """The SDK's own error base class, if the SDK is installed. Only these are turned into a
    resumable LiveAPIError; anything else (a bug, a bad argument) still surfaces as itself."""
    try:
        import anthropic
    except ImportError:  # pragma: no cover - depends on environment
        return ()
    return (anthropic.APIError,)


@dataclass(frozen=True)
class CostEstimate:
    calls: int
    input_tokens: int
    output_tokens: int
    cost_usd: float


def estimate_cost(
    prompts: Iterable[str], info: ModelInfo, max_tokens: int, max_retries: int = MAX_RETRIES
) -> CostEstimate:
    """Worst-case cost of sending `prompts`, one request each.

    Argument for the bound. Input: Claude's tokenizer is a byte-level BPE, so every token covers
    at least one byte of the text and a prompt of n UTF-8 bytes is at most n tokens (for the ASCII
    prompts this probe renders, bytes = characters; real prompts of dates and prices tokenise at
    about 3 characters per token, so the bound is loose by roughly 3x). To that we add
    PER_REQUEST_OVERHEAD_TOKENS for message framing, an unverified allowance. Output: the API
    stops generating at `max_tokens`, and thinking is disabled, so output tokens <= max_tokens.
    Retries: each request may be sent up to 1 + max_retries times, and we assume every attempt is
    billed in full.
    """
    prompts = list(prompts)
    attempts = 1 + max_retries
    tin = attempts * sum(len(p.encode("utf-8")) + PER_REQUEST_OVERHEAD_TOKENS for p in prompts)
    tout = attempts * max_tokens * len(prompts)
    cost = (tin * info.input_usd_per_mtok + tout * info.output_usd_per_mtok) / 1e6
    return CostEstimate(len(prompts), tin, tout, cost)


def build_request(info: ModelInfo, prompt: str, max_tokens: int) -> dict:
    """The exact keyword arguments sent to `messages.create`. The cache key is a hash of this whole
    dict, so a change to any request parameter (model, max_tokens, thinking, effort, prompt) is a
    cache miss rather than a stale answer produced under another configuration."""
    return {
        "model": info.model_id,
        "max_tokens": max_tokens,
        "messages": [{"role": "user", "content": prompt}],
        **info.request_extra,
    }


class ResponseCache:
    """One JSON file per request, named by sha256 of the canonical JSON of the full request.

    What the stored key protects against: an entry file that was copied or renamed to another
    request's name, swapped, truncated or corrupted is treated as a miss, never served. What it
    does NOT protect against: deliberate tampering. The key is a plain hash of the request, not a
    MAC, so anyone who can write to the cache directory can edit an entry's `text` and it will be
    served (tests/test_live.py pins this). Whoever controls the cache directory controls the
    answers the probe scores; keep it private.

    Directory handling: a directory this class creates gets mode 0700. An existing directory is
    used as it is (its mode is never changed, so `--cache-dir ~` cannot alter the home directory),
    but one that is owned by another user, or group- or world-writable, is refused, because
    another user could plant answers in it (e.g. a /tmp directory someone else pre-created at
    0755). The cache key covers the API endpoint as well as the request. With `create=False` a
    missing directory is not created: every lookup misses and `put` refuses (used by the live dry
    run, which must not touch the disk).

    Files are 0600 and written atomically (temp file + rename), so a crash cannot leave a
    half-written entry.
    """

    def __init__(
        self, root: str | os.PathLike[str], *, create: bool = True, endpoint: str = DEFAULT_BASE_URL
    ) -> None:
        self.root = Path(root)
        self.endpoint = endpoint
        if self.root.is_symlink():
            raise ValueError(f"refusing to use symlinked cache directory {self.root}")
        self._exists = self.root.is_dir()
        if self._exists:
            st = self.root.stat()
            if st.st_uid != os.getuid():
                raise ValueError(
                    f"refusing to use cache directory {self.root}: it is owned by uid {st.st_uid}, "
                    f"not by you (uid {os.getuid()}); its owner could plant answers. Use a private "
                    "directory you own"
                )
            mode = stat.S_IMODE(st.st_mode)
            if mode & (stat.S_IWGRP | stat.S_IWOTH):
                raise ValueError(
                    f"refusing to use cache directory {self.root}: it is group- or world-writable "
                    f"(mode {mode:04o}), so another user could plant answers; use a private directory"
                )
        elif self.root.exists():
            raise ValueError(f"cache path {self.root} exists and is not a directory")
        elif create:
            self.root.parent.mkdir(parents=True, exist_ok=True)
            self.root.mkdir(mode=0o700)
            os.chmod(self.root, 0o700)  # mkdir's mode is filtered by the umask; we created it
            self._exists = True

    def key(self, request: dict) -> str:
        """sha256 of the request AND the endpoint, so answers from a local stub server are never
        served on a run against the real API (or the other way round)."""
        keyed = {"endpoint": self.endpoint, "request": request}
        canonical = json.dumps(keyed, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
        return hashlib.sha256(canonical.encode()).hexdigest()

    def path_for(self, request: dict) -> Path:
        return self.root / f"{self.key(request)}.json"

    def get(self, request: dict) -> dict | None:
        if not self._exists:
            return None
        p = self.path_for(request)
        try:
            if p.is_symlink() or not p.is_file():
                return None
            entry = json.loads(p.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
        ok = (
            isinstance(entry, dict)
            and entry.get("key") == self.key(request)
            and entry.get("model") == request.get("model")
            and isinstance(entry.get("text"), str)
            and isinstance(entry.get("input_tokens"), int)
            and isinstance(entry.get("output_tokens"), int)
            and isinstance(entry.get("stop_reason"), str | None)
        )
        return entry if ok else None

    def put(self, request: dict, value: dict) -> None:
        if not self._exists:
            raise RuntimeError(f"cache directory {self.root} was opened read-only and does not exist")
        entry = {"key": self.key(request), "model": request.get("model"), **value}
        final = self.path_for(request)
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
        request = build_request(self.info, prompt, self.max_tokens)
        hit = self.cache.get(request)
        if hit is not None:
            self.cache_hits += 1
            return hit["text"]
        if self.calls >= self.max_calls:
            raise RuntimeError(f"API call budget of {self.max_calls} calls exhausted; refusing to spend more")
        if self._client is None:
            self._client = self._get_client()
        self.calls += 1  # counted before sending: a failed request may still have been billed
        try:
            resp = self._client.messages.create(**request)
        except _api_error_types() as exc:
            raise LiveAPIError(self.calls - 1, exc) from exc
        # A refusal or an empty reply yields "", which the agent counts as unparseable (ABSTAIN).
        text = "".join(b.text for b in resp.content if getattr(b, "type", None) == "text")
        tin, tout = int(resp.usage.input_tokens), int(resp.usage.output_tokens)
        self.input_tokens += tin
        self.output_tokens += tout
        self.stop_reasons[str(resp.stop_reason)] += 1
        self.cache.put(
            request,
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
    return anthropic.Anthropic(
        api_key=api_key, base_url=resolve_base_url(), max_retries=MAX_RETRIES, timeout=60.0
    )
