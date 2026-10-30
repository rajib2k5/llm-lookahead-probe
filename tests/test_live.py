"""Live mode with a fake Anthropic client: no network, no real key."""

import dataclasses
import json
import os
import stat
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest

from llm_lookahead_probe.cli import main
from llm_lookahead_probe.data import synthetic_panel
from llm_lookahead_probe.live import (
    MAX_RETRIES,
    MODEL_TABLE,
    PER_REQUEST_OVERHEAD_TOKENS,
    ClaudeModel,
    ResponseCache,
    default_client_factory,
    estimate_cost,
)

SECRET = "sk-ant-test-DO-NOT-PRINT-0123456789"


class FakeClient:
    def __init__(self, answer="LONG", fail_with=None):
        self.calls = []
        self.answer = answer
        self.fail_with = fail_with
        self.messages = self

    def create(self, **kw):
        if self.fail_with:
            raise self.fail_with
        self.calls.append(kw)
        return SimpleNamespace(
            content=[SimpleNamespace(type="text", text=self.answer)],
            usage=SimpleNamespace(input_tokens=180, output_tokens=2),
            stop_reason="end_turn",
        )


@pytest.fixture
def prices_csv(tmp_path):
    panel = synthetic_panel(seed=11, n_tickers=4, n_days=80)
    lines = ["date,ticker,close"]
    order = sorted(range(len(panel.tickers)), key=lambda j: panel.tickers[j])
    for i, d in enumerate(panel.dates):
        for j in order:
            lines.append(f"{d.isoformat()},{panel.tickers[j]},{panel.closes[i, j]:.2f}")
    p = tmp_path / "prices.csv"
    p.write_text("\n".join(lines) + "\n")
    return p, panel


@pytest.fixture
def env_key(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", SECRET)


def live_args(tmp_path, prices, *extra):
    return [
        "run",
        "--live",
        "--model",
        "claude-haiku-5-5",
        "--prices",
        str(prices),
        "--n-tickers",
        "3",
        "--n-dates",
        "4",
        "--window",
        "10",
        "--cutoff",
        "2024-03-01",
        "--out",
        str(tmp_path / "out"),
        "--cache-dir",
        str(tmp_path / "cache"),
        *extra,
    ]


def test_no_call_without_yes_and_estimate_is_printed(tmp_path, prices_csv, env_key, capsys):
    client = FakeClient()
    rc = main(live_args(tmp_path, prices_csv[0]), client_factory=lambda key: client)
    out = capsys.readouterr().out
    assert rc == 0
    assert client.calls == []
    assert "Estimated cost" in out
    assert "24 uncached calls" in out  # 3 tickers x 4 dates x 2 arms
    assert "--yes" in out
    assert not (tmp_path / "out" / "report.json").exists()


def test_cap_is_enforced_before_any_call(tmp_path, prices_csv, env_key, capsys):
    client = FakeClient()
    rc = main(
        live_args(tmp_path, prices_csv[0], "--max-episodes", "5", "--yes"),
        client_factory=lambda key: client,
    )
    err = capsys.readouterr().err
    assert rc == 2
    assert client.calls == []
    assert "exceeds --max-episodes 5" in err


def test_yes_spends_records_usage_and_rerun_uses_cache(tmp_path, prices_csv, env_key, capsys):
    first = FakeClient()
    assert main(live_args(tmp_path, prices_csv[0], "--yes"), client_factory=lambda key: first) == 0
    assert len(first.calls) == 24
    call = first.calls[0]
    assert call["model"] == "claude-haiku-5-5"
    assert call["max_tokens"] <= 32
    assert not {"temperature", "top_p", "top_k"} & set(call)  # rejected by current models
    # Thinking off is accepted only at effort high or below, so effort is pinned, not left to
    # the server default.
    assert call["thinking"] == {"type": "disabled"}
    assert call["output_config"] == {"effort": "medium"}
    rep = json.loads((tmp_path / "out" / "report.json").read_text())
    assert rep["usage"]["api_calls"] == 24
    assert rep["usage"]["input_tokens"] == 24 * 180
    assert rep["usage"]["output_tokens"] == 24 * 2
    assert rep["usage"]["cost_usd"] == pytest.approx((24 * 180 * 0.10 + 24 * 2 * 0.50) / 1e6)

    second = FakeClient()
    capsys.readouterr()
    assert main(live_args(tmp_path, prices_csv[0], "--yes"), client_factory=lambda key: second) == 0
    assert second.calls == []
    assert "0 uncached calls" in capsys.readouterr().out
    rep2 = json.loads((tmp_path / "out" / "report.json").read_text())
    assert rep2["usage"]["api_calls"] == 0 and rep2["usage"]["cache_hits"] == 24
    assert rep2["gap"] == rep["gap"]


def test_yes_without_api_key_refuses(tmp_path, prices_csv, monkeypatch, capsys):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    client = FakeClient()
    rc = main(live_args(tmp_path, prices_csv[0], "--yes"), client_factory=lambda key: client)
    assert rc == 2 and client.calls == []
    assert "ANTHROPIC_API_KEY" in capsys.readouterr().err


def test_live_requires_live_flag(tmp_path, prices_csv, env_key, capsys):
    args = [a for a in live_args(tmp_path, prices_csv[0], "--yes") if a != "--live"]
    client = FakeClient()
    with pytest.raises(SystemExit):
        main(args, client_factory=lambda key: client)
    assert client.calls == []


def test_api_key_is_never_printed_or_written(tmp_path, prices_csv, env_key, capsys):
    boom = FakeClient(fail_with=RuntimeError(f"auth failed for key {SECRET}"))
    rc = main(live_args(tmp_path, prices_csv[0], "--yes"), client_factory=lambda key: boom)
    out = capsys.readouterr()
    assert rc == 1
    assert SECRET not in out.out and SECRET not in out.err
    assert "[REDACTED]" in out.err
    ok = FakeClient()
    main(live_args(tmp_path, prices_csv[0], "--yes"), client_factory=lambda key: ok)
    out = capsys.readouterr()
    assert SECRET not in out.out + out.err
    for root, _, files in os.walk(tmp_path):
        for f in files:
            assert SECRET not in Path(root, f).read_text(errors="ignore")


VALUE = {"text": "LONG", "input_tokens": 1, "output_tokens": 1, "stop_reason": "end_turn"}


def req(prompt="prompt", model="m", max_tokens=16, **extra):
    messages = [{"role": "user", "content": prompt}]
    return {"model": model, "max_tokens": max_tokens, "messages": messages, **extra}


def test_cache_files_are_private_and_copied_or_corrupt_entries_are_ignored(tmp_path):
    cache = ResponseCache(tmp_path / "c")
    cache.put(req(), VALUE)
    assert stat.S_IMODE((tmp_path / "c").stat().st_mode) == 0o700
    (entry,) = list((tmp_path / "c").iterdir())
    assert stat.S_IMODE(entry.stat().st_mode) == 0o600
    assert cache.get(req())["text"] == "LONG"
    assert cache.get(req(model="other-model")) is None
    # An entry copied under another request's key must not be served.
    other = cache.path_for(req("different prompt"))
    other.write_text(entry.read_text())
    assert cache.get(req("different prompt")) is None
    entry.write_text("{not json")
    assert cache.get(req()) is None


def test_cache_is_not_tamper_proof_an_edited_answer_is_served(tmp_path):
    """Pins a limit, not a guarantee: the stored key is a hash of the request, not a MAC. Anyone
    who can write to the cache directory can change the answers the probe scores."""
    cache = ResponseCache(tmp_path / "c")
    cache.put(req(), VALUE)
    p = cache.path_for(req())
    p.write_text(p.read_text().replace('"LONG"', '"SHORT"'))
    assert cache.get(req())["text"] == "SHORT"


def test_cache_does_not_change_the_mode_of_an_existing_directory(tmp_path):
    d = tmp_path / "home-like"
    d.mkdir()
    os.chmod(d, 0o755)
    ResponseCache(d).put(req(), VALUE)
    assert stat.S_IMODE(d.stat().st_mode) == 0o755


def test_cache_creates_missing_directories_private(tmp_path):
    d = tmp_path / "a" / "b" / "cache"
    ResponseCache(d)
    assert stat.S_IMODE(d.stat().st_mode) == 0o700


@pytest.mark.parametrize("mode", [0o777, 0o770, 0o722, 0o702])
def test_cache_refuses_a_group_or_world_writable_existing_directory(tmp_path, mode):
    d = tmp_path / "shared"
    d.mkdir()
    os.chmod(d, mode)
    with pytest.raises(ValueError, match="writable"):
        ResponseCache(d)
    assert stat.S_IMODE(d.stat().st_mode) == mode


def test_cache_key_covers_every_request_parameter(tmp_path):
    cache = ResponseCache(tmp_path / "c")
    cache.put(req(), VALUE)
    assert cache.get(req()) is not None
    assert cache.get(req(max_tokens=32)) is None
    assert cache.get(req(thinking={"type": "disabled"})) is None
    assert cache.get(req(output_config={"effort": "high"})) is None


def test_changing_max_tokens_makes_a_new_call_not_a_stale_cache_hit(tmp_path):
    info = MODEL_TABLE["claude-haiku-5-5"]
    cache = ResponseCache(tmp_path / "c")
    first, second = FakeClient("LONG"), FakeClient("SHORT")
    ClaudeModel(info, cache, lambda: first, max_calls=5, max_tokens=16).complete("x")
    out = ClaudeModel(info, cache, lambda: second, max_calls=5, max_tokens=32).complete("x")
    assert len(second.calls) == 1 and out == "SHORT"


def test_changing_request_extras_makes_a_new_call(tmp_path):
    info = MODEL_TABLE["claude-haiku-5-5"]
    other = dataclasses.replace(info, request_extra={**info.request_extra, "thinking": {"type": "adaptive"}})
    cache = ResponseCache(tmp_path / "c")
    first, second = FakeClient("LONG"), FakeClient("SHORT")
    ClaudeModel(info, cache, lambda: first, max_calls=5).complete("x")
    assert ClaudeModel(other, cache, lambda: second, max_calls=5).complete("x") == "SHORT"


def test_dry_run_does_not_create_the_cache_directory(tmp_path, prices_csv, env_key, capsys):
    client = FakeClient()
    assert main(live_args(tmp_path, prices_csv[0]), client_factory=lambda key: client) == 0
    assert not (tmp_path / "cache").exists()


def test_model_refuses_calls_beyond_its_budget(tmp_path):
    client = FakeClient()
    m = ClaudeModel(
        MODEL_TABLE["claude-haiku-5-5"], ResponseCache(tmp_path / "c"), lambda: client, max_calls=2
    )
    m.complete("a")
    m.complete("b")
    m.complete("a")  # cached: free
    with pytest.raises(RuntimeError, match="budget"):
        m.complete("c")
    assert len(client.calls) == 2


def test_refusal_becomes_empty_text_and_is_counted(tmp_path):
    client = FakeClient()
    client.create = lambda **kw: SimpleNamespace(
        content=[], usage=SimpleNamespace(input_tokens=5, output_tokens=0), stop_reason="refusal"
    )
    m = ClaudeModel(
        MODEL_TABLE["claude-haiku-5-5"], ResponseCache(tmp_path / "c"), lambda: client, max_calls=5
    )
    assert m.complete("x") == ""
    assert m.usage()["stop_reasons"] == {"refusal": 1}


def test_input_estimate_is_utf8_bytes_plus_a_per_request_overhead():
    info = MODEL_TABLE["claude-haiku-5-5"]
    prompts = ["x" * 600, "\u00e9" * 10, ""]  # 600 bytes, 20 bytes (10 two-byte chars), 0 bytes
    est = estimate_cost(prompts, info, max_tokens=16, max_retries=0)
    assert est.input_tokens == 600 + 20 + 0 + 3 * PER_REQUEST_OVERHEAD_TOKENS
    assert est.output_tokens == 3 * 16
    assert est.calls == 3


def test_estimate_covers_sdk_retries_as_extra_billed_attempts():
    info = MODEL_TABLE["claude-haiku-5-5"]
    once = estimate_cost(["abc"] * 4, info, max_tokens=16, max_retries=0)
    thrice = estimate_cost(["abc"] * 4, info, max_tokens=16, max_retries=2)
    assert thrice.input_tokens == 3 * once.input_tokens
    assert thrice.output_tokens == 3 * once.output_tokens
    assert thrice.cost_usd == pytest.approx(3 * once.cost_usd)


def test_client_is_built_with_the_retry_count_the_estimate_assumes(monkeypatch):
    seen = {}
    fake = types.ModuleType("anthropic")
    fake.Anthropic = lambda **kw: seen.update(kw) or object()
    monkeypatch.setitem(sys.modules, "anthropic", fake)
    default_client_factory("k")
    assert seen["max_retries"] == MAX_RETRIES == 0


def test_estimate_is_printed_before_the_first_call(tmp_path, prices_csv, env_key, capsys):
    class OrderChecking(FakeClient):
        seen_estimate_first = None

        def create(self, **kw):
            if self.seen_estimate_first is None:
                self.seen_estimate_first = "Estimated cost" in capsys.readouterr().out
            return super().create(**kw)

    client = OrderChecking()
    assert main(live_args(tmp_path, prices_csv[0], "--yes"), client_factory=lambda key: client) == 0
    assert client.seen_estimate_first is True
