"""Command line: `llm-lookahead-probe run --offline ...` or `run --live ...`.

Exit codes: 0 success (including a live dry run without --yes), 1 runtime failure, 2 refused
input (validation, cap, missing key, unsafe output path).

Spending rules for live mode: nothing is sent unless --live, --yes and ANTHROPIC_API_KEY are all
present; the planned episode count must not exceed --max-episodes; every masked prompt passes the
leak check and the cost estimate is printed before the first call. The API key is read from the
environment only, passed straight to the client factory, and redacted from any error text.
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import date
from pathlib import Path

from .data import PriceDataError, load_prices_csv
from .live import (
    DEFAULT_MAX_TOKENS,
    DEFAULT_MODEL,
    MODEL_TABLE,
    ClaudeModel,
    ResponseCache,
    build_request,
    default_client_factory,
    estimate_cost,
)
from .masking import LeakError, Masker
from .probe import (
    OFFLINE_CUTOFF,
    OfflineConfig,
    execute,
    plan_episodes,
    run_offline,
    sample_episodes,
)
from .report import build_report, write_reports
from .stats import summarise

OFFLINE_DEFAULTS = {"n_tickers": 20, "n_dates": 120}
LIVE_DEFAULTS = {"n_tickers": 5, "n_dates": 10}
DEFAULT_MAX_EPISODES = 50


def _iso_date(s: str) -> date:
    try:
        return date.fromisoformat(s)
    except ValueError:
        raise argparse.ArgumentTypeError(f"expected YYYY-MM-DD, got {s!r}") from None


def _positive_int(s: str) -> int:
    v = int(s)
    if v < 1:
        raise argparse.ArgumentTypeError("must be >= 1")
    return v


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="llm-lookahead-probe",
        description="Measure how much an LLM trading agent's backtest is inflated by memorised history.",
    )
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="run the masked vs unmasked probe")
    mode = r.add_mutually_exclusive_group(required=True)
    mode.add_argument("--offline", action="store_true", help="synthetic data and a mock model")
    mode.add_argument("--live", action="store_true", help="real Claude calls (costs money; needs --yes)")
    r.add_argument("--mock", choices=["memorising", "honest"], help="offline mock model")
    r.add_argument("--model", default=DEFAULT_MODEL, help=f"live model id (default {DEFAULT_MODEL})")
    r.add_argument("--prices", help="live mode: CSV with columns date,ticker,close")
    r.add_argument("--seed", type=int, default=0)
    r.add_argument("--n-tickers", type=_positive_int, help="offline default 20, live default 5")
    r.add_argument("--n-dates", type=_positive_int, help="offline default 120, live default 10")
    r.add_argument("--window", type=_positive_int, default=20, help="closes shown per prompt")
    r.add_argument("--cutoff", type=_iso_date, help="training cutoff used to split episodes")
    r.add_argument("--bootstrap", type=_positive_int, default=2000, help="resamples (>= 2000)")
    r.add_argument("--out", default="reports/latest", help="output directory")
    r.add_argument(
        "--max-episodes",
        type=_positive_int,
        default=DEFAULT_MAX_EPISODES,
        help=f"live: hard cap on episodes (default {DEFAULT_MAX_EPISODES})",
    )
    r.add_argument("--yes", action="store_true", help="live: allow spending")
    r.add_argument("--cache-dir", default=".cache/llm-lookahead-probe", help="live: response cache")
    return p


def _err(msg: str) -> None:
    print(f"error: {msg}", file=sys.stderr)


def _redact(text: str) -> str:
    key = os.environ.get("ANTHROPIC_API_KEY")
    return text.replace(key, "[REDACTED]") if key else text


def _pct(x: float | None) -> str:
    return "n/a" if x is None else f"{x * 100:+.4f}%"


def _ci(c: dict | None) -> str:
    return "n/a" if not c else f"{_pct(c['estimate'])} [{_pct(c['low'])}, {_pct(c['high'])}]"


def print_headline(rep: dict, paths) -> None:
    u, m = rep["arms"]["unmasked"], rep["arms"]["masked"]
    hr = lambda a: "n/a" if a["hit_rate"] is None else f"{a['hit_rate']:.3f}"  # noqa: E731
    print(f"Verdict: {rep['verdict']}")
    print(f"  {rep['verdict_reason']}")
    print(f"Episodes: {rep['n_episodes']} on {rep['n_dates']} dates; cutoff {rep['cutoff']}")
    print(f"Gap (unmasked - masked, mean pnl/episode, 95% t-interval): {_ci(rep['gap'])}")
    for name, label in (("pre", "pre-cutoff "), ("post", "post-cutoff")):
        part = rep[name]
        scaled = part.get("gap_per_mean_abs_r")
        extra = "" if scaled is None else f"; gap / mean|r| {scaled:+.4f}"
        print(f"  {label} ({part['n_dates']} dates): {_ci(part['gap'])}{extra}")
    print(f"  pre - post: {_ci(rep['diff_pre_minus_post'])}")
    print(f"Unmasked: mean pnl {_pct(u['mean_pnl'])}, hit rate {hr(u)}, unparseable {u['n_unparseable']}")
    print(f"Masked:   mean pnl {_pct(m['mean_pnl'])}, hit rate {hr(m)}, unparseable {m['n_unparseable']}")
    if "usage" in rep:
        us = rep["usage"]
        print(
            f"API: {us['api_calls']} calls, {us['cache_hits']} cache hits, "
            f"{us['input_tokens']} in / {us['output_tokens']} out tokens, ${us['cost_usd']:.6f}"
        )
    print(f"Wrote {paths[0]} and {paths[1]}")


def _run_offline(a) -> int:
    if not a.mock:
        _err("--offline needs --mock memorising|honest")
        return 2
    cfg = OfflineConfig(
        seed=a.seed,
        n_tickers=a.n_tickers or OFFLINE_DEFAULTS["n_tickers"],
        n_dates=a.n_dates or OFFLINE_DEFAULTS["n_dates"],
        window=a.window,
        cutoff=a.cutoff or OFFLINE_CUTOFF,
        n_resamples=a.bootstrap,
    )
    rep = run_offline(a.mock, cfg)
    paths = write_reports(rep, a.out)
    print_headline(rep, paths)
    return 0


def _run_live(a, client_factory) -> int:
    if a.mock:
        _err("--mock is for --offline runs")
        return 2
    if a.model not in MODEL_TABLE:
        _err(f"model {a.model!r} is not in the v0.1 model table ({', '.join(MODEL_TABLE)})")
        return 2
    if not a.prices:
        _err("--live needs --prices FILE.csv (columns date,ticker,close)")
        return 2
    info = MODEL_TABLE[a.model]
    panel = load_prices_csv(a.prices)
    cutoff = a.cutoff or info.training_cutoff
    cutoff_source = "--cutoff" if a.cutoff else info.cutoff_text
    n_tickers = a.n_tickers or LIVE_DEFAULTS["n_tickers"]
    n_dates = a.n_dates or LIVE_DEFAULTS["n_dates"]
    episodes = sample_episodes(panel, n_tickers, n_dates, a.window, cutoff, a.seed)
    if len(episodes) > a.max_episodes:
        _err(
            f"planned {len(episodes)} episodes exceeds --max-episodes {a.max_episodes}; "
            "lower --n-tickers/--n-dates or raise the cap"
        )
        return 2
    plans = plan_episodes(panel, episodes, Masker.fit(panel, a.seed), a.window)  # leak check
    # Read-only for the estimate: a dry run must not create the cache directory.
    lookup = ResponseCache(a.cache_dir, create=False)
    prompts = [p for pl in plans for p in (pl.prompt_u, pl.prompt_m)]
    uncached = [p for p in prompts if lookup.get(build_request(info, p, DEFAULT_MAX_TOKENS)) is None]
    est = estimate_cost(uncached, info, DEFAULT_MAX_TOKENS)
    print(
        f"Estimated cost: {est.calls} uncached calls of {len(prompts)} planned "
        f"({len(episodes)} episodes x 2 arms), <= {est.input_tokens} input + "
        f"{est.output_tokens} output tokens at ${info.input_usd_per_mtok}/${info.output_usd_per_mtok} "
        f"per MTok = about ${est.cost_usd:.6f} (upper bound) on {info.model_id}"
    )
    if not a.yes:
        print("Dry run: no API calls made. Re-run with --yes to spend.")
        return 0
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        _err("live mode needs the ANTHROPIC_API_KEY environment variable")
        return 2
    cache = ResponseCache(a.cache_dir)
    model = ClaudeModel(info, cache, lambda: client_factory(api_key), max_calls=2 * a.max_episodes)
    result = execute(plans, model)
    summary = summarise(result.records, cutoff, a.bootstrap, 0.95, a.seed)
    config = {
        "seed": a.seed,
        "n_tickers": n_tickers,
        "n_dates": n_dates,
        "window": a.window,
        "max_episodes": a.max_episodes,
        "max_tokens": DEFAULT_MAX_TOKENS,
        "prices_file": Path(a.prices).name,
        "n_resamples": a.bootstrap,
        "level": 0.95,
    }
    rep = build_report(
        summary,
        mode="live",
        model=info.model_id,
        config=config,
        cutoff=cutoff,
        cutoff_source=cutoff_source,
        masked_prompts_checked=len(plans),
        usage=model.usage(),
    )
    paths = write_reports(rep, a.out)
    print_headline(rep, paths)
    return 0


def main(argv: list[str] | None = None, client_factory=None) -> int:
    a = build_parser().parse_args(argv)
    try:
        if a.offline:
            return _run_offline(a)
        return _run_live(a, client_factory or default_client_factory)
    except (PriceDataError, LeakError, ValueError) as e:
        _err(_redact(str(e)))
        return 2
    except Exception as e:  # noqa: BLE001 - last-resort handler; message is redacted
        _err(_redact(f"{type(e).__name__}: {e}"))
        return 1


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
