"""Episodes, the two-arm run, and the offline entry point.

Flow: sample episodes -> plan (build both observations, render both prompts, leak-check every
masked prompt; nothing has been sent to a model yet) -> execute (ask the model) -> summarise.
Planning before executing means a leak or an over-cap run fails before any money is spent.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import date

import numpy as np

from .agent import DailyLongShortAgent, Model
from .data import PricePanel, synthetic_panel
from .masking import LeakChecker, Masker
from .mocks import HonestMock, MemorisingMock
from .pit import Observation, build_observation
from .prompt import render_prompt
from .report import build_report
from .stats import EpisodeResult, summarise

# Synthetic panel: 520 business days from 2024-01-02 (to 2025-12-30); the offline "training
# cutoff" sits near the middle so both parts have plenty of dates.
OFFLINE_CUTOFF = date(2025, 1, 2)


@dataclass(frozen=True)
class OfflineConfig:
    seed: int = 0
    n_tickers: int = 20
    n_dates: int = 120
    window: int = 20
    n_days: int = 520
    cutoff: date = OFFLINE_CUTOFF  # where the analysis splits pre/post
    mock_cutoff: date = OFFLINE_CUTOFF  # what the memorising mock actually remembers up to
    n_resamples: int = 2000
    level: float = 0.95
    recall_p: float = 0.75


@dataclass(frozen=True)
class Episode:
    ticker: str
    date: date
    r: float


@dataclass(frozen=True)
class PlannedEpisode:
    episode: Episode
    obs_u: Observation
    obs_m: Observation
    prompt_u: str
    prompt_m: str


@dataclass
class RunResult:
    records: list[EpisodeResult]
    raw_answers: list[tuple[str, str]] = field(default_factory=list)


def sample_episodes(
    panel: PricePanel, n_tickers: int, n_dates: int, window: int, cutoff: date | None, seed: int
) -> list[Episode]:
    """Seeded sample of decision dates (half before, half on/after the cutoff when both sides
    have enough eligible dates; the shorter side is taken whole otherwise) crossed with a seeded
    sample of tickers. Eligible dates have a full window behind them and a next day after them."""
    rng = np.random.default_rng(np.random.SeedSequence([seed, 0xE915]))
    eligible = list(range(window - 1, len(panel.dates) - 1))
    if n_dates < 1 or not eligible:
        raise ValueError("no eligible decision dates (panel too short for the window?)")
    if cutoff is None:
        chosen = rng.choice(eligible, size=min(n_dates, len(eligible)), replace=False).tolist()
    else:
        pre = [i for i in eligible if panel.dates[i] < cutoff]
        post = [i for i in eligible if panel.dates[i] >= cutoff]
        want_pre = min(len(pre), max(n_dates // 2, n_dates - len(post)))
        want_post = min(len(post), n_dates - want_pre)
        chosen = (
            rng.choice(pre, size=want_pre, replace=False).tolist()
            + rng.choice(post, size=want_post, replace=False).tolist()
        )
    if n_tickers >= len(panel.tickers):
        tickers = list(panel.tickers)
    else:
        cols = sorted(rng.choice(len(panel.tickers), size=n_tickers, replace=False).tolist())
        tickers = [panel.tickers[j] for j in cols]
    return [
        Episode(t, panel.dates[i], panel.next_day_return(t, panel.dates[i]))
        for i in sorted(chosen)
        for t in tickers
    ]


def plan_episodes(
    panel: PricePanel, episodes: list[Episode], masker: Masker, window: int
) -> list[PlannedEpisode]:
    """Render both prompts for every episode and leak-check every masked prompt (raises LeakError)."""
    checker = LeakChecker(panel)
    plans = []
    for e in episodes:
        obs_u = build_observation(panel, e.ticker, e.date, window)
        obs_m = masker.mask(obs_u)
        prompt_m = render_prompt(obs_m)
        i = panel.date_index(e.date)
        next_close = float(panel.closes[i + 1, panel.ticker_index(e.ticker)])
        checker.check(prompt_m, real_prices=[b.close for b in obs_u.bars] + [next_close])
        plans.append(PlannedEpisode(e, obs_u, obs_m, render_prompt(obs_u), prompt_m))
    return plans


def execute(plans: list[PlannedEpisode], model: Model) -> RunResult:
    agent = DailyLongShortAgent(model)
    out = RunResult(records=[])
    for p in plans:
        du = agent.decide_detailed(p.obs_u, p.prompt_u)
        dm = agent.decide_detailed(p.obs_m, p.prompt_m)
        e = p.episode
        out.records.append(EpisodeResult(e.ticker, e.date, e.r, du.action, du.parsed, dm.action, dm.parsed))
        out.raw_answers.append((du.raw, dm.raw))
    return out


def run_offline(kind: str, cfg: OfflineConfig) -> dict:
    """Full offline probe with a deterministic mock ('memorising' or 'honest'). Returns the report."""
    panel = synthetic_panel(cfg.seed, n_tickers=cfg.n_tickers, n_days=cfg.n_days)
    if kind == "memorising":
        model: Model = MemorisingMock(panel, cfg.mock_cutoff, seed=cfg.seed, p=cfg.recall_p)
    elif kind == "honest":
        model = HonestMock(seed=cfg.seed)
    else:
        raise ValueError(f"unknown mock {kind!r}; use 'memorising' or 'honest'")
    episodes = sample_episodes(panel, cfg.n_tickers, cfg.n_dates, cfg.window, cfg.cutoff, cfg.seed)
    plans = plan_episodes(panel, episodes, Masker.fit(panel, cfg.seed), cfg.window)
    result = execute(plans, model)
    summary = summarise(result.records, cfg.cutoff, cfg.n_resamples, cfg.level, cfg.seed)
    config = {k: (v.isoformat() if isinstance(v, date) else v) for k, v in asdict(cfg).items()}
    return build_report(
        summary,
        mode="offline",
        model=f"{kind}-mock",
        config=config,
        cutoff=cfg.cutoff,
        cutoff_source="offline config (synthetic panel)",
        masked_prompts_checked=len(plans),
    )
