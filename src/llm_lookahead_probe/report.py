"""report.json / report.md construction and safe writing."""

from __future__ import annotations

import json
import os
from datetime import date
from pathlib import Path

from . import __version__


def build_report(
    summary: dict,
    *,
    mode: str,
    model: str,
    config: dict,
    cutoff: date,
    cutoff_source: str,
    masked_prompts_checked: int,
    usage: dict | None = None,
) -> dict:
    rep = {
        "tool": "llm-lookahead-probe",
        "version": __version__,
        "mode": mode,
        "model": model,
        "cutoff": cutoff.isoformat(),
        "cutoff_source": cutoff_source,
        "config": config,
        "leak_check": {"masked_prompts_checked": masked_prompts_checked, "leaks_found": 0},
    }
    rep.update(summary)
    if usage is not None:
        rep["usage"] = usage
    return rep


# Stated in every report so a reader of one file sees them. docs/DESIGN.md has the longer version.
LIMITS = (
    "pre - post removes only a masking effect that is constant in pnl. An effect constant in "
    "behaviour (e.g. abstaining more when masked) scales with each period's mean |r|; compare the "
    "'gap / mean abs r' column of the two periods before reading pre - post as memory.",
    "Masking keeps the exact return path and, under a whole-week date shift, the holiday-gap "
    "pattern. A model could recognise an episode from either, which biases toward under-detection.",
    "Masked dates fall in the real past (shifted back), not in a fictional calendar.",
    "The panel must be rectangular (every ticker on every date), i.e. a survivor-only universe.",
    "Decisions are made and scored at the observed close of the decision date.",
    "The leak check covers only the renderer's canonical forms (whole-token ticker, YYYY-MM-DD, "
    "d.dd); tests pin the prompt to that grammar.",
    "Offline mock results (e.g. power 1.0 for the memorising mock) say nothing about real models.",
)


def _f(x, pct=False) -> str:
    if x is None:
        return "n/a"
    return f"{x * 100:.3f}%" if pct else f"{x:.4f}"


def _ci(c: dict | None) -> str:
    if not c:
        return "n/a"
    return f"{_f(c['estimate'], True)} [{_f(c['low'], True)}, {_f(c['high'], True)}]"


def render_markdown(rep: dict) -> str:
    lvl = int(round(rep["interval"]["level"] * 100))
    lines = [
        "# llm-lookahead-probe report",
        "",
        f"- Mode: {rep['mode']}; model: `{rep['model']}`",
        f"- Training cutoff used for the split: {rep['cutoff']} ({rep['cutoff_source']})",
        f"- Episodes: {rep['n_episodes']} over {rep['n_dates']} decision dates",
        f"- Leak check: {rep['leak_check']['masked_prompts_checked']} masked prompts checked, "
        f"{rep['leak_check']['leaks_found']} leaks",
        "",
        f"## Verdict: **{rep['verdict']}**",
        "",
        rep["verdict_reason"],
        "",
        f"Rule: {rep['verdict_rule']}",
        "",
        f"## Gap (unmasked minus masked mean pnl per episode, {lvl}% CI)",
        "",
        "| Part | Dates | Episodes | Gap [CI] | Mean abs r | Gap / mean abs r |",
        "|---|---|---|---|---|---|",
        f"| All | {rep['n_dates']} | {rep['n_episodes']} | {_ci(rep['gap'])} "
        f"| {_f(rep.get('mean_abs_r'), True)} | {_f(rep.get('gap_per_mean_abs_r'))} |",
        f"| Before cutoff | {rep['pre']['n_dates']} | {rep['pre']['n_episodes']} "
        f"| {_ci(rep['pre']['gap'])} | {_f(rep['pre'].get('mean_abs_r'), True)} "
        f"| {_f(rep['pre'].get('gap_per_mean_abs_r'))} |",
        f"| On/after cutoff | {rep['post']['n_dates']} | {rep['post']['n_episodes']} "
        f"| {_ci(rep['post']['gap'])} | {_f(rep['post'].get('mean_abs_r'), True)} "
        f"| {_f(rep['post'].get('gap_per_mean_abs_r'))} |",
        f"| Pre minus post | | | {_ci(rep['diff_pre_minus_post'])} | | |",
        "",
        "## Arms",
        "",
        "| Arm | Mean pnl | Hit rate | Long | Short | Abstain | Unparseable |",
        "|---|---|---|---|---|---|---|",
    ]
    for name in ("unmasked", "masked"):
        a = rep["arms"][name]
        lines.append(
            f"| {name} | {_f(a['mean_pnl'], True)} | {_f(a['hit_rate'])} | {a['n_long']} | "
            f"{a['n_short']} | {a['n_abstain']} | {a['n_unparseable']} |"
        )
    lines += [
        "",
        f"Interval: {rep['interval']['method']}. Each decision date is one observation because "
        "episodes on one date share the market move.",
        f"Unparseable answers count as ABSTAIN (pnl 0); above {rep['interval']['max_unparseable_rate']:.0%} "
        "in either arm the run is invalid.",
    ]
    b = rep.get("robustness", {}).get("cluster_bootstrap")
    if b:
        lines += [
            "",
            f"## Robustness: {b['method']}",
            "",
            f"{b['n_resamples']} resamples, seed {b['seed']}. Known to under-cover at 30-60 dates.",
            "",
            "| Part | Bootstrap CI |",
            "|---|---|",
        ]
        for label, key in (
            ("All", "gap"),
            ("Before cutoff", "pre"),
            ("On/after cutoff", "post"),
            ("Pre minus post", "diff_pre_minus_post"),
        ):
            c = b[key]
            cell = "n/a" if not c else f"[{_f(c['low'], True)}, {_f(c['high'], True)}]"
            lines.append(f"| {label} | {cell} |")
    lines += ["", "## Limits of this probe", ""] + [f"- {x}" for x in LIMITS]
    if "usage" in rep:
        u = rep["usage"]
        lines += [
            "",
            "## API usage (from API responses)",
            "",
            f"- Calls made: {u['api_calls']}; served from cache: {u['cache_hits']}",
            f"- Input tokens: {u['input_tokens']}; output tokens: {u['output_tokens']}",
            f"- Cost of calls made in this run: ${u['cost_usd']:.6f}",
            f"- Stop reasons: {u['stop_reasons']}",
        ]
    return "\n".join(lines) + "\n"


def write_reports(rep: dict, out_dir: str | os.PathLike[str]) -> tuple[Path, Path]:
    """Write report.json and report.md. Refuses symlinked targets so a planted link cannot
    redirect the write to another file."""
    out = Path(out_dir)
    if out.is_symlink():
        raise ValueError(f"refusing to write into symlinked directory {out}")
    out.mkdir(parents=True, exist_ok=True)
    paths = (out / "report.json", out / "report.md")
    for p in paths:
        if p.is_symlink():
            raise ValueError(f"refusing to overwrite symlink {p}")
    paths[0].write_text(json.dumps(rep, indent=2, sort_keys=False) + "\n", encoding="utf-8")
    paths[1].write_text(render_markdown(rep), encoding="utf-8")
    return paths
