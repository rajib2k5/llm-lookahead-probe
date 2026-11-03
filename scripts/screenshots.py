"""Regenerate the how-to-use images in docs/images/.

Regenerate from the repository root with:

    uv run --with rich --with playwright==1.62.0 python scripts/screenshots.py

Rich and Playwright are not dependencies of the package. The `--with` flags install them for this
one run. Playwright 1.62.0 matches the Chromium build 1234 in the default browser cache. Another
version needs another build. If the browser is missing, run
`uv run --with playwright==1.62.0 playwright install chromium` once.

The script runs the real console script in offline mock mode (no API key, no `--live` spending)
from the neutral directory /tmp/demo-llm-lookahead-probe, and captures its real stdout and
stderr. It never types or fakes output. Each capture becomes a terminal-style SVG (rich) and then a PNG (headless Chromium).
The subprocess gets a clean environment, so no home path, PATH, or key can show in an image.
Before it finishes, the script checks the SVG text for private strings and fails on a match.
"""

from __future__ import annotations

import io
import re
import shutil
import subprocess
import sys
from pathlib import Path

from rich.console import Console
from rich.markdown import Markdown
from rich.text import Text

ROOT = Path(__file__).resolve().parents[1]
DEMO = Path("/tmp/demo-llm-lookahead-probe")  # noqa: S108 - a fixed neutral directory is the point
# The script only ever deletes a DEMO it created itself, recognised by this marker file.
MARKER = ".created-by-llm-lookahead-probe-screenshots"
IMAGES = ROOT / "docs" / "images"
SVG_DIR = DEMO / "svg"
WIDTH = 100
MAX_PNG_BYTES = 500 * 1024
BARE = "llm-lookahead-probe"
FORBIDDEN = ["rajib", "/home/", "@", "sk-ant", "ghp_", "github_pat", ".claude"]


def find_cli() -> str:
    for cand in (ROOT / ".venv" / "bin" / BARE, Path(sys.executable).parent / BARE):
        if cand.exists():
            return str(cand)
    found = shutil.which(BARE)
    if not found:
        sys.exit(f"cannot find the {BARE} console script; run `uv sync` first")
    return found


def run_cli(cli: str, args: list[str]) -> str:
    """Run the CLI in DEMO with a clean environment and return stdout and stderr together."""
    env = {"HOME": str(DEMO), "PATH": "/usr/bin:/bin", "NO_COLOR": "1", "LANG": "C.UTF-8"}
    proc = subprocess.run(  # noqa: S603
        [cli, *args], cwd=DEMO, env=env, capture_output=True, text=True, check=False, timeout=300
    )
    return proc.stdout + proc.stderr


def new_console() -> Console:
    return Console(
        record=True, width=WIDTH, file=io.StringIO(), force_terminal=True, color_system="truecolor"
    )


def prompt_line(console: Console, args: list[str]) -> None:
    console.print(Text("$ ", style="bold green") + Text(" ".join([BARE, *args]), style="bold"))


def terminal_capture(args: list[str], output: str) -> Console:
    console = new_console()
    prompt_line(console, args)
    console.print(Text(output.rstrip("\n")))
    return console


def sections(md: str) -> dict[str, str]:
    """Split a report into {heading: body}. The text before the first '## ' is under 'head'."""
    out: dict[str, str] = {}
    name, buf = "head", []
    for line in md.splitlines():
        if line.startswith("## "):
            out[name] = "\n".join(buf).strip("\n")
            name, buf = line[3:].strip(), []
        buf.append(line)
    out[name] = "\n".join(buf).strip("\n")
    return out


def pick(secs: dict[str, str], prefix: str) -> str:
    for k, v in secs.items():
        if k.startswith(prefix):
            return v
    raise KeyError(prefix)


def drop_lines(text: str, starts: tuple[str, ...]) -> str:
    return "\n".join(ln for ln in text.splitlines() if not ln.startswith(starts))


def markdown_capture(title_cmd: str, parts: list[str], width: int = WIDTH) -> Console:
    console = Console(
        record=True, width=width, file=io.StringIO(), force_terminal=True, color_system="truecolor"
    )
    # Not a shell command: the image is a rendered excerpt, so label it as one.
    console.print(Text(title_cmd, style="dim italic"))
    for part in parts:
        console.print()
        console.print(Markdown(part))
    return console


def save_svg(console: Console, name: str, title: str) -> Path:
    SVG_DIR.mkdir(parents=True, exist_ok=True)
    path = SVG_DIR / f"{name}.svg"
    console.save_svg(str(path), title=title)
    return path


def svg_to_png(svg_paths: dict[str, Path]) -> None:
    from playwright.sync_api import sync_playwright

    IMAGES.mkdir(parents=True, exist_ok=True)
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(device_scale_factor=2)
        for name, svg in svg_paths.items():
            page.set_content(f'<body style="margin:0">{svg.read_text()}</body>')
            page.locator("svg").first.screenshot(path=str(IMAGES / f"{name}.png"), omit_background=True)
        browser.close()


def privacy_check(svg_paths: dict[str, Path], captures: list[str]) -> None:
    """Fail if private strings appear in any SVG text node, title, or raw capture."""
    bad: list[str] = []
    for name, svg in svg_paths.items():
        raw = svg.read_text()
        # Text nodes only: the SVG <style> block holds CSS ('@font-face'), which is not content.
        texts = re.findall(r"<text[^>]*>(.*?)</text>", raw, flags=re.S)
        texts += re.findall(r"<title>(.*?)</title>", raw, flags=re.S)
        blob = "\n".join(texts)
        bad += [f"{name}: {f!r}" for f in FORBIDDEN if f in blob]
    for i, cap in enumerate(captures):
        bad += [f"capture {i}: {f!r}" for f in FORBIDDEN if f in cap]
    if bad:
        sys.exit("privacy check failed:\n  " + "\n  ".join(bad))
    print(f"privacy check passed: no {FORBIDDEN} in {len(svg_paths)} SVGs and {len(captures)} captures")


def write_prices_csv(path: Path) -> None:
    from llm_lookahead_probe.data import synthetic_panel

    panel = synthetic_panel(0)
    lines = ["date,ticker,close"]
    for i, d in enumerate(panel.dates):
        for j, t in enumerate(panel.tickers):
            lines.append(f"{d.isoformat()},{t},{panel.closes[i, j]:.2f}")
    # The loader wants rows sorted by (date, ticker).
    head, rows = lines[0], sorted(lines[1:], key=lambda s: tuple(s.split(",")[:2]))
    path.write_text("\n".join([head, *rows]) + "\n")


def main() -> int:
    cli = find_cli()
    if DEMO.exists():
        if not (DEMO / MARKER).is_file():
            sys.exit(f"{DEMO} exists but was not created by this script; move it away first")
        shutil.rmtree(DEMO)
    DEMO.mkdir(parents=True)
    (DEMO / MARKER).touch()

    captures: list[str] = []
    svgs: dict[str, Path] = {}

    def capture(args: list[str]) -> str:
        out = run_cli(cli, args)
        captures.append(out)
        return out

    # 01: help
    args = ["--help"]
    svgs["01-install"] = save_svg(
        terminal_capture(args, capture(args)), "01-install", "llm-lookahead-probe --help"
    )

    # 02: memorising run
    args = ["run", "--offline", "--mock", "memorising", "--out", "reports/memorising"]
    svgs["02-run-memorising"] = save_svg(
        terminal_capture(args, capture(args)), "02-run-memorising", "Run the memorising mock"
    )

    # 03 and 04: the real report.md written by that run
    md = (DEMO / "reports" / "memorising" / "report.md").read_text()
    captures.append(md)
    secs = sections(md)
    cmd = "reports/memorising/report.md, rendered as Markdown (excerpt)"
    verdict = drop_lines(pick(secs, "Verdict"), ("Rule:",)).replace("\n\n\n", "\n\n")
    gap = pick(secs, "Gap")
    svgs["03-report-memorising"] = save_svg(
        markdown_capture(cmd, [secs["head"], verdict, gap]),
        "03-report-memorising",
        "report.md (memorising mock)",
    )
    drift = pick(secs, "Trade direction")
    drift_table = "\n".join(ln for ln in drift.splitlines() if ln.startswith("|"))
    svgs["04-cutoff-split"] = save_svg(
        markdown_capture(cmd, [gap, "## Trade direction and drift\n\n" + drift_table], width=142),
        "04-cutoff-split",
        "Before and after the training cutoff",
    )

    # 05: honest run
    args = ["run", "--offline", "--mock", "honest", "--out", "reports/honest"]
    svgs["05-run-honest"] = save_svg(
        terminal_capture(args, capture(args)), "05-run-honest", "Run the honest mock"
    )

    # 06: live dry run. No key, no --yes: it prints the cost estimate and makes no call.
    write_prices_csv(DEMO / "prices.csv")
    args = [
        "run", "--live", "--prices", "prices.csv", "--n-tickers", "5", "--n-dates", "40",
        "--max-episodes", "200", "--cutoff", "2025-01-02",
    ]  # fmt: skip
    out = capture(args)
    if "Dry run: no API calls made" not in out:
        sys.exit("live dry run did not print the dry-run line:\n" + out)
    svgs["06-live-dry-run"] = save_svg(
        terminal_capture(args, out), "06-live-dry-run", "Live dry run (no calls, no cost)"
    )

    privacy_check(svgs, captures)
    svg_to_png(svgs)

    for name in svgs:
        size = (IMAGES / f"{name}.png").stat().st_size
        print(f"{name}.png {size} bytes")
        if size >= MAX_PNG_BYTES:
            sys.exit(f"{name}.png is {size} bytes; limit is {MAX_PNG_BYTES}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
