import json
import os

import pytest

from llm_lookahead_probe.cli import main


def test_offline_memorising_run_writes_reports(tmp_path, capsys):
    rc = main(["run", "--offline", "--mock", "memorising", "--out", str(tmp_path)])
    out = capsys.readouterr().out
    assert rc == 0
    rep = json.loads((tmp_path / "report.json").read_text())
    assert rep["verdict"] == "memorisation suspected"
    assert "Verdict: memorisation suspected" in out
    md = (tmp_path / "report.md").read_text()
    assert "memorisation suspected" in md and "Rule:" in md


def test_offline_honest_run(tmp_path, capsys):
    assert main(["run", "--offline", "--mock", "honest", "--out", str(tmp_path)]) == 0
    assert json.loads((tmp_path / "report.json").read_text())["verdict"] == "no evidence"


def test_offline_and_live_are_mutually_exclusive(tmp_path):
    with pytest.raises(SystemExit):
        main(["run", "--offline", "--live", "--mock", "honest", "--out", str(tmp_path)])


def test_offline_needs_mock(tmp_path, capsys):
    assert main(["run", "--offline", "--out", str(tmp_path)]) == 2


def test_refuses_to_write_through_symlink(tmp_path):
    target = tmp_path / "elsewhere.json"
    target.write_text("keep")
    out = tmp_path / "out"
    out.mkdir()
    os.symlink(target, out / "report.json")
    assert main(["run", "--offline", "--mock", "honest", "--n-tickers", "4", "--out", str(out)]) == 2
    assert target.read_text() == "keep"


def test_bad_cutoff_is_a_usage_error(tmp_path):
    with pytest.raises(SystemExit):
        main(
            [
                "run",
                "--offline",
                "--mock",
                "honest",
                "--cutoff",
                "June 2026",
                "--out",
                str(tmp_path),
            ]
        )


def test_offline_insufficient_data_exits_4_and_still_writes_reports(tmp_path, capsys):
    # A cutoff after the synthetic panel leaves no post-cutoff dates.
    rc = main(["run", "--offline", "--mock", "honest", "--cutoff", "2030-01-01", "--out", str(tmp_path)])
    assert rc == 4
    assert json.loads((tmp_path / "report.json").read_text())["verdict"] == "insufficient data"
    assert (tmp_path / "report.md").exists()


def test_help_documents_the_exit_codes(capsys):
    with pytest.raises(SystemExit):
        main(["run", "--help"])
    out = capsys.readouterr().out
    for code in ("0 ", "1 ", "2 ", "3 ", "4 ", "5 "):
        assert f"exit {code}" in out or f"  {code}" in out
    assert "invalid run" in out and "insufficient data" in out
