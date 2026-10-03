"""The README's scripted Input / Output samples still come out as quoted."""

import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "examples"))

import scenarios  # noqa: E402


@pytest.mark.parametrize(
    "name, marks",
    [
        (
            "refusal",
            [
                "Allow? run: rm notes.txt [y/N] n",
                "the user declined",
                "notes.txt still on disk: True",
            ],
        ),
        ("compaction", ["[compacted the transcript:", "handoff note now in the system prompt"]),
        ("squeeze", ["[cut this turn's tool outputs to fit:"]),
        (
            "subagent",
            [
                "[subagent]   error: write_file is not available to subagents",
                "[subagent]   error: subagents may only run read-only commands",
                "[subagent finished]",
                "README.md unchanged: True",
            ],
        ),
    ],
)
def test_each_scenario_shows_its_behaviour(name, marks, capsys, monkeypatch):
    monkeypatch.setenv("NO_COLOR", "1")
    assert scenarios.main([name]) == 0
    out = capsys.readouterr().out
    for mark in marks:
        assert mark in out, mark


def test_the_subagent_is_refused_both_writes(capsys, monkeypatch, tmp_path):
    monkeypatch.setenv("NO_COLOR", "1")
    scenarios.subagent(tmp_path)
    # The subagent's own tool results are not printed; its transcript is not kept.
    # What shows the refusals is that no approval was asked and the file survived.
    out = capsys.readouterr().out
    assert (
        "Allow?" not in out and (tmp_path / "README.md").read_text() == "# svc\nA small service.\n"
    )


def test_unknown_scenarios_are_refused(capsys):
    assert scenarios.main(["nope"]) == 2


def test_cli_help_and_errors_as_quoted(tmp_path):
    def harness(*args):
        return subprocess.run(
            [sys.executable, "-m", "harness", *args],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=False,
            timeout=120,
        )

    help_ = harness("--help")
    assert help_.returncode == 0 and "--max-steps N" in help_.stdout
    missing = harness("-w", str(tmp_path / "nope"), "-p", "fix it")
    assert missing.returncode == 2 and "does not exist" in missing.stderr
    empty = harness("-w", str(tmp_path), "-p", "")
    assert empty.returncode == 2 and "-p needs a task" in empty.stderr
