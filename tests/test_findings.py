"""The committed results still describe the committed code.

Both studies write JSON under results/. These tests recompute what they can
without the tokenizer (the estimate side, and every permission decision) and
fail when the code has moved away from the file, or when the README quotes a
number the file does not hold.
"""

import json
import re
import subprocess
from pathlib import Path

import pytest

from harness.compaction import text_tokens
from harness.permissions import ALLOW, classify
from scripts import permission_study, token_corpus

ROOT = Path(__file__).resolve().parent.parent
CALIBRATION = json.loads((ROOT / "results" / "token_estimate_calibration.json").read_text("utf-8"))
PERMISSIONS = json.loads((ROOT / "results" / "permission_classifier.json").read_text("utf-8"))
README = (ROOT / "README.md").read_text("utf-8")


def _history_available() -> bool:
    try:
        subprocess.run(
            ["git", "cat-file", "-e", token_corpus.PINNED],
            cwd=ROOT,
            check=True,
            capture_output=True,
        )
        return True
    except (OSError, subprocess.CalledProcessError):
        return False


needs_history = pytest.mark.skipif(
    not _history_available(), reason="the corpus reads pinned commits (shallow clone)"
)


@needs_history
def test_the_corpus_is_deterministic_and_matches_the_results_file():
    first, second = token_corpus.build(), token_corpus.build()
    assert first == second
    assert len(first) == CALIBRATION["corpus"]["kinds"]
    for name, samples in first.items():
        assert len(samples) == token_corpus.SAMPLES
        assert all(len(s) == token_corpus.SIZE for s in samples), name


@needs_history
def test_the_estimate_is_still_never_below_the_measured_real_count():
    # Changing text_tokens changes these estimates; each must stay at or above
    # the real Qwen2.5 count recorded for the same text.
    corpus = token_corpus.build()
    for name, samples in corpus.items():
        recorded = CALIBRATION["kinds"][name]["per_sample"]
        for text, row in zip(samples, recorded, strict=True):
            assert text_tokens(text) >= row["real"], name
            assert text_tokens(text) == row["estimate"], f"{name}: rerun calibrate_tokens.py"


def test_the_calibration_summary_is_consistent():
    h = CALIBRATION["harness_estimate"]
    rows = [r for k in CALIBRATION["kinds"].values() for r in k["per_sample"]]
    assert len(rows) == CALIBRATION["corpus"]["samples"]
    assert h["max_ratio"] == max(r["ratio"] for r in rows) <= 1
    assert h["samples_underestimated"] == 0
    lo, hi = h["mean_ratio_ci95"]
    assert lo <= h["mean_ratio_over_kinds"] <= hi


def test_no_labelled_change_command_from_the_agent_corpus_runs_unasked(tmp_path):
    ws = tmp_path.resolve()
    permission_study.make_repo(ws)
    rows = permission_study.load("swesmith_commands.jsonl")
    allowed_changes = [
        r["command"]
        for r in rows
        if r["label"] == "change"
        and classify(r["command"].replace("/testbed", ws.as_posix()), ws) == ALLOW
    ]
    assert allowed_changes == []


def test_no_dangerous_command_from_the_independent_corpus_runs_unasked(tmp_path):
    rows = permission_study.load("terminal_agent_risky.jsonl")
    allowed = [
        r["command"]
        for r in rows
        if r["label"] == "dangerous" and classify(r["command"], tmp_path) == ALLOW
    ]
    assert allowed == []


def test_the_permission_results_match_the_code(tmp_path):
    ws = tmp_path.resolve()
    permission_study.make_repo(ws)
    fresh = permission_study.swesmith(ws)
    assert fresh["decisions"] == PERMISSIONS["swesmith_agent_commands"]["decisions"]
    assert fresh["vs_labels"] == PERMISSIONS["swesmith_agent_commands"]["vs_labels"]


def test_wilson_interval():
    assert permission_study.wilson(0, 0) == [0.0, 0.0]
    lo, hi = permission_study.wilson(176, 176)
    assert hi == 1.0 and 0.97 < lo < 0.98
    lo, hi = permission_study.wilson(5, 10)
    assert lo < 0.5 < hi


def _pct(x: float) -> str:
    return f"{x * 100:.1f}%"


def test_the_readme_quotes_the_results_files():
    h = CALIBRATION["harness_estimate"]
    b = CALIBRATION["baseline_chars_div_3"]
    s = PERMISSIONS["swesmith_agent_commands"]
    t = PERMISSIONS["terminal_agent_labelled_commands"]["by_label"]
    expected = [
        f"{h['max_ratio']:.3f}",
        f"{h['mean_ratio_over_kinds']:.3f}",
        f"{h['mean_ratio_ci95'][0]:.3f}-{h['mean_ratio_ci95'][1]:.3f}",
        f"{b['kinds_underestimated']} of {CALIBRATION['corpus']['kinds']}",
        f"{b['max_ratio']:.2f}x",
        f"{s['vs_labels']['allow_recall']['k']}/{s['vs_labels']['allow_recall']['n']}",
        _pct(s["vs_labels"]["allow_recall"]["rate"]),
        f"{t['dangerous']['allow']['k']}/{t['dangerous']['allow']['n']}",
        f"{t['safe']['allow']['k']}/{t['safe']['allow']['n']}",
    ]
    missing = [e for e in expected if e not in README]
    assert missing == []
    # A figure that once appeared three ways (23, 27 and 34 kinds) now appears once.
    kinds = {int(n) for n in re.findall(r"(\d+) kinds of text", README)}
    assert kinds <= {CALIBRATION["corpus"]["kinds"]}
