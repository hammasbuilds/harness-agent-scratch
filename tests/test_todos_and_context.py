import os
import shutil
import subprocess
import time
from datetime import date

import pytest

from harness.context import build_reminder, fingerprint, git_summary, stale_files
from harness.todos import TodoError, TodoList


def test_todo_list_renders_status_marks():
    todos = TodoList()
    out = todos.replace([{"content": "write hello.txt", "status": "completed"},
                         {"content": "star pattern", "status": "in_progress"},
                         {"content": "fibonacci", "status": "pending"}])
    assert out == "[x] write hello.txt\n[>] star pattern\n[ ] fibonacci"


@pytest.mark.parametrize("raw, message", [
    ("not a list", "must be a list"),
    (["text"], "not an object"),
    ([{"content": " ", "status": "pending"}], "no content"),
    ([{"content": "a", "status": "doing"}], "status 'doing'"),
    ([{"content": "a", "status": "in_progress"}, {"content": "b", "status": "in_progress"}], "only one"),
])
def test_todo_validation(raw, message):
    todos = TodoList()
    todos.replace([{"content": "keep me", "status": "pending"}])
    with pytest.raises(TodoError, match=message):
        todos.replace(raw)
    assert todos.render() == "[ ] keep me"  # a rejected update leaves the list alone


def test_todo_list_size_is_bounded():
    # It rides in every request's reminder, which cannot be trimmed: 80 items of
    # 400 characters once made every later request overflow.
    todos = TodoList()
    with pytest.raises(TodoError, match="at most 30"):
        todos.replace([{"content": f"step {i}", "status": "pending"} for i in range(31)])
    with pytest.raises(TodoError, match="keep each under 200"):
        todos.replace([{"content": "x" * 201, "status": "pending"}])


def test_reminder_parts_are_bounded(tmp_path):
    seen = {tmp_path / f"gone{i}.txt": (0, 0) for i in range(50)}
    text = build_reminder(workspace=tmp_path, todos=TodoList(), seen=seen, today=date(2026, 1, 1),
                          git="branch main, last commit abc " + "s" * 5000)
    assert "... and 30 more" in text and len(text) < 7000


def test_empty_todo_list():
    assert TodoList().render() == "(no todos)"


def test_stale_files_catch_edits_and_deletions(tmp_path):
    a, b, c = (tmp_path / n for n in "abc")
    for p in (a, b, c):
        p.write_text("x")
    seen = {p: fingerprint(p) for p in (a, b, c)}
    os.utime(b, (time.time() + 5, time.time() + 5))
    c.unlink()
    assert stale_files(seen) == [b, c]


def test_same_mtime_but_different_size_is_stale(tmp_path):
    f = tmp_path / "f"
    f.write_text("short")
    before = f.stat()
    seen = {f: fingerprint(f)}
    f.write_text("a longer rewrite in the same clock tick")
    os.utime(f, ns=(before.st_atime_ns, before.st_mtime_ns))  # pin mtime back
    assert stale_files(seen) == [f]


def test_reminder_contents(tmp_path):
    todos = TodoList()
    todos.replace([{"content": "step one", "status": "in_progress"}])
    f = tmp_path / "src" / "app.py"
    f.parent.mkdir()
    f.write_text("x")
    seen = {f: (0, 0)}
    text = build_reminder(workspace=tmp_path, todos=todos, seen=seen, today=date(2026, 9, 26), git="branch main")
    assert text.startswith("<system-reminder>") and text.endswith("</system-reminder>")
    assert "Today: 2026-09-26" in text
    assert "Git: branch main" in text
    assert "[>] step one" in text
    assert "- src/app.py" in text


def test_reminder_minimal(tmp_path):
    text = build_reminder(workspace=tmp_path, todos=TodoList(), seen={}, today=date(2026, 1, 1), git=None)
    assert "Git" not in text and "Todo" not in text and "changed on disk" not in text


def test_git_summary_outside_a_repo(tmp_path):
    assert git_summary(tmp_path) is None


def test_git_summary_never_uses_a_repository_planted_at_the_root(tmp_path, monkeypatch):
    # HEAD + objects/ + refs/ + config written by the model would make git treat
    # the root as a repo, and its config could run a program on `git log`.
    ran = []
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: ran.append(a) or (_ for _ in ()).throw(OSError()))
    (tmp_path / "HEAD").write_text("ref: refs/heads/main\n")
    (tmp_path / "objects").mkdir()
    (tmp_path / "refs").mkdir()
    (tmp_path / "config").write_text("[gpg]\n\tprogram = ./evil.sh\n[log]\n\tshowSignature = true\n")
    assert git_summary(tmp_path.resolve()) is None
    assert ran == []  # git was not even started


def test_git_summary_skips_a_repository_whose_config_runs_programs(tmp_path, monkeypatch):
    ran = []
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: ran.append(a))
    (tmp_path / ".git").mkdir()
    (tmp_path / ".git" / "config").write_text("[gpg]\n\tprogram = ./evil.sh\n")
    assert git_summary(tmp_path.resolve()) is None and ran == []


@pytest.mark.skipif(shutil.which("git") is None, reason="git not installed")
def test_git_summary_in_a_repo(tmp_path):
    env = {**os.environ, "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@example.com",
           "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@example.com"}
    run = lambda *a: subprocess.run(["git", *a], cwd=tmp_path, env=env, check=True, capture_output=True)
    run("init", "-q", "-b", "main")
    assert git_summary(tmp_path) == "branch main, no commits yet"
    (tmp_path / "f").write_text("x")
    run("add", "f")
    run("commit", "-q", "-m", "first commit")
    summary = git_summary(tmp_path)
    assert summary.startswith("branch main, last commit ") and summary.endswith("first commit")
    run("checkout", "-q", "--detach")
    assert git_summary(tmp_path).startswith("branch (detached at ")


@pytest.mark.skipif(shutil.which("git") is None, reason="git not installed")
def test_git_summary_survives_non_ascii_commit_subjects(tmp_path):
    env = {**os.environ, "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@example.com",
           "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@example.com"}
    run = lambda *a: subprocess.run(["git", *a], cwd=tmp_path, env=env, check=True, capture_output=True)
    run("init", "-q", "-b", "main")
    (tmp_path / "f").write_text("x")
    run("add", "f")
    run("commit", "-q", "-m", "🐛 fix crash — naïve café")  # crashed the cp1252 decoder
    assert git_summary(tmp_path).endswith("🐛 fix crash — naïve café")
