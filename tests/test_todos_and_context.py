import os
import shutil
import subprocess
import time
from datetime import date

import pytest

from harness.context import build_reminder, git_summary, stale_files
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


def test_empty_todo_list():
    assert TodoList().render() == "(no todos)"


def test_stale_files_catch_edits_and_deletions(tmp_path):
    a, b, c = (tmp_path / n for n in "abc")
    for p in (a, b, c):
        p.write_text("x")
    seen = {p: p.stat().st_mtime for p in (a, b, c)}
    os.utime(b, (time.time() + 5, time.time() + 5))
    c.unlink()
    assert stale_files(seen) == [b, c]


def test_reminder_contents(tmp_path):
    todos = TodoList()
    todos.replace([{"content": "step one", "status": "in_progress"}])
    f = tmp_path / "src" / "app.py"
    f.parent.mkdir()
    f.write_text("x")
    seen = {f: f.stat().st_mtime - 10}
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
