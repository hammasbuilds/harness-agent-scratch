"""Late injection: facts the model needs now, added at the END of each request.

The date, the git branch, the todo list and files that changed on disk all
change between calls. Putting them in the system prompt would change the start
of every request and throw away the provider's prefix cache (on CPU that cache
is the difference between re-reading 6,000 tokens and reading 200). So they go
last, in a message that is sent once and never stored in the transcript.
"""

from __future__ import annotations

import subprocess
from datetime import date
from pathlib import Path

from .todos import TodoList


def git_summary(workspace: Path) -> str | None:
    def git(*args: str) -> str | None:
        try:
            out = subprocess.run(["git", *args], cwd=workspace, capture_output=True, text=True,
                                 timeout=5, stdin=subprocess.DEVNULL)
        except (OSError, subprocess.TimeoutExpired):
            return None
        return out.stdout.strip() if out.returncode == 0 else None

    # symbolic-ref works in a repo with no commits yet; rev-parse covers a detached HEAD.
    branch = git("symbolic-ref", "--short", "HEAD")
    if branch is None:
        detached = git("rev-parse", "--short", "HEAD")
        if detached is None:
            return None  # not a git repository
        branch = f"(detached at {detached})"
    head = git("log", "-1", "--format=%h %s")
    return f"branch {branch}" + (f", last commit {head}" if head else ", no commits yet")


def stale_files(seen: dict[Path, float]) -> list[Path]:
    """Files whose modification time moved since the agent last read or wrote them."""
    stale = []
    for path, mtime in seen.items():
        try:
            if path.stat().st_mtime != mtime:
                stale.append(path)
        except FileNotFoundError:
            stale.append(path)
    return stale


def build_reminder(*, workspace: Path, todos: TodoList, seen: dict[Path, float],
                   today: date, git: str | None) -> str:
    lines = ["<system-reminder>",
             "Context from the harness, not a message from the user.",
             f"Today: {today.isoformat()}"]
    if git:
        lines.append(f"Git: {git}")
    if todos.items:
        lines += ["Todo list:", todos.render()]
    stale = stale_files(seen)
    if stale:
        lines.append("These files changed on disk since you last read them; read them again before editing:")
        for p in stale:
            try:
                shown = p.relative_to(workspace)
            except ValueError:
                shown = p
            lines.append(f"- {shown.as_posix()}")
    lines.append("</system-reminder>")
    return "\n".join(lines)
