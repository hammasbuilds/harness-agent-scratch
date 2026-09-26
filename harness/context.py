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

from .permissions import repo_config_is_plain
from .sandbox import trusted_which
from .todos import TodoList

# The reminder rides in every request and cannot be trimmed, so every part of
# it is bounded.
GIT_SUMMARY_CHARS = 200
MAX_STALE_SHOWN = 20


def _git_dir(workspace: Path) -> Path | None:
    """The repository the workspace belongs to, found without git's own search.

    git's search would also accept the workspace root itself as a repository if
    it holds HEAD, objects/ and refs/, which the model can write; its config
    could then run a program (gpg.program with log.showSignature) every time the
    harness asks for the branch. So only a real `.git` folder counts, and a
    workspace `.git` only with a plain config.

    A parent folder's `.git` is used as it is: the model cannot write outside
    the workspace, and only symbolic-ref, rev-parse and `log --format` run, none
    of which starts a program from config. A `.git` *file* (a worktree or a
    submodule checkout) is not followed, and the reminder then shows no branch.
    """
    own = workspace / ".git"
    if own.exists():
        return own if repo_config_is_plain(workspace) else None
    return next((p / ".git" for p in workspace.parents if (p / ".git").is_dir()), None)


def git_summary(workspace: Path) -> str | None:
    git_dir = _git_dir(workspace)
    # An absolute git from PATH: never one the repository itself ships.
    git_exe = trusted_which("git", workspace)
    if git_dir is None or git_exe is None:
        return None

    def git(*args: str) -> str | None:
        try:
            # git prints UTF-8; the locale codec (cp1252 here) failed on an emoji
            # in a commit subject and took the whole session down with it.
            out = subprocess.run([git_exe, f"--git-dir={git_dir}", "-c", "core.fsmonitor=false",
                                  "-c", "log.showSignature=false", *args],
                                 cwd=workspace, capture_output=True, encoding="utf-8",
                                 errors="replace", timeout=5, stdin=subprocess.DEVNULL)
        except (OSError, subprocess.TimeoutExpired):
            return None
        return (out.stdout or "").strip() if out.returncode == 0 else None

    # symbolic-ref works in a repo with no commits yet; rev-parse covers a detached HEAD.
    branch = git("symbolic-ref", "--short", "HEAD")
    if branch is None:
        detached = git("rev-parse", "--short", "HEAD")
        if detached is None:
            return None  # not a git repository
        branch = f"(detached at {detached})"
    head = git("log", "-1", "--format=%h %s")
    summary = f"branch {branch}" + (f", last commit {head}" if head else ", no commits yet")
    return summary[:GIT_SUMMARY_CHARS]  # it rides in every request; a huge commit subject must not


def fingerprint(path: Path) -> tuple[int, int]:
    """(mtime in nanoseconds, size). The size catches a rewrite that lands in the
    same mtime tick, which on coarse filesystems a float mtime alone misses."""
    st = path.stat()
    return st.st_mtime_ns, st.st_size


def stale_files(seen: dict[Path, tuple[int, int]]) -> list[Path]:
    """Files that changed or vanished since the agent last read or wrote them."""
    stale = []
    for path, known in seen.items():
        try:
            if fingerprint(path) != known:
                stale.append(path)
        except OSError:
            stale.append(path)
    return stale


def build_reminder(*, workspace: Path, todos: TodoList, seen: dict[Path, tuple[int, int]],
                   today: date, git: str | None) -> str:
    lines = ["<system-reminder>",
             "Context from the harness, not a message from the user.",
             f"Today: {today.isoformat()}"]
    if git:
        lines.append(f"Git: {git}")
    if todos.items:
        lines += ["Todo list:", todos.render_open()]
    stale = stale_files(seen)
    if stale:
        lines.append("These files changed on disk since you last read them; read them again before editing:")
        for p in stale[:MAX_STALE_SHOWN]:
            try:
                shown = p.relative_to(workspace)
            except ValueError:
                shown = p
            lines.append(f"- {shown.as_posix()}")
        if len(stale) > MAX_STALE_SHOWN:
            lines.append(f"- ... and {len(stale) - MAX_STALE_SHOWN} more")
    lines.append("</system-reminder>")
    return "\n".join(lines)
