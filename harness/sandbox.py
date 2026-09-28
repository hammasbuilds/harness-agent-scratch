"""Confine bash commands to the workspace, where the OS allows it.

Permissions ask "may this command run?". The sandbox answers a different
question: "whatever runs, what can it touch?". A command the user approved
still cannot write outside the workspace or reach the network.

- Linux:   bubblewrap (`bwrap`): the whole filesystem read-only, the workspace
           writable, private /tmp and /run, no network, own PID namespace.
- macOS:   `sandbox-exec` with a Seatbelt profile: writes allowed only in the
           workspace and the system temp folders, no network.
- Windows: no equivalent ships with the OS, so commands run unconfined. Run
           the harness inside WSL (where bwrap works) or a container if that
           matters.
"""

from __future__ import annotations

import ntpath
import os
import platform
import shutil
from collections.abc import Callable
from pathlib import Path

SEATBELT_PROFILE = """(version 1)
(allow default)
(deny network*)
(deny file-write*)
(allow file-write*
  (subpath (param "WORKSPACE"))
  (subpath "/private/tmp")
  (subpath "/private/var/folders")
  (literal "/dev/null")
  (literal "/dev/tty")
  (regex #"^/dev/fd/"))
"""


class SandboxUnavailable(RuntimeError):
    pass


class Sandbox:
    def __init__(
        self,
        workspace: Path,
        mode: str = "auto",
        system: str | None = None,
        which: Callable[[str], str | None] = shutil.which,
    ):
        if mode not in ("auto", "none", "required"):
            raise ValueError(f"sandbox mode must be auto, none or required, got {mode!r}")
        self.workspace = str(Path(workspace).resolve())
        system = system or platform.system()
        self.kind = "none"
        self.program = None
        if mode != "none":
            if system == "Linux" and (found := trusted_which("bwrap", self.workspace, which)):
                self.kind, self.program = "bwrap", found
            elif system == "Darwin" and (
                found := trusted_which("sandbox-exec", self.workspace, which)
            ):
                self.kind, self.program = "seatbelt", found
        if mode == "required" and self.kind == "none":
            raise SandboxUnavailable(
                f"no sandbox available on {system}; use HARNESS_SANDBOX=none to run unconfined"
            )

    def wrap(self, argv: list[str]) -> list[str]:
        if self.kind == "bwrap":
            ws = self.workspace
            # A read-only / still lets a process connect() to Unix sockets: the
            # D-Bus session bus (systemd-run --user) and docker.sock are both ways
            # out, so /run is replaced by an empty tmpfs (/var/run links to it).
            # --unshare-pid keeps commands from signalling host processes, such as
            # a training job; --new-session blocks TIOCSTI keystroke injection.
            # Order matters: /tmp and /run are replaced first, so a workspace under
            # either is bound back in afterwards and stays writable.
            return [
                self.program,
                "--ro-bind",
                "/",
                "/",
                "--dev",
                "/dev",
                "--proc",
                "/proc",
                "--tmpfs",
                "/tmp",
                "--tmpfs",
                "/run",
                "--bind",
                ws,
                ws,
                "--unshare-net",
                "--unshare-pid",
                "--unshare-ipc",
                "--new-session",
                "--die-with-parent",
                "--chdir",
                ws,
                *argv,
            ]
        if self.kind == "seatbelt":
            return [
                self.program,
                "-p",
                SEATBELT_PROFILE,
                "-D",
                f"WORKSPACE={self.workspace}",
                *argv,
            ]
        return list(argv)

    def describe(self) -> str:
        return {
            "bwrap": "bubblewrap: writes confined to the workspace, no network",
            "seatbelt": "seatbelt: writes confined to the workspace, no network",
            "none": "none: bash commands run with your full user rights",
        }[self.kind]


# Windows looks in the current folder for a bare program name before PATH, so
# a cloned repository shipping its own git.exe or bash.exe would run in place
# of the real one. This turns that off for shutil.which and for cmd.exe.
if os.name == "nt":
    os.environ.setdefault("NoDefaultCurrentDirectoryInExePath", "1")

# The command itself travels in this environment variable, never on the command
# line. Git Bash's runtime re-parses its own Windows command line: an argument
# with no space in it arrives unquoted, and the runtime strips quotes and
# expands braces and globs before bash sees anything. `cat<'a.txt\n>x'` was
# classified as reading a.txt and then emptied x. Environment values are not
# re-parsed, so bash runs exactly the string the permission check read.
COMMAND_VAR = "HARNESS_CMD"
POSIX_RUNNER = f'eval "${COMMAND_VAR}"'
CMD_RUNNER = f"%{COMMAND_VAR}%"


def trusted_which(
    name: str,
    workspace: Path | None = None,
    which: Callable[[str], str | None] = shutil.which,
    isabs: Callable[[str], bool] = os.path.isabs,
) -> str | None:
    """An absolute path to `name` from PATH, never relative (a `.` in PATH) and
    never inside the workspace, where a repository could have put its own."""
    found = which(name)
    if not found or not isabs(found):
        return None
    if workspace is not None:
        try:
            resolved = Path(found).resolve()
        except OSError:
            return None
        ws = Path(workspace).resolve()
        if resolved == ws or resolved.is_relative_to(ws):
            return None
    return found


def system_cmd() -> str:
    return ntpath.join(os.environ.get("SYSTEMROOT", r"C:\Windows"), "System32", "cmd.exe")


def find_shell(
    configured: str | None = None,
    system: str | None = None,
    which: Callable[[str], str | None] = shutil.which,
    exists: Callable[[str], bool] = os.path.exists,
    workspace: Path | None = None,
) -> tuple[list[str], str]:
    """Return (argv, description): the argv runs whatever is in COMMAND_VAR."""
    system = system or platform.system()
    # Judge absoluteness by the target system's rules, so the Windows branch
    # behaves the same when it is exercised on a Linux CI runner.
    isabs = ntpath.isabs if system == "Windows" else os.path.isabs
    find = lambda name: trusted_which(name, workspace, which, isabs)  # noqa: E731
    if configured:
        if configured.lower().removesuffix(".exe").endswith("cmd"):
            return [configured, "/d", "/c", CMD_RUNNER], "cmd.exe (Windows command syntax)"
        return [configured, "-c", POSIX_RUNNER], configured
    if system == "Windows":
        git_bash = "Git Bash (POSIX shell syntax; use forward slashes)"
        roots = [
            ntpath.join(os.environ.get(v, d), "Git")
            for v, d in (
                ("ProgramFiles", r"C:\Program Files"),
                ("ProgramFiles(x86)", r"C:\Program Files (x86)"),
            )
        ]
        # Portable or per-user Git: find git.exe and look for bin\bash.exe beside it.
        git = find("git")
        if git:
            parent = ntpath.dirname(ntpath.dirname(git))  # ...\cmd\git.exe or ...\bin\git.exe
            roots += [parent, ntpath.dirname(parent)]  # ...\mingw64\bin\git.exe
        for root in roots:
            candidate = ntpath.join(root, "bin", "bash.exe")  # Windows separators on any host
            if exists(candidate):
                return [candidate, "-c", POSIX_RUNNER], git_bash
        found = find("bash")
        # System32\bash.exe and the WindowsApps alias are WSL: a different
        # filesystem entirely, where the workspace paths do not exist.
        if found and not any(s in found.lower() for s in ("system32", "windowsapps")):
            return [found, "-c", POSIX_RUNNER], "bash"
        return [system_cmd(), "/d", "/c", CMD_RUNNER], "cmd.exe (Windows command syntax)"
    found = find("bash")
    return (
        ([found, "-c", POSIX_RUNNER], "bash") if found else (["/bin/sh", "-c", POSIX_RUNNER], "sh")
    )
