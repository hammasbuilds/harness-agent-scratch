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
from pathlib import Path
from typing import Callable

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
    def __init__(self, workspace: Path, mode: str = "auto", system: str | None = None,
                 which: Callable[[str], str | None] = shutil.which):
        if mode not in ("auto", "none", "required"):
            raise ValueError(f"sandbox mode must be auto, none or required, got {mode!r}")
        self.workspace = str(Path(workspace).resolve())
        system = system or platform.system()
        self.kind = "none"
        if mode != "none":
            if system == "Linux" and which("bwrap"):
                self.kind = "bwrap"
            elif system == "Darwin" and which("sandbox-exec"):
                self.kind = "seatbelt"
        if mode == "required" and self.kind == "none":
            raise SandboxUnavailable(f"no sandbox available on {system}; use HARNESS_SANDBOX=none to run unconfined")

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
            return ["bwrap", "--ro-bind", "/", "/", "--dev", "/dev", "--proc", "/proc",
                    "--tmpfs", "/tmp", "--tmpfs", "/run", "--bind", ws, ws,
                    "--unshare-net", "--unshare-pid", "--unshare-ipc", "--new-session", "--die-with-parent",
                    "--chdir", ws, *argv]
        if self.kind == "seatbelt":
            return ["sandbox-exec", "-p", SEATBELT_PROFILE, "-D", f"WORKSPACE={self.workspace}", *argv]
        return list(argv)

    def describe(self) -> str:
        return {
            "bwrap": "bubblewrap: writes confined to the workspace, no network",
            "seatbelt": "seatbelt: writes confined to the workspace, no network",
            "none": "none: bash commands run with your full user rights",
        }[self.kind]


def find_shell(configured: str | None = None, system: str | None = None,
               which: Callable[[str], str | None] = shutil.which,
               exists: Callable[[str], bool] = os.path.exists) -> tuple[list[str], str]:
    """Return (argv prefix, description) for running one command string."""
    system = system or platform.system()
    if configured:
        if configured.lower().removesuffix(".exe").endswith("cmd"):
            return [configured, "/d", "/c"], "cmd.exe (Windows command syntax)"
        return [configured, "-c"], configured
    if system == "Windows":
        git_bash = "Git Bash (POSIX shell syntax; use forward slashes)"
        roots = [ntpath.join(os.environ.get(v, d), "Git") for v, d in
                 (("ProgramFiles", r"C:\Program Files"), ("ProgramFiles(x86)", r"C:\Program Files (x86)"))]
        # Portable or per-user Git: find git.exe and look for bin\bash.exe beside it.
        git = which("git")
        if git:
            parent = ntpath.dirname(ntpath.dirname(git))  # ...\cmd\git.exe or ...\bin\git.exe
            roots += [parent, ntpath.dirname(parent)]     # ...\mingw64\bin\git.exe
        for root in roots:
            candidate = ntpath.join(root, "bin", "bash.exe")  # Windows separators on any host
            if exists(candidate):
                return [candidate, "-c"], git_bash
        found = which("bash")
        # System32\bash.exe and the WindowsApps alias are WSL: a different
        # filesystem entirely, where the workspace paths do not exist.
        if found and not any(s in found.lower() for s in ("system32", "windowsapps")):
            return [found, "-c"], "bash"
        return ["cmd.exe", "/d", "/c"], "cmd.exe (Windows command syntax)"
    found = which("bash")
    return ([found, "-c"], "bash") if found else (["/bin/sh", "-c"], "sh")
