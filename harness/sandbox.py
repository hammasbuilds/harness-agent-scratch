"""Confine bash commands to the workspace, where the OS allows it.

Permissions ask "may this command run?". The sandbox answers a different
question: "whatever runs, what can it touch?". A command the user approved
still cannot write outside the workspace or reach the network.

- Linux:   bubblewrap (`bwrap`): the whole filesystem read-only, the workspace
           writable, a private /tmp, no network.
- macOS:   `sandbox-exec` with a Seatbelt profile doing the same.
- Windows: no equivalent ships with the OS, so commands run unconfined. Run
           the harness inside WSL (where bwrap works) or a container if that
           matters.
"""

from __future__ import annotations

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
            # Order matters: /tmp is replaced first, so a workspace under /tmp is
            # bound back in afterwards and stays writable.
            return ["bwrap", "--ro-bind", "/", "/", "--dev", "/dev", "--proc", "/proc",
                    "--tmpfs", "/tmp", "--bind", ws, ws, "--unshare-net", "--die-with-parent",
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
        program_files = [os.environ.get("ProgramFiles", r"C:\Program Files"),
                         os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)")]
        for base in program_files:
            candidate = os.path.join(base, "Git", "bin", "bash.exe")
            if exists(candidate):
                return [candidate, "-c"], "Git Bash (POSIX shell syntax; use forward slashes)"
        found = which("bash")
        # C:\Windows\System32\bash.exe is WSL: a different filesystem entirely.
        if found and "system32" not in found.lower():
            return [found, "-c"], "bash"
        return ["cmd.exe", "/d", "/c"], "cmd.exe (Windows command syntax)"
    found = which("bash")
    return ([found, "-c"], "bash") if found else (["/bin/sh", "-c"], "sh")
