"""The tools: everything the model can do beyond writing text.

Each tool is a Python function plus a JSON schema the model reads. The toolbox
validates arguments, runs the function, and turns every failure into text the
model can read and react to; one bad call never ends the loop.
"""

from __future__ import annotations

import codecs
import collections
import itertools
import json
import os
import re
import shutil
import signal
import subprocess
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping

from . import permissions
from .config import Config
from .context import fingerprint
from .llm import LLMError
from .sandbox import Sandbox, find_shell
from .skills import Skill
from .todos import TodoError, TodoList

Approver = Callable[[str], bool]
MAX_BASH_TIMEOUT = 600
PIPE_GRACE = 2.0  # seconds to finish reading output after the shell exits
MAX_READ_BYTES = 10_000_000
# A command that prints without end must not fill RAM on a machine that is
# training: keep this much of the start and of the end, drop the middle.
MAX_CAPTURE_CHARS = 1_000_000
BLOCKED_DIRS = (".git",)  # hooks and config there run code on the next git command
# Files that change how the harness itself behaves next time: .env feeds the
# settings, .agents/skills feeds the system prompt. Writing them needs a yes.
GUARDED_DIRS = (".agents", ".harness")

# Variables whose values are credentials. The model's commands run without them.
# Distinctive words match anywhere (PGPASSWORD); short ones only as a whole
# part of the name (GITHUB_PAT, MYSQL_PWD, but not PWD, the current folder).
_SECRET_NAME = re.compile(
    r"PASSWORD|PASSWD|SECRET|TOKEN|CREDENTIAL|API_?KEY|PRIVATE_?KEY|WEBHOOK|"
    r"(^|_)(KEY|PAT|PASS|DSN)($|_)|_PWD$|^DATABASE_URL$", re.I)
_URL_WITH_PASSWORD = re.compile(r"://[^/\s:@]*:[^/\s@]+@")  # user may be empty: redis://:pw@host
_ANSI = re.compile(r"\x1b\[[0-9;?]*[ -/]*[@-~]|\x1b\][^\x07\x1b]*(\x07|\x1b\\)")
# Encoding labels decode() returns for files with a byte-order mark.
_BOMS = [(codecs.BOM_UTF8, "utf-8-sig"), (codecs.BOM_UTF16_LE, "utf-16-le+bom"), (codecs.BOM_UTF16_BE, "utf-16-be+bom")]
# Control and bidirectional-override characters: a command containing "\r" or
# U+202E can make an approval prompt display something other than what runs.
_UNPRINTABLE = re.compile(r"[\x00-\x1f\x7f-\x9f  ‪-‮⁦-⁩]")


class ToolError(Exception):
    """Reported to the model as `error: ...`; the loop carries on."""


@dataclass
class Tool:
    name: str
    description: str
    parameters: dict
    run: Callable[..., str]

    def schema(self) -> dict:
        return {"type": "function",
                "function": {"name": self.name, "description": self.description, "parameters": self.parameters}}


def _params(required: list[str], **props: dict) -> dict:
    return {"type": "object", "properties": props, "required": required}


def _json_type_ok(value, expected: str | None) -> bool:
    """Small models send "20" for an integer; accept digit strings, reject the rest."""
    if expected == "string":
        return isinstance(value, str)
    if expected == "integer":
        return (isinstance(value, int) and not isinstance(value, bool)) or (
            isinstance(value, str) and value.strip().lstrip("-").isdigit())
    if expected == "boolean":
        return isinstance(value, bool)
    if expected == "array":
        return isinstance(value, list)
    return True


def scrubbed_env(env: Mapping[str, str]) -> dict[str, str]:
    """The environment minus credentials. `echo $GITHUB_TOKEN` would otherwise
    print a secret into a transcript that is sent to the model provider."""
    return {k: v for k, v in env.items() if not _SECRET_NAME.search(k) and not _URL_WITH_PASSWORD.search(v)}


def printable(text: str) -> str:
    """Show control characters as escapes, so a prompt shows what will run."""
    return _UNPRINTABLE.sub(lambda m: m.group().encode("unicode_escape").decode("ascii"), text)


def encode(text: str, encoding: str) -> bytes:
    """The inverse of decode(), byte-order mark included."""
    if encoding.endswith("+bom"):
        base = encoding.removesuffix("+bom")
        bom = codecs.BOM_UTF16_LE if base == "utf-16-le" else codecs.BOM_UTF16_BE
        return bom + text.encode(base)
    return text.encode(encoding)


def strip_ansi(text: str) -> str:
    """Colour codes cost tokens and confuse small models."""
    return _ANSI.sub("", text)


def decode(raw: bytes) -> tuple[str, str]:
    """Return (text, encoding) such that text.encode(encoding) gives raw back.

    UTF-8 first; anything else is read as Latin-1, which maps every byte to one
    character and so round-trips exactly. An edit to a Windows-1252 file then
    changes only the bytes it meant to, instead of turning every accented
    letter into U+FFFD.
    """
    for bom, encoding in _BOMS:
        if raw.startswith(bom):
            return raw[len(bom):].decode(encoding.removesuffix("+bom")), encoding
    try:
        return raw.decode("utf-8"), "utf-8"
    except UnicodeDecodeError:
        return raw.decode("latin-1"), "latin-1"


class _Capture:
    """Reads a pipe on a thread, keeping the first and last MAX_CAPTURE_CHARS."""

    def __init__(self, pipe, limit: int | None = None):
        self.limit = MAX_CAPTURE_CHARS if limit is None else limit
        self.head: list[str] = []
        self.head_len = 0
        self.tail: collections.deque[str] = collections.deque()
        self.tail_len = 0
        self.dropped = 0
        self.thread = threading.Thread(target=self._read, args=(pipe,), daemon=True)
        self.thread.start()

    def _read(self, pipe) -> None:
        try:
            for chunk in iter(lambda: pipe.read(65536), ""):
                if self.head_len < self.limit:
                    take = chunk[:self.limit - self.head_len]
                    self.head.append(take)
                    self.head_len += len(take)
                    chunk = chunk[len(take):]
                if chunk:
                    self.tail.append(chunk)
                    self.tail_len += len(chunk)
                    if self.tail_len > self.limit:  # keep exactly the last `limit` characters
                        joined = "".join(self.tail)
                        excess = len(joined) - self.limit
                        self.dropped += excess
                        self.tail = collections.deque([joined[excess:]])
                        self.tail_len = self.limit
        except (OSError, ValueError):
            pass  # pipe closed under us after a kill

    def text(self) -> str:
        head, tail = "".join(self.head), "".join(self.tail)
        if not self.dropped:
            return head + tail
        return f"{head}\n[... {self.dropped:,} characters of output dropped ...]\n{tail}"


class _WindowsJob:
    """A Windows Job Object that kills every process in it when closed.

    taskkill /T walks the parent-child tree, which breaks once the shell has
    exited: its orphaned background children can no longer be found. A job
    holds them regardless. (A child started in the milliseconds between
    process creation and assignment could escape; bash starts nothing that fast.)
    """

    def __init__(self, proc: subprocess.Popen):
        import ctypes
        from ctypes import wintypes

        class Basic(ctypes.Structure):
            _fields_ = [("PerProcessUserTimeLimit", ctypes.c_int64), ("PerJobUserTimeLimit", ctypes.c_int64),
                        ("LimitFlags", wintypes.DWORD), ("MinimumWorkingSetSize", ctypes.c_size_t),
                        ("MaximumWorkingSetSize", ctypes.c_size_t), ("ActiveProcessLimit", wintypes.DWORD),
                        ("Affinity", ctypes.c_size_t), ("PriorityClass", wintypes.DWORD),
                        ("SchedulingClass", wintypes.DWORD)]

        class Extended(ctypes.Structure):
            _fields_ = [("Basic", Basic), ("IoInfo", ctypes.c_ulonglong * 6),
                        ("ProcessMemoryLimit", ctypes.c_size_t), ("JobMemoryLimit", ctypes.c_size_t),
                        ("PeakProcessMemoryUsed", ctypes.c_size_t), ("PeakJobMemoryUsed", ctypes.c_size_t)]

        k32 = ctypes.WinDLL("kernel32", use_last_error=True)
        k32.CreateJobObjectW.restype = wintypes.HANDLE
        k32.CreateJobObjectW.argtypes = [ctypes.c_void_p, wintypes.LPCWSTR]
        k32.SetInformationJobObject.argtypes = [wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD]
        k32.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]
        k32.TerminateJobObject.argtypes = [wintypes.HANDLE, wintypes.UINT]
        k32.CloseHandle.argtypes = [wintypes.HANDLE]
        self.k32 = k32
        self.handle = k32.CreateJobObjectW(None, None)
        if not self.handle:
            raise OSError("CreateJobObject failed")
        info = Extended()
        info.Basic.LimitFlags = 0x2000  # JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
        if not (k32.SetInformationJobObject(self.handle, 9, ctypes.byref(info), ctypes.sizeof(info))
                and k32.AssignProcessToJobObject(self.handle, int(proc._handle))):
            self.close()
            raise OSError("could not put the command in a job object")

    def kill(self) -> None:
        if self.handle:
            self.k32.TerminateJobObject(self.handle, 1)

    def close(self) -> None:
        if self.handle:
            self.k32.CloseHandle(self.handle)  # kills whatever is still in the job
            self.handle = None


def _contain(proc: subprocess.Popen) -> "_WindowsJob | None":
    if os.name != "nt":
        return None  # POSIX: the command leads its own process group instead
    try:
        return _WindowsJob(proc)
    except (OSError, AttributeError, ValueError):
        return None


def _kill_tree(proc: subprocess.Popen, job: "_WindowsJob | None" = None) -> None:
    """Kill a command and everything it started.

    Killing only the shell is not enough: its children (a `sleep`, a dev server)
    keep the output pipes open, and reading them then blocks until they exit on
    their own, which for a server is never.
    """
    if job is not None:
        job.kill()
    elif os.name == "nt":
        subprocess.run(["taskkill", "/F", "/T", "/PID", str(proc.pid)], capture_output=True)
    else:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass
    proc.kill()
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        pass
    for pipe in (proc.stdout, proc.stderr):
        if pipe:
            try:
                pipe.close()
            except OSError:
                pass


class Toolbox:
    def __init__(self, cfg: Config, approve: Approver, *, skills: dict[str, Skill], todos: TodoList,
                 sandbox: Sandbox, subagent: Callable[[str], str] | None = None,
                 read_only: bool = False, spill_prefix: str = "output"):
        self.cfg = cfg
        self.workspace = cfg.workspace.resolve()
        self.approve = lambda question: approve(printable(question))
        self.skills = skills
        self.todos = todos
        self.sandbox = sandbox
        self.subagent = subagent
        # A read-only toolbox (the subagent's) never asks the user anything: what
        # would need a yes is simply refused.
        self.read_only = read_only
        self.spill_prefix = spill_prefix
        self.shell_argv, self.shell_name = find_shell(cfg.shell)
        self.cmd_shell = Path(self.shell_argv[0]).name.lower() in ("cmd", "cmd.exe")
        # (mtime_ns, size) of every file at the moment the agent last read or
        # wrote it. context.py reports drift; str_replace refuses stale edits.
        self.seen: dict[Path, tuple[int, int]] = {}
        # Skills may point at files in their own folder (scripts, references);
        # reading those never needs a prompt.
        self.readable_roots = [self.workspace, *{s.path.parent.resolve() for s in skills.values()}]
        self.spill_dir = self.workspace / ".harness" / "spill"
        self._spill_ids = itertools.count(1)
        self.tools: dict[str, Tool] = {t.name: t for t in self._build()}

    def read_only_copy(self) -> "Toolbox":
        """The subagent's toolbox: same workspace and skills, its own record of
        what it has read, no writes, no questions to the user."""
        return Toolbox(self.cfg, lambda question: False, skills=self.skills, todos=TodoList(),
                       sandbox=self.sandbox, read_only=True, spill_prefix="subagent-output")

    # ---- registry -------------------------------------------------------

    def _build(self) -> list[Tool]:
        s = {"type": "string"}
        return [
            Tool("bash",
                 f"Run a shell command in the workspace; returns stdout, stderr and exit code. Shell: {self.shell_name}. "
                 "Read-only commands run at once; others ask the user first.",
                 _params(["command"], command=s,
                         timeout={"type": "integer", "description": f"seconds, default 120, max {MAX_BASH_TIMEOUT}"}),
                 self.bash),
            Tool("read_file",
                 "Read a text file (path relative to the workspace). For big files pass offset and limit in lines.",
                 _params(["path"], path=s, offset={"type": "integer", "description": "first line, 1-based"},
                         limit={"type": "integer", "description": "number of lines"}),
                 self.read_file),
            Tool("write_file",
                 "Create a file, or overwrite one you have read. Creates parent folders. Workspace only.",
                 _params(["path", "content"], path=s, content=s),
                 self.write_file),
            Tool("str_replace",
                 "Edit a file you have read: replace old_string with new_string. old_string must occur exactly once "
                 "unless replace_all is true; include surrounding lines to make it unique.",
                 _params(["path", "old_string", "new_string"], path=s, old_string=s, new_string=s,
                         replace_all={"type": "boolean"}),
                 self.str_replace),
            Tool("read_skill",
                 "Load the full instructions of a skill listed in the system prompt.",
                 _params(["name"], name=s),
                 self.read_skill),
            Tool("write_todos",
                 "Replace the todo list. Send the whole list each time. Keep one item in_progress while working.",
                 _params(["todos"], todos={"type": "array", "items": {
                     "type": "object", "required": ["content", "status"],
                     "properties": {"content": s, "status": {"type": "string", "enum": ["pending", "in_progress", "completed"]}},
                 }}),
                 self.write_todos),
            Tool("task",
                 "Give a self-contained exploration question to a subagent with a fresh context. It can only read "
                 "(read-only bash commands, read_file, read_skill) and returns just its final answer. It cannot see "
                 "this conversation, so put everything it needs in the prompt.",
                 _params(["prompt"], prompt=s),
                 self.task),
        ]

    def schemas(self, names: tuple[str, ...] | None = None) -> list[dict]:
        return [t.schema() for n, t in self.tools.items() if names is None or n in names]

    def call(self, name: str, arguments: str) -> str:
        tool = self.tools.get(name)
        if tool is None:
            return f"error: unknown tool {name!r}. Available: {', '.join(self.tools)}"
        try:
            args = json.loads(arguments or "{}")
        except json.JSONDecodeError as e:
            return f"error: arguments for {name} are not valid JSON ({e.msg}). Send one JSON object."
        if not isinstance(args, dict):
            return f"error: arguments for {name} must be a JSON object"
        schema = tool.parameters
        missing = [k for k in schema["required"] if k not in args]
        unknown = [k for k in args if k not in schema["properties"]]
        wrong = [f"{k} must be {schema['properties'][k]['type']}" for k, v in args.items()
                 if k in schema["properties"] and not _json_type_ok(v, schema["properties"][k].get("type"))]
        if missing or unknown or wrong:
            parts = (([f"missing {', '.join(missing)}"] if missing else [])
                     + ([f"unknown {', '.join(unknown)}"] if unknown else []) + wrong)
            return f"error: bad arguments for {name}: {'; '.join(parts)}"
        try:
            result = tool.run(**args)
        except ToolError as e:
            return f"error: {e}"
        except OSError as e:
            return f"error: {e.strerror or e} ({getattr(e, 'filename', '') or name})"
        except (ValueError, TypeError) as e:  # e.g. offset="abc"
            return f"error: bad arguments for {name}: {e}"
        return self.cap(result)

    # ---- output capping -------------------------------------------------

    def cap(self, text: str) -> str:
        """Show the model the head of a long result; keep the rest on disk.

        One `cat` of a log should not fill a context window. The full output goes
        to a file inside the workspace, so the model can page it with head, tail,
        sed or grep, and the file is deleted when the turn ends.
        """
        limit = self.cfg.output_cap
        if len(text) <= limit:
            return text
        if self.read_only or not self._spill_dir_is_safe():
            # A read-only toolbox (the subagent's) writes nothing, spill files included.
            return (f"{text[:limit]}\n\n[truncated: showing {limit} of {len(text)} characters; the rest was not kept. "
                    "Narrow the command, or read the file in parts with offset and limit.]")
        self.spill_dir.mkdir(parents=True, exist_ok=True)
        path = self.spill_dir / f"{self.spill_prefix}-{next(self._spill_ids)}.txt"
        path.write_text(text, encoding="utf-8", newline="")
        rel = path.relative_to(self.workspace).as_posix()
        return (f"{text[:limit]}\n\n[truncated: showing {limit} of {len(text)} characters. "
                f"Full output: {rel}. Page it with head, tail, sed -n or grep. Deleted when this turn ends.]")

    def _spill_dir_is_safe(self) -> bool:
        """.harness could be a symlink or junction pointing elsewhere; then the
        spill would be written, and later deleted, outside the workspace."""
        try:
            return self.spill_dir.resolve() == self.spill_dir and self.spill_dir.parent.resolve() == self.spill_dir.parent
        except OSError:
            return False

    def cleanup_spill(self) -> None:
        if self._spill_dir_is_safe():
            shutil.rmtree(self.spill_dir, ignore_errors=True)
        self._spill_ids = itertools.count(1)

    # ---- paths ----------------------------------------------------------

    def _resolve(self, path: str) -> Path:
        p = Path(path).expanduser()
        return (p if p.is_absolute() else self.workspace / p).resolve()

    def _inside(self, p: Path) -> bool:
        return p == self.workspace or p.is_relative_to(self.workspace)

    def _show(self, p: Path) -> str:
        return p.relative_to(self.workspace).as_posix() if self._inside(p) else str(p)

    def _ask(self, question: str) -> bool:
        return not self.read_only and self.approve(question)

    def _check_read(self, p: Path) -> None:
        inside = any(p == root or p.is_relative_to(root) for root in self.readable_roots)
        if not inside and not self._ask(f"read outside the workspace: {p}"):
            raise ToolError("reading a file outside the workspace was not approved")
        if inside and permissions.is_secret_file(p.name) and not self._ask(f"read a credentials file: {p}"):
            raise ToolError(f"{self._show(p)} looks like a credentials file; reading it was not approved")

    def _check_write(self, p: Path) -> None:
        if self.read_only:
            raise ToolError("this agent can only read")
        if not self._inside(p):
            raise ToolError(f"{p} is outside the workspace ({self.workspace}); writes are confined to it")
        rel = p.relative_to(self.workspace).parts
        if os.name == "nt" and any(part != part.rstrip(". ") for part in rel):
            # Windows drops a trailing dot or space when it creates a name, so
            # `.env.` becomes .env and `.agents./x` lands in .agents, past every
            # name check below, which would see the undotted name only after.
            raise ToolError(f"{self._show(p)}: a name ending in a dot or space is not allowed on Windows")
        top = rel[0].lower() if rel else ""
        blocked = next((part for part in rel if part.lower() in BLOCKED_DIRS), None)
        if blocked:  # at any depth: sub/.git/hooks runs code as surely as .git/hooks
            raise ToolError(f"{self._show(p)} is inside {blocked}/, which the harness does not let tools change")
        if (top in GUARDED_DIRS or permissions.is_secret_file(p.name)) and not self._ask(
                f"change {self._show(p)} (it configures the harness or holds credentials)"):
            raise ToolError(f"changing {self._show(p)} was not approved")

    def _is_spill(self, p: Path) -> bool:
        return p.is_relative_to(self.spill_dir)

    def _require_fresh(self, p: Path) -> None:
        if p not in self.seen:
            raise ToolError(f"read {self._show(p)} before changing it")
        if fingerprint(p) != self.seen[p]:
            raise ToolError(f"{self._show(p)} changed on disk since you read it; read it again first")

    @staticmethod
    def _load(p: Path) -> tuple[str, str, str]:
        """(raw text, line-ending style "lf" | "crlf" | "mixed", encoding)."""
        text, encoding = decode(p.read_bytes())
        crlf = text.count("\r\n")
        style = "lf" if crlf == 0 else "crlf" if crlf == text.count("\n") else "mixed"
        return text, style, encoding

    def _save(self, p: Path, text: str, style: str, encoding: str = "utf-8") -> None:
        # The model edits with \n. A CRLF file goes back as CRLF and an LF file as
        # LF, instead of every line changing; a mixed file is written exactly as
        # the edit left it, since there is no single style to restore.
        if style != "mixed":
            text = text.replace("\r\n", "\n")
            if style == "crlf":
                text = text.replace("\n", "\r\n")
        try:
            data = encode(text, encoding)
        except UnicodeEncodeError:
            raise ToolError(f"{self._show(p)} is stored as {encoding}, which cannot hold some of the new "
                            "characters; keep to characters that encoding supports") from None
        p.write_bytes(data)
        self.seen[p] = fingerprint(p)

    # ---- tools ----------------------------------------------------------

    def bash(self, command: str, timeout: int = 120) -> str:
        verdict = permissions.classify(command, self.workspace, cmd_shell=self.cmd_shell)
        if verdict == permissions.DENY:
            raise ToolError("this command is blocked by the harness and will not run")
        if verdict == permissions.ASK:
            if self.read_only:
                raise ToolError("subagents may only run read-only commands inside the workspace")
            if not self.approve(f"run: {command}"):
                raise ToolError("the user declined to run this command")
        timeout = max(1, min(int(timeout), MAX_BASH_TIMEOUT))
        argv = self.sandbox.wrap([*self.shell_argv, command])
        group = ({"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP} if os.name == "nt"
                 else {"start_new_session": True})
        proc = subprocess.Popen(argv, cwd=self.workspace, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                stdin=subprocess.DEVNULL, text=True, encoding="utf-8", errors="replace",
                                env=scrubbed_env(os.environ), **group)
        job = _contain(proc)
        out_cap, err_cap = _Capture(proc.stdout), _Capture(proc.stderr)
        try:
            try:
                proc.wait(timeout=timeout)
                status = f"[exit code {proc.returncode}]"
            except subprocess.TimeoutExpired:
                _kill_tree(proc, job)
                # Keep what it printed: a hung test run's last lines say where it hung.
                status = f"[timed out after {timeout}s and was killed; output so far is above]"
            except BaseException:  # Ctrl-C: do not leave the command running behind the agent
                _kill_tree(proc, job)
                raise
            if not self._drain(out_cap, err_cap, seconds=PIPE_GRACE):
                # The shell is gone but something it started in the background still
                # holds the output. Commands are one-shot here: stop it.
                _kill_tree(proc, job)
                self._drain(out_cap, err_cap, seconds=PIPE_GRACE)
                status += "\n[a background process it started was stopped; commands here cannot outlive their call]"
        finally:
            if job is not None:
                job.close()
        out, err = strip_ansi(out_cap.text()), strip_ansi(err_cap.text())
        if err:
            out += ("\n" if out and not out.endswith("\n") else "") + "[stderr]\n" + err
        return f"{out.rstrip()}\n{status}".lstrip()

    @staticmethod
    def _drain(*captures: _Capture, seconds: float) -> bool:
        """Wait for the readers against one shared deadline; True if all finished."""
        deadline = time.monotonic() + seconds
        for cap in captures:
            cap.thread.join(timeout=max(0.0, deadline - time.monotonic()))
        return not any(cap.thread.is_alive() for cap in captures)

    def read_file(self, path: str, offset: int = 1, limit: int = 2000) -> str:
        p = self._resolve(path)
        self._check_read(p)
        if not p.is_file():
            raise ToolError(f"{self._show(p)} does not exist or is not a file")
        size = p.stat().st_size
        if size > MAX_READ_BYTES:
            raise ToolError(f"{self._show(p)} is {size:,} bytes; look inside it with head, tail, sed -n or grep instead")
        with open(p, "rb") as f:
            start_bytes = f.read(8192)
        if b"\0" in start_bytes and not any(start_bytes.startswith(bom) for bom, _ in _BOMS):
            raise ToolError(f"{self._show(p)} is a binary file ({size:,} bytes); it cannot be shown as text")
        text = self._load(p)[0].replace("\r\n", "\n")
        if not self._is_spill(p):  # spill files vanish at the end of the turn; never track them
            self.seen[p] = fingerprint(p)
        lines = text.split("\n")
        start = max(1, int(offset)) - 1
        if start >= len(lines) and start > 0:
            raise ToolError(f"offset {start + 1} is past the end of {self._show(p)} ({len(lines)} lines)")
        chunk = lines[start:start + max(1, int(limit))]
        body = "\n".join(chunk)
        if start > 0 or start + len(chunk) < len(lines):
            body += f"\n[lines {start + 1}-{start + len(chunk)} of {len(lines)}]"
        return body

    def write_file(self, path: str, content: str) -> str:
        p = self._resolve(path)
        self._check_write(p)
        style, encoding = "lf", "utf-8"
        if p.exists():
            if p.is_dir():
                raise ToolError(f"{self._show(p)} is a folder")
            self._require_fresh(p)
            _, style, encoding = self._load(p)
            if style == "mixed":
                style = "lf"  # a whole new content has no old lines to keep
        p.parent.mkdir(parents=True, exist_ok=True)
        self._save(p, content, style, encoding)
        return f"wrote {len(content)} characters to {self._show(p)}"

    def str_replace(self, path: str, old_string: str, new_string: str, replace_all: bool = False) -> str:
        p = self._resolve(path)
        self._check_write(p)
        if not p.is_file():
            raise ToolError(f"{self._show(p)} does not exist; use write_file to create it")
        self._require_fresh(p)
        if not old_string:
            raise ToolError("old_string is empty")
        if old_string == new_string:
            raise ToolError("old_string and new_string are identical")
        text, style, encoding = self._load(p)
        old, new = old_string.replace("\r\n", "\n"), new_string.replace("\r\n", "\n")
        if style == "mixed":
            # Edit the raw text. read_file showed \n everywhere, so an old_string
            # that spans CRLF lines is tried in its CRLF form too.
            if old not in text and old.replace("\n", "\r\n") in text:
                old, new = old.replace("\n", "\r\n"), new.replace("\n", "\r\n")
        else:
            text = text.replace("\r\n", "\n")
        count = text.count(old)
        if count == 0:
            raise ToolError(f"old_string not found in {self._show(p)}; read the file again and copy it exactly")
        if count > 1 and not replace_all:
            raise ToolError(f"old_string occurs {count} times; add surrounding lines to make it unique, "
                            "or set replace_all to true")
        self._save(p, text.replace(old, new) if replace_all else text.replace(old, new, 1), style, encoding)
        return f"replaced {count if replace_all else 1} occurrence(s) in {self._show(p)}"

    def read_skill(self, name: str) -> str:
        skill = self.skills.get(name)
        if skill is None:
            known = ", ".join(sorted(self.skills)) or "none installed"
            raise ToolError(f"no skill named {name!r} (known: {known})")
        folder = skill.path.parent.resolve().as_posix()
        return (f"[skill folder: {folder}. Paths this skill mentions are relative to it; "
                f"read them with read_file using the full path.]\n\n"
                + skill.path.read_text(encoding="utf-8", errors="replace"))

    def write_todos(self, todos: list) -> str:
        try:
            return self.todos.replace(todos)
        except TodoError as e:
            raise ToolError(str(e)) from None

    def task(self, prompt: str) -> str:
        if self.subagent is None:
            raise ToolError("subagents are not available here")
        try:
            return self.subagent(prompt)
        except LLMError as e:
            # The subagent's model call failed; the main agent can still go on
            # without its answer, so this is a tool error, not the end of the turn.
            raise ToolError(f"the subagent failed: {e}") from None
