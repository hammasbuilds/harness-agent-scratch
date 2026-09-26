"""The tools: everything the model can do beyond writing text.

Each tool is a Python function plus a JSON schema the model reads. The toolbox
validates arguments, runs the function, and turns every failure into text the
model can read and react to; one bad call never ends the loop.
"""

from __future__ import annotations

import itertools
import json
import os
import shutil
import signal
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from . import permissions
from .config import Config
from .sandbox import Sandbox, find_shell
from .skills import Skill
from .todos import TodoError, TodoList

Approver = Callable[[str], bool]
MAX_BASH_TIMEOUT = 600


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


def _kill_tree(proc: subprocess.Popen) -> None:
    """Kill a timed-out command and everything it started.

    Killing only the shell is not enough: its children (a `sleep`, a dev server)
    keep the output pipes open, and reading them then blocks until they exit on
    their own, which for a server is never.
    """
    if os.name == "nt":
        subprocess.run(["taskkill", "/F", "/T", "/PID", str(proc.pid)], capture_output=True)
    else:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass
    proc.kill()
    try:
        proc.communicate(timeout=5)
    except subprocess.TimeoutExpired:
        for pipe in (proc.stdout, proc.stderr):
            if pipe:
                pipe.close()


class Toolbox:
    def __init__(self, cfg: Config, approve: Approver, *, skills: dict[str, Skill], todos: TodoList,
                 sandbox: Sandbox, subagent: Callable[[str], str] | None = None):
        self.cfg = cfg
        self.workspace = cfg.workspace.resolve()
        self.approve = approve
        self.skills = skills
        self.todos = todos
        self.sandbox = sandbox
        self.subagent = subagent
        self.shell_argv, self.shell_name = find_shell(cfg.shell)
        # Modification time of every file at the moment the agent last read or
        # wrote it. context.py reports drift; str_replace refuses stale edits.
        self.seen: dict[Path, float] = {}
        self.spill_dir = self.workspace / ".harness" / "spill"
        self._spill_ids = itertools.count(1)
        self.tools: dict[str, Tool] = {t.name: t for t in self._build()}

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
                 "(bash, read_file, read_skill) and returns just its final answer. It cannot see this conversation, "
                 "so put everything it needs in the prompt.",
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
        if missing or unknown:
            parts = ([f"missing {', '.join(missing)}"] if missing else []) + ([f"unknown {', '.join(unknown)}"] if unknown else [])
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
        self.spill_dir.mkdir(parents=True, exist_ok=True)
        path = self.spill_dir / f"output-{next(self._spill_ids)}.txt"
        path.write_text(text, encoding="utf-8", newline="")
        rel = path.relative_to(self.workspace).as_posix()
        return (f"{text[:limit]}\n\n[truncated: showing {limit} of {len(text)} characters. "
                f"Full output: {rel}. Page it with head, tail, sed -n or grep. Deleted when this turn ends.]")

    def cleanup_spill(self) -> None:
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

    def _require_writable(self, p: Path) -> None:
        if not self._inside(p):
            raise ToolError(f"{p} is outside the workspace ({self.workspace}); writes are confined to it")

    def _require_fresh(self, p: Path) -> None:
        if p not in self.seen:
            raise ToolError(f"read {self._show(p)} before changing it")
        if p.stat().st_mtime != self.seen[p]:
            raise ToolError(f"{self._show(p)} changed on disk since you read it; read it again first")

    @staticmethod
    def _load(p: Path) -> tuple[str, bool]:
        """Text with \\n line endings, and whether the file used \\r\\n."""
        with open(p, encoding="utf-8", errors="replace", newline="") as f:
            raw = f.read()
        return raw.replace("\r\n", "\n"), "\r\n" in raw

    def _save(self, p: Path, text: str, crlf: bool) -> None:
        # newline="" writes exactly what we give it; the model edits with \n and
        # a CRLF file stays CRLF instead of silently changing every line.
        with open(p, "w", encoding="utf-8", newline="") as f:
            f.write(text.replace("\n", "\r\n") if crlf else text)
        self.seen[p] = p.stat().st_mtime

    # ---- tools ----------------------------------------------------------

    def bash(self, command: str, timeout: int = 120) -> str:
        verdict = permissions.classify(command)
        if verdict == permissions.DENY:
            raise ToolError("this command is blocked by the harness and will not run")
        if verdict == permissions.ASK and not self.approve(f"run: {command}"):
            raise ToolError("the user declined to run this command")
        timeout = max(1, min(int(timeout), MAX_BASH_TIMEOUT))
        argv = self.sandbox.wrap([*self.shell_argv, command])
        group = ({"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP} if os.name == "nt"
                 else {"start_new_session": True})
        proc = subprocess.Popen(argv, cwd=self.workspace, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                stdin=subprocess.DEVNULL, text=True, encoding="utf-8", errors="replace", **group)
        try:
            stdout, stderr = proc.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            _kill_tree(proc)
            raise ToolError(f"command timed out after {timeout}s") from None
        out = stdout
        if stderr:
            out += ("\n" if out and not out.endswith("\n") else "") + "[stderr]\n" + stderr
        return f"{out.rstrip()}\n[exit code {proc.returncode}]".lstrip()

    def read_file(self, path: str, offset: int = 1, limit: int = 2000) -> str:
        p = self._resolve(path)
        if not self._inside(p) and not self.approve(f"read outside the workspace: {p}"):
            raise ToolError("the user declined reading a file outside the workspace")
        if not p.is_file():
            raise ToolError(f"{self._show(p)} does not exist or is not a file")
        text, _ = self._load(p)
        self.seen[p] = p.stat().st_mtime
        lines = text.split("\n")
        start = max(1, int(offset)) - 1
        chunk = lines[start:start + max(1, int(limit))]
        body = "\n".join(chunk)
        if start > 0 or start + len(chunk) < len(lines):
            body += f"\n[lines {start + 1}-{start + len(chunk)} of {len(lines)}]"
        return body

    def write_file(self, path: str, content: str) -> str:
        p = self._resolve(path)
        self._require_writable(p)
        crlf = False
        if p.exists():
            if p.is_dir():
                raise ToolError(f"{self._show(p)} is a folder")
            self._require_fresh(p)
            _, crlf = self._load(p)
        p.parent.mkdir(parents=True, exist_ok=True)
        self._save(p, content, crlf)
        return f"wrote {len(content)} characters to {self._show(p)}"

    def str_replace(self, path: str, old_string: str, new_string: str, replace_all: bool = False) -> str:
        p = self._resolve(path)
        self._require_writable(p)
        if not p.is_file():
            raise ToolError(f"{self._show(p)} does not exist; use write_file to create it")
        self._require_fresh(p)
        if not old_string:
            raise ToolError("old_string is empty")
        if old_string == new_string:
            raise ToolError("old_string and new_string are identical")
        text, crlf = self._load(p)
        old, new = old_string.replace("\r\n", "\n"), new_string.replace("\r\n", "\n")
        count = text.count(old)
        if count == 0:
            raise ToolError(f"old_string not found in {self._show(p)}; read the file again and copy it exactly")
        if count > 1 and not replace_all:
            raise ToolError(f"old_string occurs {count} times; add surrounding lines to make it unique, "
                            "or set replace_all to true")
        self._save(p, text.replace(old, new) if replace_all else text.replace(old, new, 1), crlf)
        return f"replaced {count if replace_all else 1} occurrence(s) in {self._show(p)}"

    def read_skill(self, name: str) -> str:
        skill = self.skills.get(name)
        if skill is None:
            known = ", ".join(sorted(self.skills)) or "none installed"
            raise ToolError(f"no skill named {name!r} (known: {known})")
        return skill.path.read_text(encoding="utf-8", errors="replace")

    def write_todos(self, todos: list) -> str:
        try:
            return self.todos.replace(todos)
        except TodoError as e:
            raise ToolError(str(e)) from None

    def task(self, prompt: str) -> str:
        if self.subagent is None:
            raise ToolError("subagents are not available here")
        return self.subagent(prompt)
