"""The agent: an LLM called in a loop, with its tool results fed back in.

    user message -> model -> tool calls? -> run them, append results -> model -> ...
                                     no  -> the reply is the answer

The request is laid out so that the start never changes between calls:

    [system prompt (fixed for the session)] [transcript, append-only] [reminder]

Everything that varies (date, todos, stale files) sits in the reminder at the end
and is never stored. The two deliberate exceptions are rare: compaction rewrites
the start once when the context fills, and old tool outputs are trimmed once
when a new user turn begins.
"""

from __future__ import annotations

import platform
from datetime import date
from typing import Callable

from .compaction import compact, estimate_tokens
from .config import Config
from .context import build_reminder, git_summary
from .llm import LLM
from .sandbox import Sandbox
from .skills import default_skill_dirs, discover_skills, skills_prompt
from .subagent import run_subagent
from .todos import TodoList
from .tools import Approver, Toolbox

SYSTEM_PROMPT = """You are a coding agent working in the folder {workspace} on {os}.

Work by using tools; never guess what a file contains.
- bash runs commands in the workspace. Shell: {shell}.
- Read a file before changing it. Edit with str_replace; create files with write_file.
- For work with three or more steps, keep a todo list with write_todos and update it as you go.
- To explore a question without filling this conversation, hand it to a subagent with task.
- Long tool output is cut short; the note at its end says where the full text is.
- A <system-reminder> at the end of a request comes from the harness, not the user.
- When the work is done, reply with a short summary of what you changed. Do not call a tool in that reply.

Skills (load one with read_skill when it fits the task):
{skills}"""

STRIPPED_KEEP = 200


class Agent:
    def __init__(self, cfg: Config, llm: LLM, approve: Approver, *,
                 on_event: Callable[[str, object], None] | None = None,
                 today: Callable[[], date] = date.today,
                 sandbox: Sandbox | None = None):
        self.cfg = cfg
        self.llm = llm
        self.on_event = on_event or (lambda kind, data: None)
        self.today = today
        self.todos = TodoList()
        dirs = cfg.skills_dirs if cfg.skills_dirs is not None else default_skill_dirs(cfg.workspace)
        self.skills = discover_skills(dirs)
        self.sandbox = sandbox or Sandbox(cfg.workspace, cfg.sandbox)
        self.toolbox = Toolbox(cfg, approve, skills=self.skills, todos=self.todos, sandbox=self.sandbox,
                               subagent=self._subagent)
        self.system = SYSTEM_PROMPT.format(workspace=self.toolbox.workspace, os=platform.system(),
                                           shell=self.toolbox.shell_name, skills=skills_prompt(self.skills))
        self.messages: list[dict] = []
        self.summary = ""

    def system_message(self) -> dict:
        content = self.system
        if self.summary:
            content += f"\n\n# Handoff note from earlier in this session\n{self.summary}"
        return {"role": "system", "content": content}

    def reminder(self) -> str:
        return build_reminder(workspace=self.toolbox.workspace, todos=self.todos, seen=self.toolbox.seen,
                              today=self.today(), git=git_summary(self.toolbox.workspace))

    def context_tokens(self) -> int:
        return estimate_tokens([self.system_message(), *self.messages])

    def send(self, text: str) -> str:
        """Run one user turn to completion and return the final reply."""
        self._trim_old_tool_outputs()
        self.messages.append({"role": "user", "content": text})
        try:
            for _ in range(self.cfg.max_steps):
                self._maybe_compact()
                request = [self.system_message(), *self.messages, {"role": "user", "content": self.reminder()}]
                reply = self.llm.chat(request, self.toolbox.schemas())
                self.on_event("usage", reply.usage)
                self.messages.append(reply.to_message())
                if not reply.tool_calls:
                    return reply.content
                if reply.content:
                    self.on_event("assistant_text", reply.content)
                for c in reply.tool_calls:
                    self.on_event("tool_call", c)
                    result = self.toolbox.call(c.name, c.arguments)
                    self.on_event("tool_result", (c.name, result))
                    self.messages.append({"role": "tool", "tool_call_id": c.id, "name": c.name, "content": result})
            return f"[stopped after {self.cfg.max_steps} steps without a final answer]"
        finally:
            self.toolbox.cleanup_spill()

    def reset(self) -> None:
        self.messages.clear()
        self.summary = ""
        self.todos.items.clear()
        self.toolbox.seen.clear()

    def _subagent(self, prompt: str) -> str:
        return run_subagent(prompt, llm=self.llm, toolbox=self.toolbox,
                            max_steps=self.cfg.subagent_max_steps, on_event=self.on_event)

    def _trim_old_tool_outputs(self) -> None:
        """A finished turn's tool outputs are rarely needed again and are the
        bulk of the transcript. Keep a short head of each so the model still
        knows what it looked at."""
        for m in self.messages:
            text = m.get("content") or ""
            if m["role"] == "tool" and len(text) > STRIPPED_KEEP and not m.get("trimmed"):
                m["content"] = text[:STRIPPED_KEEP] + f"\n[rest of this output ({len(text)} characters) removed after its turn ended]"
                m["trimmed"] = True

    def _maybe_compact(self) -> None:
        limit = self.cfg.context_limit
        if self.context_tokens() <= self.cfg.compact_at * limit:
            return
        before, count = self.context_tokens(), len(self.messages)
        self.summary, self.messages = compact(
            self.messages, self.summary, self.llm,
            keep_tokens=int(self.cfg.compact_to * limit),
            budget_chars=int(limit * 3 * 0.6),
        )
        # Nothing droppable (one oversized message) leaves the transcript as it was.
        if len(self.messages) != count:
            self.on_event("compacted", (before, self.context_tokens()))
