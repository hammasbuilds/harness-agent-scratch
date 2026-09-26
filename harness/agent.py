"""The agent: an LLM called in a loop, with its tool results fed back in.

    user message -> model -> tool calls? -> run them, append results -> model -> ...
                                     no  -> the reply is the answer

The request is laid out so that the start never changes between calls:

    [system prompt (fixed for the session)] [transcript, append-only] [reminder]

Everything that varies (date, todos, stale files) sits in the reminder at the end
and is never stored. Three deliberate exceptions change earlier messages, each
rarely: compaction rewrites the start once when the context fills; old tool
outputs are trimmed once when a new user turn begins; and when the current turn
alone outgrows the budget, its tool outputs and finished call arguments are cut.
"""

from __future__ import annotations

import json
import platform
from datetime import date
from typing import Callable

from .compaction import compact, estimate_tokens, text_tokens
from .config import Config
from .context import build_reminder, git_summary
from .llm import LLM, ContextOverflow
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
REPEAT_WARNING = 3  # identical call + identical result this many times in a row
MIN_KEEP_TOKENS = 256
SUMMARY_SHARE = 0.10  # the handoff note may use at most this share of the context
COMPACT_OVERHEAD = 400  # tokens for the summariser's instructions and the message framing


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
        self._recent_calls: list[tuple[str, str, str]] = []
        self._schema_tokens = estimate_tokens(self.toolbox.schemas())  # fixed for the session
        self._git: str | None = None
        self._git_fresh = False

    def system_message(self) -> dict:
        content = self.system
        if self.summary:
            content += f"\n\n# Handoff note from earlier in this session\n{self.summary}"
        return {"role": "system", "content": content}

    def reminder(self) -> str:
        # git is asked once per step, not once per token estimate: each call is
        # a process start, about 100 ms on Windows.
        if not self._git_fresh:
            self._git, self._git_fresh = git_summary(self.toolbox.workspace), True
        return build_reminder(workspace=self.toolbox.workspace, todos=self.todos, seen=self.toolbox.seen,
                              today=self.today(), git=self._git)

    def overhead_tokens(self) -> int:
        """What every request carries besides the transcript: the system prompt,
        the tool schemas (about 900 tokens, more than a short transcript) and
        the reminder. Leaving these out would let the real prompt pass num_ctx
        before compaction ever triggers."""
        return (estimate_tokens([self.system_message(), {"role": "user", "content": self.reminder()}])
                + self._schema_tokens)

    def prompt_room(self) -> int:
        """Tokens the prompt may use. The context holds the reply too; a prompt that
        leaves no room for it makes Ollama shift the context mid-reply, dropping the
        start of the prompt."""
        return self.cfg.context_limit - self.cfg.max_output_tokens

    def prompt_budget(self) -> float:
        """Where compaction and squeezing start: compact_at of the context, and never
        past the room left for the reply."""
        return min(self.cfg.compact_at * self.cfg.context_limit, self.prompt_room())

    def context_tokens(self) -> int:
        return self.overhead_tokens() + estimate_tokens(self.messages)

    def send(self, text: str) -> str:
        """Run one user turn to completion and return the final reply."""
        self._trim_old_tool_outputs()
        self.messages.append({"role": "user", "content": text})
        self._recent_calls: list[tuple[str, str, str]] = []
        try:
            for _ in range(self.cfg.max_steps):
                self._git_fresh = False  # the last step's commands may have committed or switched branch
                self._maybe_compact()
                request = [self.system_message(), *self.messages, {"role": "user", "content": self.reminder()}]
                self._forget_deleted_files()
                reply = self.llm.chat(request, self.toolbox.schemas())
                self.on_event("usage", reply.usage)
                if not reply.tool_calls:
                    content = reply.content
                    if reply.truncated:
                        content += "\n[reply cut off: the model reached its output limit]"
                    elif not content.strip():
                        content = "[the model returned an empty reply]"
                    self.messages.append({"role": "assistant", "content": content})
                    return content
                self.messages.append(reply.to_message())
                if reply.content:
                    self.on_event("assistant_text", reply.content)
                self._run_calls(reply.tool_calls)
            return f"[stopped after {self.cfg.max_steps} steps without a final answer]"
        finally:
            self.toolbox.cleanup_spill()

    def _run_calls(self, calls: list) -> None:
        done = 0
        try:
            for c in calls:
                self.on_event("tool_call", c)
                result = self._note_repeats(c, self.toolbox.call(c.name, c.arguments))
                self.on_event("tool_result", (c.name, result))
                self.messages.append({"role": "tool", "tool_call_id": c.id, "name": c.name, "content": result})
                done += 1
        finally:
            # Ctrl-C mid-call must not leave a tool call without a result: hosted
            # APIs reject every later request whose history contains one.
            for c in calls[done:]:
                self.messages.append({"role": "tool", "tool_call_id": c.id, "name": c.name,
                                      "content": "error: interrupted by the user before this call finished"})

    def _note_repeats(self, c, result: str) -> str:
        """Small models get stuck repeating one call. Say so in the result, where
        the model is looking, instead of silently burning steps."""
        key = (c.name, c.arguments, result)
        self._recent_calls.append(key)
        run = 0
        for previous in reversed(self._recent_calls):
            if previous != key:
                break
            run += 1
        if run >= REPEAT_WARNING:
            result += (f"\n[harness: this exact call has returned this exact result {run} times in a row. "
                       "Repeating it will not change anything; try something different or answer.]")
        return result

    def reset(self) -> None:
        self.messages.clear()
        self.summary = ""
        self.todos.items.clear()
        self.toolbox.seen.clear()

    def _subagent(self, prompt: str) -> str:
        return run_subagent(prompt, llm=self.llm, toolbox=self.toolbox, max_steps=self.cfg.subagent_max_steps,
                            context_limit=self.prompt_room(), on_event=self.on_event)

    def _trim_old_tool_outputs(self) -> None:
        """A finished turn's tool outputs are rarely needed again and are the
        bulk of the transcript. Keep a short head of each so the model still
        knows what it looked at."""
        for m in self.messages:
            text = m.get("content") or ""
            if m["role"] == "tool" and len(text) > STRIPPED_KEEP and not m.get("trimmed"):
                m["content"] = text[:STRIPPED_KEEP] + f"\n[rest of this output ({len(text)} characters) removed after its turn ended]"
                m["trimmed"] = True

    def _forget_deleted_files(self) -> None:
        """A deleted file has just been reported once in the reminder; keeping it
        would repeat "read it again" on every request for a file that is gone."""
        for p in [p for p in self.toolbox.seen if not p.exists()]:
            del self.toolbox.seen[p]

    def _maybe_compact(self) -> None:
        # The fixed part (system prompt, skills, schemas, handoff note, reminder)
        # cannot be compacted or squeezed; if it alone is too big, compacting only
        # burns a model call and the message to the user would blame the wrong thing.
        overhead = self.overhead_tokens()
        if overhead > self.prompt_budget():
            raise ContextOverflow(
                f"the harness's own context (system prompt, skills, tool schemas, handoff note, todo list: "
                f"~{overhead} tokens) leaves no room in the {self.prompt_room()} tokens available. /clear "
                "resets the note and the todos; fewer or shorter skills, or a larger HARNESS_NUM_CTX, also help.")
        self._compact_old_turns()
        self._squeeze_current_turn()

    def _squeeze_current_turn(self) -> None:
        """Compaction only removes finished turns. One turn with many large tool
        results can still overflow on its own, and Ollama would then silently drop
        the start of the prompt, system prompt included. So cut this turn's tool
        outputs to a short head, oldest first, the latest one last."""
        budget = self.prompt_budget()
        tokens = self.context_tokens()
        if tokens <= budget:
            return
        for m in [m for m in self.messages if m["role"] == "tool" and not m.get("trimmed")]:
            text = m.get("content") or ""
            if len(text) > STRIPPED_KEEP:
                before = estimate_tokens([m])
                m["content"] = text[:STRIPPED_KEEP] + f"\n[rest of this output ({len(text)} characters) removed to fit the context]"
                m["trimmed"] = True
                tokens -= before - estimate_tokens([m])  # one message re-measured, not the whole request
                if tokens <= budget:
                    break
        if tokens > budget:
            # Then the arguments of calls that already ran: a 30 KB write_file is
            # as big as any tool result, and the file on disk still has it.
            for m in [m for m in self.messages if m["role"] == "assistant" and m.get("tool_calls")]:
                for tc in m["tool_calls"]:
                    args = tc["function"]["arguments"]
                    if len(args) > STRIPPED_KEEP:
                        before = estimate_tokens([m])
                        tc["function"]["arguments"] = json.dumps(
                            {"_note": f"arguments ({len(args)} characters) removed to fit the context"})
                        tokens -= before - estimate_tokens([m])
                if tokens <= budget:
                    break
        tokens = self.context_tokens()
        self.on_event("squeezed", (tokens, int(budget)))
        if tokens > self.prompt_room():
            # Sending it anyway would let Ollama silently drop the start of the
            # prompt, system prompt included; better to stop and say so.
            raise ContextOverflow(
                f"the request (~{tokens} tokens) does not fit the {self.prompt_room()} tokens the {self.cfg.context_limit}-token "
                f"context leaves beside a {self.cfg.max_output_tokens}-token reply, even after "
                "compacting and trimming; the current message is too large. Start over with /clear, send less at "
                "once, or raise HARNESS_NUM_CTX.")

    def _compact_old_turns(self) -> None:
        limit = self.cfg.context_limit
        if self.context_tokens() <= self.prompt_budget():
            return
        before, count = self.context_tokens(), len(self.messages)
        self.summary, self.messages = compact(
            self.messages, self.summary, self.llm,
            # compact_to is a share of the whole prompt, so the fixed overhead
            # comes out of it before deciding how much transcript to keep.
            keep_tokens=max(MIN_KEEP_TOKENS, int(self.cfg.compact_to * limit) - self.overhead_tokens()),
            # The summarising request must fit beside its reply too: what is left
            # after the previous note and the instructions goes to the transcript.
            budget_tokens=max(MIN_KEEP_TOKENS, self.prompt_room() - text_tokens(self.summary) - COMPACT_OVERHEAD),
            max_summary_tokens=int(limit * SUMMARY_SHARE),
        )
        # Nothing droppable (one oversized message) leaves the transcript as it was.
        if len(self.messages) != count:
            self.on_event("compacted", (before, self.context_tokens()))
