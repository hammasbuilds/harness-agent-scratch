"""Subagents: a fresh context for a self-contained question.

The subagent gets its own message list, a read-only toolbox (read-only bash
commands inside the workspace, read_file, read_skill; it never asks the user
anything), and none of the main conversation. Only its final text comes back,
so a long exploration costs the main transcript a few hundred tokens instead of
every file it read. Depth is one: a subagent cannot start another.

It has no compaction of its own. Instead it keeps a budget: past TRIM_AT of the
context limit its older tool outputs are cut to a short head, and past
FINISH_AT it must answer with what it has.
"""

from __future__ import annotations

from typing import Callable

from .compaction import estimate_tokens
from .llm import LLM
from .tools import Toolbox

SUBAGENT_TOOLS = ("bash", "read_file", "read_skill")
TRIM_AT = 0.6
FINISH_AT = 0.8
TRIM_KEEP = 200

SUBAGENT_SYSTEM = """You are an exploration subagent working in {workspace}.
Answer the question you are given by inspecting files with your tools. You cannot
change files or run commands that change anything. Finish with a concise,
specific answer: paths, names, line numbers, and anything the caller should know.
Your final message is all the caller sees."""

OUT_OF_ROOM = ("<system-reminder>Your context is nearly full. Stop exploring and answer now "
               "with what you have found; say what you could not check.</system-reminder>")


def _trim_older_outputs(messages: list[dict]) -> None:
    tool_msgs = [m for m in messages if m["role"] == "tool"]
    for m in tool_msgs[:-1]:
        text = m.get("content") or ""
        if len(text) > TRIM_KEEP and not m.get("trimmed"):
            m["content"] = text[:TRIM_KEEP] + f"\n[rest of this output ({len(text)} characters) removed to save room]"
            m["trimmed"] = True


def run_subagent(prompt: str, *, llm: LLM, toolbox: Toolbox, max_steps: int, context_limit: int,
                 on_event: Callable[[str, object], None] = lambda kind, data: None) -> str:
    box = toolbox if toolbox.read_only else toolbox.read_only_copy()
    messages: list[dict] = [
        {"role": "system", "content": SUBAGENT_SYSTEM.format(workspace=box.workspace)},
        {"role": "user", "content": prompt},
    ]
    schemas = box.schemas(SUBAGENT_TOOLS)
    schema_tokens = estimate_tokens(schemas)
    on_event("subagent_start", prompt)
    for _ in range(max_steps):
        if schema_tokens + estimate_tokens(messages) > TRIM_AT * context_limit:
            _trim_older_outputs(messages)
        if schema_tokens + estimate_tokens(messages) > FINISH_AT * context_limit:
            reply = llm.chat([*messages, {"role": "user", "content": OUT_OF_ROOM}], None)
            return _finish(reply.content, on_event)
        reply = llm.chat(messages, schemas)
        messages.append(reply.to_message())
        if not reply.tool_calls:
            return _finish(reply.content, on_event)
        for c in reply.tool_calls:
            on_event("subagent_tool_call", c)
            result = (box.call(c.name, c.arguments) if c.name in SUBAGENT_TOOLS
                      else f"error: {c.name} is not available to subagents; you can only read")
            messages.append({"role": "tool", "tool_call_id": c.id, "name": c.name, "content": result})
    on_event("subagent_end", None)
    return f"(the subagent stopped after {max_steps} steps without a final answer)"


def _finish(content: str, on_event) -> str:
    answer = content.strip() or "(the subagent returned no text)"
    on_event("subagent_end", answer)
    return answer
