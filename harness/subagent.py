"""Subagents: a fresh context for a self-contained question.

The subagent gets its own message list, read-only tools, and none of the main
conversation. Only its final text comes back, so a long exploration costs the
main transcript a few hundred tokens instead of every file it read. Depth is
one: a subagent cannot start another.
"""

from __future__ import annotations

from typing import Callable

from .llm import LLM
from .tools import Toolbox

SUBAGENT_TOOLS = ("bash", "read_file", "read_skill")

SUBAGENT_SYSTEM = """You are an exploration subagent working in {workspace}.
Answer the question you are given by inspecting files with your tools. You cannot
change files. Finish with a concise, specific answer: paths, names, line numbers,
and anything the caller should know. Your final message is all the caller sees."""


def run_subagent(prompt: str, *, llm: LLM, toolbox: Toolbox, max_steps: int,
                 on_event: Callable[[str, object], None] = lambda kind, data: None) -> str:
    messages: list[dict] = [
        {"role": "system", "content": SUBAGENT_SYSTEM.format(workspace=toolbox.workspace)},
        {"role": "user", "content": prompt},
    ]
    schemas = toolbox.schemas(SUBAGENT_TOOLS)
    on_event("subagent_start", prompt)
    for _ in range(max_steps):
        reply = llm.chat(messages, schemas)
        messages.append(reply.to_message())
        if not reply.tool_calls:
            answer = reply.content.strip() or "(the subagent returned no text)"
            on_event("subagent_end", answer)
            return answer
        for c in reply.tool_calls:
            on_event("subagent_tool_call", c)
            result = (toolbox.call(c.name, c.arguments) if c.name in SUBAGENT_TOOLS
                      else f"error: {c.name} is not available to subagents; you can only read")
            messages.append({"role": "tool", "tool_call_id": c.id, "name": c.name, "content": result})
    on_event("subagent_end", None)
    return f"(the subagent stopped after {max_steps} steps without a final answer)"
