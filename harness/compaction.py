"""Compaction: when the transcript nears the context limit, summarise its start.

At `compact_at` of the limit, the oldest messages are replaced by a handoff note
written by the model, until the rest fits under `compact_to`. The note lives at
the end of the system prompt, so the prefix changes once per compaction instead
of on every call, and the transcript then has room to grow again before the
next one.
"""

from __future__ import annotations

import json

from .llm import LLM

COMPACT_SYSTEM = """You are compacting the transcript of a coding session that ran out of context.
Write the handoff note that lets a fresh agent continue without rereading everything.
Cover: the user's goal; decisions made and why; files created or changed (exact paths);
what is done; what failed and why; the immediate next steps.
Be specific: exact paths, names, commands and error messages. No preamble."""

TOOL_EXCERPT = 1500


def estimate_tokens(messages: list[dict]) -> int:
    """About three characters per token: deliberately pessimistic, because
    underestimating means Ollama silently drops the start of the prompt."""
    return sum(len(json.dumps(m, ensure_ascii=False)) for m in messages) // 3


def choose_cut(messages: list[dict], keep_tokens: int) -> int:
    """Index of the first message to keep; 0 means nothing can go.

    Cuts only at a user message, where a turn begins, so a tool result is never
    separated from the call that asked for it (APIs reject that). And only
    finished turns go: the current turn, from the last user message on, is never
    cut, because it holds the request being worked on. If the current turn alone
    is too big, the agent shortens its tool outputs instead.
    """
    users = [i for i in range(1, len(messages)) if messages[i]["role"] == "user"]
    for i in users:
        if estimate_tokens(messages[i:]) <= keep_tokens:
            return i
    return users[-1] if users else 0


def render_for_summary(messages: list[dict], budget_chars: int) -> str:
    parts = []
    for m in messages:
        role = m["role"]
        if role == "tool":
            text = m.get("content") or ""
            if len(text) > TOOL_EXCERPT:
                text = text[:TOOL_EXCERPT] + " [...]"
            parts.append(f"TOOL RESULT ({m.get('name', '?')}):\n{text}")
        else:
            line = f"{role.upper()}: {m.get('content') or ''}".rstrip()
            for tc in m.get("tool_calls") or []:
                line += f"\n  -> called {tc['function']['name']} {tc['function']['arguments']}"
            parts.append(line)
    text = "\n\n".join(parts)
    if len(text) > budget_chars:
        # Keep the opening (the goal) and the most recent work; drop the middle.
        head = budget_chars // 4
        text = text[:head] + "\n\n[... middle of transcript omitted ...]\n\n" + text[-(budget_chars - head):]
    return text


def compact(messages: list[dict], previous_summary: str, llm: LLM, *, keep_tokens: int,
            budget_chars: int, max_summary_chars: int = 4000) -> tuple[str, list[dict]]:
    """Return (new handoff note, messages to keep).

    The note is capped: it sits in every later request, and each compaction
    folds the previous note into the next, so an uncapped note would grow
    until it alone filled the context.
    """
    cut = choose_cut(messages, keep_tokens)
    if cut == 0:
        return previous_summary, messages
    body = render_for_summary(messages[:cut], budget_chars)
    if previous_summary:
        body = f"Handoff note from an earlier compaction:\n{previous_summary}\n\nTranscript since then:\n{body}"
    words = max(50, max_summary_chars // 7)
    system = f"{COMPACT_SYSTEM}\nKeep the note under {words} words."
    note = llm.chat([{"role": "system", "content": system}, {"role": "user", "content": body}], None).content.strip()
    if not note:  # an empty summary would silently lose everything that was cut
        note = previous_summary + ("\n\n" if previous_summary else "") + f"[summary failed; earlier transcript excerpt]\n{body}"
    if len(note) > max_summary_chars:
        note = note[:max_summary_chars] + "\n[note cut to fit the context]"
    return note, messages[cut:]
