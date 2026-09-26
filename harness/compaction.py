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
    """Three ASCII characters per token, and one token per other character:
    deliberately pessimistic, because underestimating means Ollama silently
    drops the start of the prompt. (A flat chars/3 put 4,000 CJK characters at
    1,343 tokens; real tokenizers give about one per character.)"""
    text = "".join(json.dumps(m, ensure_ascii=False) for m in messages)
    wide = sum(1 for ch in text if ord(ch) > 127)
    return (len(text) - wide) // 3 + wide


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


def render_for_summary(messages: list[dict]) -> str:
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
    return "\n\n".join(parts)


def text_tokens(text: str) -> int:
    return estimate_tokens([{"c": text}])


def fit_tokens(text: str, max_tokens: int, head_share: float = 0.25,
               marker: str = "\n\n[... middle omitted ...]\n\n") -> str:
    """Cut `text` to about `max_tokens`, keeping its start and its end.

    Measured in tokens, not characters: a character cap sized for English let a
    Chinese handoff note fill 45% of the context and forced a compaction every turn.
    """
    if text_tokens(text) <= max_tokens:
        return text
    lo, hi = 0, len(text)  # the largest number of characters that fits
    while lo < hi:
        mid = (lo + hi + 1) // 2
        head = int(mid * head_share)
        candidate = text[:head] + marker + text[len(text) - (mid - head):]
        if text_tokens(candidate) <= max_tokens:
            lo = mid
        else:
            hi = mid - 1
    head = int(lo * head_share)
    return text[:head] + marker + text[len(text) - (lo - head):] if lo else marker.strip()


def compact(messages: list[dict], previous_summary: str, llm: LLM, *, keep_tokens: int,
            budget_tokens: int, max_summary_tokens: int = 1000) -> tuple[str, list[dict]]:
    """Return (new handoff note, messages to keep).

    The note is capped: it sits in every later request, and each compaction
    folds the previous note into the next, so an uncapped note would grow
    until it alone filled the context. The transcript sent for summarising is
    capped too, so the compaction request itself fits the context.
    """
    cut = choose_cut(messages, keep_tokens)
    if cut == 0:
        return previous_summary, messages
    # Keep the opening (the goal) and the most recent work; drop the middle.
    body = fit_tokens(render_for_summary(messages[:cut]), budget_tokens)
    if previous_summary:
        body = f"Handoff note from an earlier compaction:\n{previous_summary}\n\nTranscript since then:\n{body}"
    words = max(50, int(max_summary_tokens * 0.6))
    system = f"{COMPACT_SYSTEM}\nKeep the note under {words} words."
    note = llm.chat([{"role": "system", "content": system}, {"role": "user", "content": body}], None).content.strip()
    if not note:  # an empty summary would silently lose everything that was cut
        note = previous_summary + ("\n\n" if previous_summary else "") + f"[summary failed; earlier transcript excerpt]\n{body}"
    if text_tokens(note) > max_summary_tokens:
        note = fit_tokens(note, max_summary_tokens - 12, head_share=1.0, marker="\n[note cut to fit the context]")
    return note, messages[cut:]
