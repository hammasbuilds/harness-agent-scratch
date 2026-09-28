"""Compaction: when the transcript nears the context limit, summarise its start.

At `compact_at` of the limit, the oldest messages are replaced by a handoff note
written by the model, until the rest fits under `compact_to`. The note lives at
the end of the system prompt, so the prefix changes once per compaction instead
of on every call, and the transcript then has room to grow again before the
next one.
"""

from __future__ import annotations

import json
import math
import re
from itertools import pairwise

from .llm import LLM

COMPACT_SYSTEM = """You are compacting the transcript of a coding session that ran out of context.
Write the handoff note that lets a fresh agent continue without rereading everything.
Cover: the user's goal; decisions made and why; files created or changed (exact paths);
what is done; what failed and why; the immediate next steps.
Be specific: exact paths, names, commands and error messages. No preamble."""

TOOL_EXCERPT = 1500


_RUN = re.compile(r"[A-Za-z]+|[0-9]|\s+|[^\sA-Za-z0-9]")
SAFETY = 1.15


_VOWELS = frozenset("aeiouAEIOU")
# Scripts the tokenizer has large vocabularies for: about one token a character.
_CJK_RANGES = (
    (0x3000, 0x30FF),
    (0x3400, 0x4DBF),
    (0x4E00, 0x9FFF),
    (0xAC00, 0xD7AF),
    (0xF900, 0xFAFF),
    (0xFF00, 0xFFEF),
)


def _is_cjk(c: str) -> bool:
    o = ord(c)
    return any(lo <= o <= hi for lo, hi in _CJK_RANGES)


def _letter_tokens(run: str) -> int:
    """Random-looking letters cost about 1.5-1.9 characters a token; words about four.

    Random in mixed case switches case often (base64, API keys). Random in one
    case (DNA, protein sequences, base32, lowercase noise) is short of vowels:
    about 19% against 35-40% for real words. Runs over 24 letters are rarely words.
    """
    n = len(run)
    switches = sum(1 for a, b in pairwise(run) if a.isupper() != b.isupper())
    vowels = sum(c in _VOWELS for c in run) / n
    if (
        (n >= 4 and switches * 3 >= n)
        or (n >= 6 and vowels < 0.28)
        or (3 <= n < 6 and vowels < 0.2)
        or n > 24
    ):
        return math.ceil(n / 1.5)
    return math.ceil(n / 4)


def text_tokens(text: str) -> int:
    """A pessimistic token count, shaped like BPE pre-tokenisation.

    Calibrated against the Qwen2.5 tokenizer on 34 kinds of text (prose, code,
    JSON, CSV, `seq` output, hashes, base64, base32, paths, DNA and protein
    sequences, random letters in either case, long words, CJK, Cyrillic, Hindi,
    Georgian, Thai, Amharic, Hebrew and Arabic, math symbols, Braille spinners,
    emoji): on every one the real count was at most 0.92 of the estimate, which
    is about 1.7x the real count on average. Underestimating is the failure that matters,
    since Ollama silently drops the start of an overlong prompt; overestimating
    only compacts sooner. The flat "characters / 3" this replaced was 3x too low
    on digits (Qwen makes every digit a token) and 2x on JSON; the version before
    this one was 2x too low on single-case random letters such as DNA.
    """
    n = 0
    for m in _RUN.finditer(text):
        run = m.group()
        c = run[0]
        if c.isascii() and c.isalpha():
            n += _letter_tokens(run)
        elif c.isspace():
            n += 1 if "\n" in run else 0  # a space merges into the next word
        elif ord(c) > 0xFFFF:
            n += 3  # emoji and other astral characters take several byte tokens
        elif 0x2000 <= ord(c) <= 0x2BFF:
            n += (
                3  # symbols (arrows, math, box drawing, Braille spinners): one token per UTF-8 byte
            )
        elif ord(c) > 0x7FF and not _is_cjk(c):
            n += 2  # other three-byte scripts, rarely merged: Ethiopic, Devanagari, Georgian
        else:
            n += 1  # each digit, punctuation mark, CJK, Latin-1 or Cyrillic character
    return math.ceil(n * SAFETY)


def estimate_tokens(messages: list[dict]) -> int:
    return text_tokens("".join(json.dumps(m, ensure_ascii=False) for m in messages))


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


def fit_tokens(
    text: str,
    max_tokens: int,
    head_share: float = 0.25,
    marker: str = "\n\n[... middle omitted ...]\n\n",
) -> str:
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
        candidate = text[:head] + marker + text[len(text) - (mid - head) :]
        if text_tokens(candidate) <= max_tokens:
            lo = mid
        else:
            hi = mid - 1
    head = int(lo * head_share)
    return text[:head] + marker + text[len(text) - (lo - head) :] if lo else marker.strip()


def compact(
    messages: list[dict],
    previous_summary: str,
    llm: LLM,
    *,
    keep_tokens: int,
    budget_tokens: int,
    max_summary_tokens: int = 1000,
) -> tuple[str, list[dict]]:
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
        body = (
            f"Handoff note from an earlier compaction:\n{previous_summary}\n\n"
            f"Transcript since then:\n{body}"
        )
    words = max(50, int(max_summary_tokens * 0.6))
    system = f"{COMPACT_SYSTEM}\nKeep the note under {words} words."
    note = llm.chat(
        [{"role": "system", "content": system}, {"role": "user", "content": body}], None
    ).content.strip()
    if not note:  # an empty summary would silently lose everything that was cut
        note = (
            previous_summary
            + ("\n\n" if previous_summary else "")
            + f"[summary failed; earlier transcript excerpt]\n{body}"
        )
    if text_tokens(note) > max_summary_tokens:
        note = fit_tokens(
            note, max_summary_tokens - 12, head_share=1.0, marker="\n[note cut to fit the context]"
        )
    return note, messages[cut:]
