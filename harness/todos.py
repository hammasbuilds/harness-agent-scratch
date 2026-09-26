"""The todo list the model keeps for multi-step work.

The model always rewrites the whole list; the harness shows it back at the end
of every request (context.py), so the plan stays in recent memory however long
the transcript gets.
"""

from __future__ import annotations

from dataclasses import dataclass

STATUSES = ("pending", "in_progress", "completed")
# The list rides in every request's reminder, which cannot be trimmed; an
# unbounded one (80 items of 400 characters) made every later request overflow.
MAX_ITEMS = 20
MAX_ITEM_CHARS = 120
MARKS = {"pending": "[ ]", "in_progress": "[>]", "completed": "[x]"}


class TodoError(ValueError):
    pass


@dataclass
class Todo:
    content: str
    status: str


class TodoList:
    def __init__(self):
        self.items: list[Todo] = []

    def replace(self, raw: list) -> str:
        if not isinstance(raw, list):
            raise TodoError("todos must be a list of {content, status} objects")
        if len(raw) > MAX_ITEMS:
            raise TodoError(f"at most {MAX_ITEMS} todos; group smaller steps together")
        items = []
        for n, entry in enumerate(raw, 1):
            if not isinstance(entry, dict):
                raise TodoError(f"item {n} is not an object")
            content = str(entry.get("content", "")).strip()
            status = entry.get("status", "pending")
            if not content:
                raise TodoError(f"item {n} has no content")
            if len(content) > MAX_ITEM_CHARS:
                raise TodoError(f"item {n} is {len(content)} characters; keep each under {MAX_ITEM_CHARS}")
            if status not in STATUSES:
                raise TodoError(f"item {n} has status {status!r}; use one of {', '.join(STATUSES)}")
            items.append(Todo(content, status))
        if sum(t.status == "in_progress" for t in items) > 1:
            raise TodoError("only one item may be in_progress at a time")
        self.items = items
        return self.render()

    def render_open(self) -> str:
        """What the reminder shows: open items only, since finished ones need no
        attention and every character here is paid for on every request."""
        open_items = [t for t in self.items if t.status != "completed"]
        done = len(self.items) - len(open_items)
        lines = [f"{MARKS[t.status]} {t.content}" for t in open_items]
        if done:
            lines.append(f"({done} completed)")
        return "\n".join(lines) or "(no todos)"

    def render(self) -> str:
        if not self.items:
            return "(no todos)"
        return "\n".join(f"{MARKS[t.status]} {t.content}" for t in self.items)
