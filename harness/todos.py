"""The todo list the model keeps for multi-step work.

The model always rewrites the whole list; the harness shows it back at the end
of every request (context.py), so the plan stays in recent memory however long
the transcript gets.
"""

from __future__ import annotations

from dataclasses import dataclass

STATUSES = ("pending", "in_progress", "completed")
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
        items = []
        for n, entry in enumerate(raw, 1):
            if not isinstance(entry, dict):
                raise TodoError(f"item {n} is not an object")
            content = str(entry.get("content", "")).strip()
            status = entry.get("status", "pending")
            if not content:
                raise TodoError(f"item {n} has no content")
            if status not in STATUSES:
                raise TodoError(f"item {n} has status {status!r}; use one of {', '.join(STATUSES)}")
            items.append(Todo(content, status))
        if sum(t.status == "in_progress" for t in items) > 1:
            raise TodoError("only one item may be in_progress at a time")
        self.items = items
        return self.render()

    def render(self) -> str:
        if not self.items:
            return "(no todos)"
        return "\n".join(f"{MARKS[t.status]} {t.content}" for t in self.items)
