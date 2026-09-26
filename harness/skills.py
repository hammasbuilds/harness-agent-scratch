"""Skills: markdown instruction files the model loads only when it needs them.

Each skill is a folder holding SKILL.md, which starts with front matter:

    ---
    name: frontend-design
    description: Build distinctive, production-grade web interfaces.
    ---
    <instructions>

Only name + description go into the system prompt. The body costs nothing
until the model calls read_skill.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

BLOCK_MARKERS = {">", "|", ">-", "|-", ">+", "|+", ""}


@dataclass
class Skill:
    name: str
    description: str
    path: Path


def default_skill_dirs(workspace: Path, home: Path | None = None) -> list[Path]:
    home = Path.home() if home is None else home
    return [home / ".agents" / "skills", workspace / ".agents" / "skills"]


def parse_front_matter(text: str) -> dict[str, str] | None:
    """Read the flat `key: value` pairs between the opening `---` lines.

    Handles quoted values and folded/literal blocks (`description: >`), which is
    all SKILL.md files use. Nested YAML is ignored rather than guessed at.
    """
    lines = text.lstrip("﻿").splitlines()
    if not lines or lines[0].strip() != "---":
        return None
    end = next((i for i in range(1, len(lines)) if lines[i].strip() == "---"), None)
    if end is None:
        return None
    meta: dict[str, str] = {}
    key = None
    for line in lines[1:end]:
        if line[:1] in (" ", "\t"):
            if key is not None and line.strip():
                meta[key] = f"{meta[key]} {line.strip()}".strip()
            continue
        key = None
        if ":" not in line or line.startswith("#"):
            continue
        k, v = line.split(":", 1)
        k, v = k.strip(), v.strip()
        if v in BLOCK_MARKERS:
            meta[k] = ""
            key = k
        else:
            meta[k] = v.strip('"').strip("'")
            key = k  # plain scalars may also continue on indented lines
    return meta


def discover_skills(dirs: list[Path]) -> dict[str, Skill]:
    """Later folders win, so a project skill overrides a same-named home one."""
    skills: dict[str, Skill] = {}
    for base in dirs:
        if not base.is_dir():
            continue
        for folder in sorted(p for p in base.iterdir() if p.is_dir()):
            path = next((folder / n for n in ("SKILL.md", "skill.md") if (folder / n).is_file()), None)
            if path is None:
                continue
            meta = parse_front_matter(path.read_text(encoding="utf-8", errors="replace"))
            if not meta or not meta.get("name"):
                continue
            skills[meta["name"]] = Skill(meta["name"], meta.get("description", ""), path)
    return skills


def skills_prompt(skills: dict[str, Skill]) -> str:
    if not skills:
        return "No skills are installed."
    return "\n".join(f"- {s.name}: {s.description}" for s in sorted(skills.values(), key=lambda s: s.name))
