from pathlib import Path

from harness.skills import default_skill_dirs, discover_skills, parse_front_matter, skills_prompt


def make_skill(base: Path, folder: str, text: str, filename: str = "SKILL.md") -> Path:
    d = base / folder
    d.mkdir(parents=True)
    (d / filename).write_text(text, encoding="utf-8")
    return d / filename


def test_front_matter_flat_and_quoted():
    meta = parse_front_matter('---\nname: copywriting\ndescription: "Write sharp copy."\nlicense: MIT\n---\nbody')
    assert meta == {"name": "copywriting", "description": "Write sharp copy.", "license": "MIT"}


def test_front_matter_folded_block():
    text = "---\nname: frontend\ndescription: >\n  Build distinctive\n  production-grade interfaces.\n---\n"
    assert parse_front_matter(text)["description"] == "Build distinctive production-grade interfaces."


def test_front_matter_ignores_nested_yaml():
    text = "---\nname: x\nmetadata:\n  author: y\ndescription: d\n---\n"
    meta = parse_front_matter(text)
    assert meta["name"] == "x" and meta["description"] == "d"


def test_no_or_unclosed_front_matter():
    assert parse_front_matter("# just markdown") is None
    assert parse_front_matter("---\nname: x\n") is None


def test_front_matter_with_bom():
    assert parse_front_matter("﻿---\nname: x\n---\n")["name"] == "x"


def test_discovery_and_project_overrides_home(tmp_path):
    home, project = tmp_path / "home", tmp_path / "project"
    make_skill(home, "a", "---\nname: alpha\ndescription: from home\n---\n")
    make_skill(home, "b", "---\nname: beta\ndescription: only home\n---\n")
    make_skill(project, "a", "---\nname: alpha\ndescription: from project\n---\n")
    make_skill(project, "nameless", "---\ndescription: no name\n---\n")
    make_skill(project, "notes", "no front matter")
    (project / "stray.md").write_text("not in a folder")
    skills = discover_skills([home, project, tmp_path / "missing"])
    assert sorted(skills) == ["alpha", "beta"]
    assert skills["alpha"].description == "from project"


def test_lowercase_filename_is_found(tmp_path):
    make_skill(tmp_path, "s", "---\nname: lower\ndescription: d\n---\n", filename="skill.md")
    assert "lower" in discover_skills([tmp_path])


def test_prompt_lists_name_and_description_only(tmp_path):
    make_skill(tmp_path, "s", "---\nname: pdf\ndescription: Work with PDFs.\n---\nSECRET BODY")
    prompt = skills_prompt(discover_skills([tmp_path]))
    assert prompt == "- pdf: Work with PDFs."
    assert skills_prompt({}) == "No skills are installed."


def test_a_wordy_skill_cannot_swell_the_system_prompt(tmp_path):
    make_skill(tmp_path, "w", "---\nname: wordy\ndescription: " + "blah " * 5000 + "\n---\n")
    for i in range(60):
        make_skill(tmp_path, f"s{i:02d}", f"---\nname: skill{i:02d}\ndescription: {'does things ' * 20}\n---\n")
    prompt = skills_prompt(discover_skills([tmp_path]))
    assert len(prompt) <= 4200 and "more skills not listed" in prompt
    assert "- skill00: does things" in prompt and "..." in prompt


def test_default_dirs(tmp_path):
    assert default_skill_dirs(tmp_path / "ws", home=tmp_path / "h") == [
        tmp_path / "h" / ".agents" / "skills", tmp_path / "ws" / ".agents" / "skills"]
