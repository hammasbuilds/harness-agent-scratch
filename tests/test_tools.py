import json
import os
import time
from pathlib import Path

import pytest

from harness.sandbox import Sandbox, find_shell
from harness.skills import Skill
from harness.todos import TodoList
from harness.tools import Toolbox

HAS_SHELL = find_shell()[0][0] != "cmd.exe"
needs_posix_shell = pytest.mark.skipif(not HAS_SHELL, reason="no POSIX shell on this machine")


@pytest.fixture
def box(cfg, deny):
    return Toolbox(cfg, deny, skills={}, todos=TodoList(), sandbox=Sandbox(cfg.workspace, "none"))


def run(box, tool, /, **args):
    return box.call(tool, json.dumps(args))


def bump_mtime(p: Path):
    later = p.stat().st_mtime + 5
    os.utime(p, (later, later))


# ---- dispatch ------------------------------------------------------------

def test_schemas_describe_every_tool(box):
    names = [s["function"]["name"] for s in box.schemas()]
    assert names == ["bash", "read_file", "write_file", "str_replace", "read_skill", "write_todos", "task"]
    assert [s["function"]["name"] for s in box.schemas(("bash", "read_file"))] == ["bash", "read_file"]


def test_unknown_tool_and_bad_arguments_become_messages(box):
    assert run(box, "delete_everything").startswith("error: unknown tool 'delete_everything'")
    assert box.call("read_file", "{not json").startswith("error: arguments for read_file are not valid JSON")
    assert box.call("read_file", "[1, 2]") == "error: arguments for read_file must be a JSON object"
    assert run(box, "read_file") == "error: bad arguments for read_file: missing path"
    assert run(box, "read_file", path="a", colour="red") == "error: bad arguments for read_file: unknown colour"


def test_wrong_value_types_do_not_escape_the_toolbox(box, workspace):
    (workspace / "f.txt").write_text("x")
    assert run(box, "read_file", path="f.txt", offset="abc").startswith("error: bad arguments for read_file")
    assert run(box, "write_file", path="f.txt", content=None).startswith("error:")


# ---- files ---------------------------------------------------------------

def test_write_then_read_round_trip(box, workspace):
    assert run(box, "write_file", path="notes/hello.txt", content="hello\nworld\n") == \
        "wrote 12 characters to notes/hello.txt"
    assert (workspace / "notes" / "hello.txt").read_bytes() == b"hello\nworld\n"
    assert run(box, "read_file", path="notes/hello.txt") == "hello\nworld\n"


def test_read_with_offset_and_limit(box, workspace):
    (workspace / "f.txt").write_text("\n".join(f"line {i}" for i in range(1, 11)))
    assert run(box, "read_file", path="f.txt", offset=3, limit=2) == "line 3\nline 4\n[lines 3-4 of 10]"


def test_read_missing_file(box):
    assert run(box, "read_file", path="nope.txt") == "error: nope.txt does not exist or is not a file"


def test_overwriting_requires_a_read_first(box, workspace):
    (workspace / "a.txt").write_text("original")
    assert run(box, "write_file", path="a.txt", content="new") == "error: read a.txt before changing it"
    assert (workspace / "a.txt").read_text() == "original"
    run(box, "read_file", path="a.txt")
    assert run(box, "write_file", path="a.txt", content="new").startswith("wrote")


def test_writes_outside_the_workspace_are_refused(box, workspace, tmp_path):
    outside = tmp_path / "outside.txt"
    assert "outside the workspace" in run(box, "write_file", path=str(outside), content="x")
    assert "outside the workspace" in run(box, "write_file", path="../escape.txt", content="x")
    assert not outside.exists() and not (tmp_path / "escape.txt").exists()


def test_reading_outside_the_workspace_asks(box, tmp_path, deny):
    (tmp_path / "secret.txt").write_text("s")
    assert run(box, "read_file", path=str(tmp_path / "secret.txt")) == \
        "error: the user declined reading a file outside the workspace"
    assert deny.asked and "secret.txt" in deny.asked[0]


def test_reading_outside_with_approval(cfg, allow, tmp_path):
    box = Toolbox(cfg, allow, skills={}, todos=TodoList(), sandbox=Sandbox(cfg.workspace, "none"))
    (tmp_path / "shared.txt").write_text("ok")
    assert run(box, "read_file", path=str(tmp_path / "shared.txt")) == "ok"


# ---- str_replace ---------------------------------------------------------

@pytest.fixture
def hello(box, workspace):
    f = workspace / "hello.txt"
    f.write_text("hello world\n" * 5)
    run(box, "read_file", path="hello.txt")
    return f


def test_replace_unique(box, workspace):
    (workspace / "app.py").write_text("x = 1\ny = 2\n")
    run(box, "read_file", path="app.py")
    assert run(box, "str_replace", path="app.py", old_string="y = 2", new_string="y = 3") == \
        "replaced 1 occurrence(s) in app.py"
    assert (workspace / "app.py").read_text() == "x = 1\ny = 3\n"


def test_replace_ambiguous_is_refused_not_guessed(box, hello):
    out = run(box, "str_replace", path="hello.txt", old_string="hello world", new_string="goodbye")
    assert out.startswith("error: old_string occurs 5 times")
    assert hello.read_text() == "hello world\n" * 5


def test_replace_all(box, hello):
    assert run(box, "str_replace", path="hello.txt", old_string="hello", new_string="bye", replace_all=True) == \
        "replaced 5 occurrence(s) in hello.txt"
    assert hello.read_text() == "bye world\n" * 5


def test_replace_not_found_empty_and_identical(box, hello):
    assert "not found" in run(box, "str_replace", path="hello.txt", old_string="nope", new_string="x")
    assert "empty" in run(box, "str_replace", path="hello.txt", old_string="", new_string="x")
    assert "identical" in run(box, "str_replace", path="hello.txt", old_string="hello", new_string="hello")


def test_replace_refuses_a_file_changed_since_it_was_read(box, hello):
    hello.write_text("someone else edited this\n")
    bump_mtime(hello)
    out = run(box, "str_replace", path="hello.txt", old_string="someone", new_string="x")
    assert out == "error: hello.txt changed on disk since you read it; read it again first"
    run(box, "read_file", path="hello.txt")
    assert run(box, "str_replace", path="hello.txt", old_string="someone", new_string="x").startswith("replaced")


def test_consecutive_edits_do_not_need_a_reread(box, hello):
    run(box, "str_replace", path="hello.txt", old_string="hello", new_string="a", replace_all=True)
    assert run(box, "str_replace", path="hello.txt", old_string="a", new_string="b", replace_all=True).startswith("replaced")


def test_crlf_files_stay_crlf(box, workspace):
    f = workspace / "win.txt"
    f.write_bytes(b"one\r\ntwo\r\nthree\r\n")
    assert run(box, "read_file", path="win.txt") == "one\ntwo\nthree\n"
    # the model copies what it saw, with \n; the match still works
    assert run(box, "str_replace", path="win.txt", old_string="one\ntwo", new_string="1\n2").startswith("replaced")
    assert f.read_bytes() == b"1\r\n2\r\nthree\r\n"


def test_replace_in_missing_file(box):
    assert "use write_file" in run(box, "str_replace", path="new.py", old_string="a", new_string="b")


# ---- output cap ----------------------------------------------------------

def test_long_output_is_capped_and_spilled_then_cleaned(cfg, deny, workspace):
    cfg.output_cap = 100
    box = Toolbox(cfg, deny, skills={}, todos=TodoList(), sandbox=Sandbox(workspace, "none"))
    big = "".join(f"{i:05d}\n" for i in range(1000))
    (workspace / "big.txt").write_text(big)
    out = run(box, "read_file", path="big.txt")
    assert out.startswith(big[:100])
    assert f"showing 100 of {len(big)} characters" in out
    spill = workspace / ".harness" / "spill" / "output-1.txt"
    assert spill.read_text() == big
    assert ".harness/spill/output-1.txt" in out
    box.cleanup_spill()
    assert not spill.exists()


def test_short_output_is_untouched(box):
    assert box.cap("short") == "short"


# ---- bash ----------------------------------------------------------------

@needs_posix_shell
def test_read_only_command_runs_without_asking(box, workspace, deny):
    (workspace / "a.txt").write_text("x")
    out = run(box, "bash", command="ls")
    assert "a.txt" in out and out.endswith("[exit code 0]")
    assert deny.asked == []


@needs_posix_shell
def test_stderr_and_exit_code_are_reported(cfg, allow, workspace):
    box = Toolbox(cfg, allow, skills={}, todos=TodoList(), sandbox=Sandbox(workspace, "none"))
    out = run(box, "bash", command="echo out; echo err 1>&2; exit 3")
    assert out == "out\n[stderr]\nerr\n[exit code 3]"


@needs_posix_shell
def test_declined_command_does_not_run(box, workspace, deny):
    (workspace / "keep.txt").write_text("x")
    assert run(box, "bash", command="rm keep.txt") == "error: the user declined to run this command"
    assert (workspace / "keep.txt").exists()
    assert deny.asked == ["run: rm keep.txt"]


@needs_posix_shell
def test_approved_command_runs(cfg, allow, workspace):
    box = Toolbox(cfg, allow, skills={}, todos=TodoList(), sandbox=Sandbox(workspace, "none"))
    (workspace / "gone.txt").write_text("x")
    assert run(box, "bash", command="rm gone.txt").endswith("[exit code 0]")
    assert not (workspace / "gone.txt").exists()


def test_blocked_command_never_asks(box, deny):
    assert run(box, "bash", command="rm -rf /") == "error: this command is blocked by the harness and will not run"
    assert deny.asked == []


@needs_posix_shell
def test_timeout(cfg, allow, workspace):
    box = Toolbox(cfg, allow, skills={}, todos=TodoList(), sandbox=Sandbox(workspace, "none"))
    start = time.time()
    # The shell's child keeps the pipes open; without killing the whole tree this
    # returned only after the full 5 seconds (and never, for a server).
    assert run(box, "bash", command="sleep 5", timeout=1) == "error: command timed out after 1s"
    assert time.time() - start < 4.5


@needs_posix_shell
def test_quotes_and_pipes_survive_the_trip_to_the_shell(cfg, allow, workspace):
    # On Windows the command string goes through CreateProcess quoting on its way
    # into Git Bash; double spaces and quotes must come out unchanged.
    box = Toolbox(cfg, allow, skills={}, todos=TodoList(), sandbox=Sandbox(workspace, "none"))
    assert run(box, "bash", command='echo "a  b" | cat') == "a  b\n[exit code 0]"
    assert allow.asked == []
    # `$((` looks like command substitution to the permission check, so it asks.
    assert run(box, "bash", command="echo 'it''s' \"$((2+3))\"") == "its 5\n[exit code 0]"
    assert len(allow.asked) == 1


@needs_posix_shell
def test_bash_runs_in_the_workspace(box, workspace):
    out = run(box, "bash", command="pwd")
    assert out.splitlines()[0].rstrip("/").lower().endswith(workspace.name.lower())


# ---- skills, todos, task ---------------------------------------------------

def test_read_skill(cfg, deny, tmp_path):
    path = tmp_path / "SKILL.md"
    path.write_text("---\nname: pdf\ndescription: d\n---\nUse pypdf.")
    box = Toolbox(cfg, deny, skills={"pdf": Skill("pdf", "d", path)}, todos=TodoList(),
                  sandbox=Sandbox(cfg.workspace, "none"))
    assert run(box, "read_skill", name="pdf").endswith("Use pypdf.")
    assert run(box, "read_skill", name="docx") == "error: no skill named 'docx' (known: pdf)"


def test_write_todos_and_validation(box):
    out = run(box, "write_todos", todos=[{"content": "a", "status": "in_progress"}])
    assert out == "[>] a"
    assert "status 'started'" in run(box, "write_todos", todos=[{"content": "a", "status": "started"}])


def test_task_without_a_subagent(box):
    assert run(box, "task", prompt="x") == "error: subagents are not available here"
