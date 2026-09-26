import codecs
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


def test_offset_past_the_end_is_an_error(box, workspace):
    (workspace / "six.txt").write_text("1\n2\n3\n4\n5\n6")
    assert run(box, "read_file", path="six.txt", offset=100) == "error: offset 100 is past the end of six.txt (6 lines)"
    assert run(box, "read_file", path="six.txt", offset=6) == "6\n[lines 6-6 of 6]"


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
        "error: reading a file outside the workspace was not approved"
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
    out = run(box, "bash", command="echo started; sleep 5", timeout=1)
    assert time.time() - start < 4.5
    # What it printed before hanging is kept: it says where it hung.
    assert out == "started\n[timed out after 1s and was killed; output so far is above]"


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
def test_a_background_process_is_stopped_not_waited_on(cfg, allow, workspace):
    box = Toolbox(cfg, allow, skills={}, todos=TodoList(), sandbox=Sandbox(workspace, "none"))
    start = time.time()
    out = run(box, "bash", command="(sleep 30; echo late > late.txt) & echo started")
    elapsed = time.time() - start
    assert out.startswith("started\n[exit code 0]") and "background process it started was stopped" in out
    assert elapsed < 8  # was two 5-second waits, and the process kept running
    time.sleep(1)
    assert not (workspace / "late.txt").exists()


@needs_posix_shell
def test_bash_runs_in_the_workspace(box, workspace):
    out = run(box, "bash", command="pwd")
    assert out.splitlines()[0].rstrip("/").lower().endswith(workspace.name.lower())


@needs_posix_shell
def test_secrets_are_not_in_the_commands_environment(cfg, allow, workspace, monkeypatch):
    # `$` makes the command ask; even when the user says yes, the secrets are not there.
    box = Toolbox(cfg, allow, skills={}, todos=TodoList(), sandbox=Sandbox(workspace, "none"))
    monkeypatch.setenv("HARNESS_API_KEY", "sk-live-123")
    monkeypatch.setenv("GITHUB_TOKEN", "ghp_456")
    monkeypatch.setenv("DB_PASSWORD", "hunter2")
    monkeypatch.setenv("HARNESS_MODEL", "keep-me")
    out = run(box, "bash", command="echo k=$HARNESS_API_KEY t=$GITHUB_TOKEN p=$DB_PASSWORD m=$HARNESS_MODEL")
    assert out == "k= t= p= m=keep-me\n[exit code 0]"


@pytest.mark.parametrize("name, secret", [
    ("OPENROUTER_API_KEY", True), ("AWS_SECRET_ACCESS_KEY", True), ("GH_TOKEN", True), ("KEY", True),
    ("PGPASSWORD", True), ("GITHUB_PAT", True), ("MYSQL_PWD", True), ("PWD", False), ("PATH", False), ("KEYBOARD_LAYOUT", False), ("SSH_AUTH_SOCK", False),
    ("MONKEY_BUSINESS", False),
])
def test_secret_name_detection(name, secret):
    from harness.tools import scrubbed_env
    assert (name not in scrubbed_env({name: "v"})) is secret


def test_urls_carrying_a_password_are_scrubbed_whatever_their_name():
    from harness.tools import scrubbed_env
    env = scrubbed_env({"DATABASE_URL": "postgres://u:p@h/db", "REDIS": "redis://:pw@h:6379",
                        "MIRROR": "https://user:tok@example.com/", "HOMEPAGE": "https://example.com/"})
    assert env == {"HOMEPAGE": "https://example.com/"}


@needs_posix_shell
def test_python_children_write_utf8_that_arrives_intact(cfg, allow, workspace):
    import sys
    box = Toolbox(cfg, allow, skills={}, todos=TodoList(), sandbox=Sandbox(workspace, "none"))
    (workspace / "u.py").write_text("print(chr(233), chr(10003), chr(0x1F41B))\n", encoding="utf-8")
    py = Path(sys.executable).as_posix()
    out = run(box, "bash", command=f'"{py}" u.py')
    assert out == "é ✓ 🐛\n[exit code 0]"  # was U+FFFD, and ✓ crashed the child on Windows


@needs_posix_shell
def test_colour_codes_are_stripped(box):
    assert run(box, "bash", command=r"printf '\033[31mred\033[0m plain'") == "red plain\n[exit code 0]"


def test_binary_files_are_refused(box, workspace):
    (workspace / "img.png").write_bytes(b"\x89PNG\r\n\x1a\n\0\0\0\rIHDR" + bytes(100))
    assert run(box, "read_file", path="img.png").startswith("error: img.png is a binary file")


def test_huge_files_are_refused_before_reading(box, workspace, monkeypatch):
    monkeypatch.setattr("harness.tools.MAX_READ_BYTES", 1000)
    (workspace / "log.txt").write_text("x" * 5000)
    out = run(box, "read_file", path="log.txt")
    assert out.startswith("error: log.txt is 5,000 bytes") and "grep" in out


def test_git_internals_are_not_writable(box, workspace):
    (workspace / ".git" / "hooks").mkdir(parents=True)
    assert "inside .git/" in run(box, "write_file", path=".git/hooks/pre-commit", content="rm -rf ~")
    assert not (workspace / ".git" / "hooks" / "pre-commit").exists()
    assert run(box, "write_file", path=".gitignore", content="x").startswith("wrote")  # only the folder


@pytest.mark.parametrize("args, expected", [
    ({"path": "f.txt", "content": 5}, "content must be string"),
    ({"path": ["f.txt"], "content": "x"}, "path must be string"),
])
def test_argument_types_are_checked(box, args, expected):
    assert expected in box.call("write_file", json.dumps(args))


def test_numeric_strings_are_accepted_for_integers(box, workspace):
    (workspace / "f.txt").write_text("a\nb\nc")
    assert run(box, "read_file", path="f.txt", offset="2", limit="1") == "b\n[lines 2-2 of 3]"


# ---- encodings and line endings ---------------------------------------------

def test_latin1_file_survives_an_edit_byte_for_byte(box, workspace):
    f = workspace / "legacy.py"
    f.write_bytes("name = 'café'\nx = 1\n".encode("latin-1"))
    assert "café" in run(box, "read_file", path="legacy.py")
    assert run(box, "str_replace", path="legacy.py", old_string="x = 1", new_string="x = 2").startswith("replaced")
    assert f.read_bytes() == "name = 'café'\nx = 2\n".encode("latin-1")


def test_latin1_file_refuses_characters_it_cannot_hold(box, workspace):
    f = workspace / "legacy.txt"
    f.write_bytes("café\n".encode("latin-1"))
    run(box, "read_file", path="legacy.txt")
    out = run(box, "str_replace", path="legacy.txt", old_string="café", new_string="咖啡")
    assert "stored as latin-1" in out
    assert f.read_bytes() == "café\n".encode("latin-1")


@pytest.mark.parametrize("encoding, bom", [("utf-16", b""), ("utf-8-sig", b"")])
def test_bom_files_keep_their_encoding(box, workspace, encoding, bom):
    f = workspace / "ps.txt"
    f.write_bytes("line one\r\nline two\r\n".encode(encoding))
    assert run(box, "read_file", path="ps.txt") == "line one\nline two\n"  # UTF-16 has NULs but is text
    run(box, "str_replace", path="ps.txt", old_string="two", new_string="2")
    assert f.read_bytes() == "line one\r\nline 2\r\n".encode(encoding)


def test_mixed_line_endings_are_left_alone_except_where_edited(box, workspace):
    f = workspace / "mixed.txt"
    f.write_bytes(b"one\ntwo\r\nthree\n")
    run(box, "read_file", path="mixed.txt")
    run(box, "str_replace", path="mixed.txt", old_string="three", new_string="3")
    assert f.read_bytes() == b"one\ntwo\r\n3\n"  # was rewritten to all-CRLF once


def test_mixed_file_edit_spanning_a_crlf_line(box, workspace):
    f = workspace / "mixed.txt"
    f.write_bytes(b"a\r\nb\r\nc\n")
    run(box, "read_file", path="mixed.txt")
    assert run(box, "str_replace", path="mixed.txt", old_string="a\nb", new_string="A\nB").startswith("replaced")
    assert f.read_bytes() == b"A\r\nB\r\nc\n"


def test_write_file_does_not_double_carriage_returns(box, workspace):
    f = workspace / "win.txt"
    f.write_bytes(b"old\r\n")
    run(box, "read_file", path="win.txt")
    run(box, "write_file", path="win.txt", content="a\r\nb\r\n")
    assert f.read_bytes() == b"a\r\nb\r\n"


def test_utf16_big_endian_round_trips_byte_for_byte(box, workspace):
    f = workspace / "be.txt"
    original = codecs.BOM_UTF16_BE + "hello\nworld\n".encode("utf-16-be")
    f.write_bytes(original)
    run(box, "read_file", path="be.txt")
    run(box, "str_replace", path="be.txt", old_string="world", new_string="there")
    assert f.read_bytes() == codecs.BOM_UTF16_BE + "hello\nthere\n".encode("utf-16-be")  # was rewritten as LE


# ---- guarded files ------------------------------------------------------------

@pytest.mark.skipif(os.name != "nt", reason="Windows drops trailing dots and spaces from names")
@pytest.mark.parametrize("path", [".env.", ".agents./skills/x/SKILL.md", ".harness./x", "notes.txt ", ".git./hooks/x"])
def test_windows_name_aliases_cannot_slip_past_the_guards(box, workspace, deny, path):
    out = run(box, "write_file", path=path, content="planted")
    assert out.startswith("error:") and "dot or space" in out
    assert not any(workspace.iterdir())
    assert deny.asked == []


def test_git_folders_are_blocked_at_any_depth(box, workspace):
    assert "inside .git/" in run(box, "write_file", path="vendor/lib/.git/hooks/pre-commit", content="x")
    assert "inside .git/" in run(box, "write_file", path=".git", content="gitdir: /elsewhere")


def test_approval_prompts_show_control_characters(cfg, workspace):
    asked = []
    box = Toolbox(cfg, lambda q: asked.append(q) or False, skills={}, todos=TodoList(),
                  sandbox=Sandbox(workspace, "none"))
    run(box, "bash", command="rm -rf src;\rAllow? run: ls -la")
    rlo = chr(0x202E)  # right-to-left override: reverses how the rest of the line displays
    run(box, "bash", command=f"rm x {rlo} txt.exe")
    assert asked[0] == "run: rm -rf src;\\rAllow? run: ls -la"  # \r can no longer hide the rm
    assert rlo not in asked[1] and "\\u202e" in asked[1]


def test_spill_is_not_written_through_a_redirected_harness_folder(cfg, deny, workspace, tmp_path):
    outside = tmp_path / "elsewhere"
    outside.mkdir()
    (outside / "keep.txt").write_text("precious")
    try:
        if os.name == "nt":
            import _winapi
            _winapi.CreateJunction(str(outside), str(workspace / ".harness"))
        else:
            (workspace / ".harness").symlink_to(outside, target_is_directory=True)
    except OSError:
        pytest.skip("cannot create a junction or symlink here")
    cfg.output_cap = 200
    box = Toolbox(cfg, deny, skills={}, todos=TodoList(), sandbox=Sandbox(workspace, "none"))
    (workspace / "big.txt").write_text("x" * 1000)
    out = run(box, "read_file", path="big.txt")
    assert "the rest was not kept" in out
    box.cleanup_spill()
    assert sorted(p.name for p in outside.iterdir()) == ["keep.txt"]  # nothing written or deleted there

@pytest.mark.parametrize("path", [".env", "config/.env.local", ".agents/skills/x/SKILL.md", ".harness/notes.txt",
                                  "certs/server.pem"])
def test_files_that_configure_the_harness_or_hold_secrets_need_a_yes(box, workspace, deny, path):
    out = run(box, "write_file", path=path, content="HARNESS_BASE_URL=https://attacker.example")
    assert out == f"error: changing {path} was not approved"
    assert not (workspace / path).exists()
    assert deny.asked == [f"change {path} (it configures the harness or holds credentials)"]


def test_guarded_write_goes_ahead_when_approved(cfg, allow, workspace):
    box = Toolbox(cfg, allow, skills={}, todos=TodoList(), sandbox=Sandbox(workspace, "none"))
    assert run(box, "write_file", path=".env", content="HARNESS_MODEL=x").startswith("wrote")


def test_reading_a_credentials_file_asks(box, workspace, deny):
    (workspace / ".env").write_text("HARNESS_API_KEY=sk-live")
    assert "credentials file" in run(box, "read_file", path=".env")
    assert deny.asked == [f"read a credentials file: {workspace / '.env'}"]
    (workspace / ".env.example").write_text("HARNESS_API_KEY=")
    assert run(box, "read_file", path=".env.example") == "HARNESS_API_KEY="


# ---- read-only toolbox (the subagent's) ---------------------------------------

def test_read_only_copy_writes_no_spill_files(cfg, deny, workspace):
    cfg.output_cap = 200
    ro = Toolbox(cfg, deny, skills={}, todos=TodoList(), sandbox=Sandbox(workspace, "none")).read_only_copy()
    (workspace / "big.txt").write_text("x" * 1000)
    out = run(ro, "read_file", path="big.txt")
    assert "the rest was not kept" in out and "offset and limit" in out
    assert not (workspace / ".harness").exists()  # "cannot change files" holds exactly


def test_read_only_copy_refuses_writes_and_never_asks(box, workspace, deny):
    ro = box.read_only_copy()
    (workspace / "a.txt").write_text("x")
    assert run(ro, "read_file", path="a.txt") == "x"
    assert run(ro, "write_file", path="b.txt", content="y") == "error: this agent can only read"
    assert run(ro, "str_replace", path="a.txt", old_string="x", new_string="z") == "error: this agent can only read"
    assert ro.seen and not box.seen  # its reads do not count as the main agent's reads
    assert deny.asked == []


@needs_posix_shell
def test_read_only_copy_refuses_commands_that_would_ask(box, workspace, deny):
    ro = box.read_only_copy()
    (workspace / "keep.txt").write_text("x")
    assert run(ro, "bash", command="rm keep.txt") == \
        "error: subagents may only run read-only commands inside the workspace"
    assert (workspace / "keep.txt").exists() and deny.asked == []
    assert run(ro, "bash", command="ls").endswith("[exit code 0]")


def test_spill_files_are_never_tracked_as_seen(cfg, deny, workspace):
    cfg.output_cap = 50
    box = Toolbox(cfg, deny, skills={}, todos=TodoList(), sandbox=Sandbox(workspace, "none"))
    (workspace / "long.txt").write_text("L" * 500)
    run(box, "read_file", path="long.txt")
    run(box, "read_file", path=".harness/spill/output-1.txt")
    assert list(box.seen) == [workspace / "long.txt"]


# ---- output capture -----------------------------------------------------------

@needs_posix_shell
def test_endless_output_keeps_only_head_and_tail(cfg, allow, workspace, monkeypatch):
    monkeypatch.setattr("harness.tools.MAX_CAPTURE_CHARS", 1000)
    box = Toolbox(cfg, allow, skills={}, todos=TodoList(), sandbox=Sandbox(workspace, "none"))
    out = box.bash("seq 1 100000")  # ~590 KB of output
    assert out.startswith("1\n2\n3\n")
    assert "characters of output dropped" in out
    assert out.rstrip().endswith("100000\n[exit code 0]".rstrip())
    assert len(out) < 2500


def test_capture_keeps_everything_under_the_limit():
    import io
    from harness.tools import _Capture
    cap = _Capture(io.StringIO("abc" * 10), limit=100)
    cap.thread.join()
    assert cap.text() == "abc" * 10
    # One read returns all 4,000 characters: the tail must still be cut to the limit.
    cap = _Capture(io.StringIO("".join(f"{i:04d}" for i in range(1000))), limit=40)
    cap.thread.join()
    text = cap.text()
    assert text.startswith("0000000100020003") and text.endswith("0999") and "3,920 characters" in text
    assert len(text) < 140


# ---- skills, todos, task ---------------------------------------------------

def test_skill_folders_are_readable_without_asking(cfg, deny, tmp_path):
    folder = tmp_path / "skills" / "pdf"
    (folder / "reference").mkdir(parents=True)
    (folder / "SKILL.md").write_text("---\nname: pdf\ndescription: d\n---\nSee reference/forms.md")
    (folder / "reference" / "forms.md").write_text("fill forms like this")
    box = Toolbox(cfg, deny, skills={"pdf": Skill("pdf", "d", folder / "SKILL.md")}, todos=TodoList(),
                  sandbox=Sandbox(cfg.workspace, "none"))
    body = run(box, "read_skill", name="pdf")
    assert f"[skill folder: {folder.resolve().as_posix()}." in body
    assert run(box, "read_file", path=str(folder / "reference" / "forms.md")) == "fill forms like this"
    assert deny.asked == []


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


def test_a_failing_subagent_is_a_tool_error_not_the_end_of_the_turn(box):
    from harness.llm import LLMError

    def broken(prompt):
        raise LLMError("HTTP 503 from server")

    box.subagent = broken
    assert run(box, "task", prompt="x") == "error: the subagent failed: HTTP 503 from server"
