import os
from pathlib import Path

import pytest

from harness.permissions import ALLOW, ASK, DENY, classify, is_secret_file


@pytest.mark.parametrize("command", [
    "ls -la",
    "pwd",
    "cat README.md | head -20",
    "grep -rn TODO . && wc -l harness/*.py",
    "git status",
    "git --no-pager log --oneline -5",
    "find . -name '*.py'",
    "ls missing 2>/dev/null",
    "grep foo bar.txt 2>&1",
    "wc -l < file.txt",
])
def test_read_only_commands_run_at_once(command, tmp_path):
    # A workspace is needed: without one, recursive reads cannot be checked and ask.
    (tmp_path / "harness").mkdir()
    assert classify(command, tmp_path.resolve()) == ALLOW


@pytest.mark.parametrize("command", [
    "/usr/bin/ls",  # a path, not a program name: once pinned as ALLOW by this very file
    "./ls", "sub/cat a.txt", "../ls", "C:/Users/Public/cat.exe a.txt",
    "tree", "sort a.txt", "file a.txt", "date",  # have writing/executing options; off the list
])
def test_command_names_must_be_bare_listed_programs(command, tmp_path):
    assert classify(command, tmp_path) == ASK


@pytest.mark.parametrize("command", [
    "rm notes.txt",
    "python script.py",
    "echo hi > out.txt",
    "echo hi >> out.txt",
    "cat a | tee b",
    "ls && rm x",
    "ls; curl http://example.com",
    "find . -name '*.pyc' -delete",
    "find . -exec rm {} ;",
    "sort -o out.txt in.txt",
    "git push",
    "git reset --hard",
    "git branch -D main",
    "echo $(rm x)",
    "echo `whoami`",
    "echo 'unbalanced",
    "sed -i s/a/b/ f.txt",
    "",
    "ls\nrm x",
])
def test_anything_else_asks(command):
    assert classify(command) == ASK


@pytest.mark.parametrize("command", [
    "rm -rf /",
    "rm -rf ~",
    "sudo rm -fr / ",
    "rm -r -f $HOME",
    "rm -rf C:\\",
    "mkfs.ext4 /dev/sda1",
    "dd if=/dev/zero of=/dev/sda",
    ":(){ :|:& };:",
    "shutdown -h now",
    "format C:",
])
def test_catastrophic_commands_never_run(command):
    assert classify(command) == DENY


@pytest.mark.parametrize("command", [
    "git log --output=pwned.txt",
    "git diff --output x",
    "git -c core.pager=sh log",
    "git --exec-path=/tmp/evil status",
    "git diff --ext-diff",
    "rg --pre 'rm x' foo",
    "rg --pre=./script foo",
    "sort --compress-program=sh big.txt",
    "sort -uo out.txt in.txt",
    "tree -o listing.txt",
    "date -s 2020-01-01",
    "find . -fprint0 out",
])
def test_read_only_commands_with_writing_or_executing_flags_ask(command):
    assert classify(command) == ASK


def test_paths_outside_the_workspace_ask(tmp_path):
    ws = (tmp_path / "ws").resolve()
    (ws / "sub").mkdir(parents=True)
    for command in ["cat ~/.ssh/id_rsa", "cat ../secret.txt", "cat sub/../../secret.txt",
                    f"cat {(tmp_path / 'secret.txt').as_posix()}", "head -5 /etc/passwd", "grep -r password /",
                    "ls ..", "wc -l < ../secret.txt", "rg --file=../patterns x"]:
        assert classify(command, ws) == ASK, command


def test_paths_inside_the_workspace_still_run(tmp_path):
    ws = (tmp_path / "ws").resolve()
    (ws / "sub").mkdir(parents=True)
    for command in ["cat sub/a.txt", "ls sub/..", f"cat {(ws / 'a.txt').as_posix()}", "grep -rn TODO .",
                    "ls missing 2>/dev/null", "cat -", "git log -5"]:
        assert classify(command, ws) == ALLOW, command


@pytest.mark.parametrize("command", [
    "ls # list\nrm -f important.txt",  # shlex used to read everything after # as a comment
    "cat a#b; touch pwned",
    "ls #; rm x",
])
def test_hash_does_not_hide_the_rest_of_the_command(command):
    assert classify(command) == ASK


@pytest.mark.parametrize("command", [
    "cat $HOME/.gitconfig",
    "cat ${USERPROFILE}/.aws/credentials",
    "grep x < $HOME/.netrc",
    "echo $GITHUB_TOKEN",
])
def test_dollar_expansions_ask(command, tmp_path):
    assert classify(command, tmp_path) == ASK


@pytest.mark.parametrize("command", [
    "uniq a.txt b.txt",  # overwrites b.txt
    "uniq -c in.txt out.txt",
    "file -C -m magic",
])
def test_other_writing_forms_ask(command):
    assert classify(command) == ASK


def test_uniq_with_one_file_runs():
    assert classify("uniq -c words.txt") == ALLOW
    assert classify("cat words.txt | uniq -c") == ALLOW


@pytest.mark.parametrize("command", [
    "cat .env", "cat config/.env.local", "head ~/.ssh/id_ed25519", "cat server.pem", "cat .netrc",
    "grep key credentials.json",
])
def test_credentials_files_ask_even_inside_the_workspace(command, tmp_path):
    assert classify(command, tmp_path) == ASK


@pytest.mark.parametrize("name, secret", [
    (".env", True), (".env.production", True), ("id_rsa", True), ("id_ed25519.pub", True), ("tls.key", True),
    (".env.example", False), (".env.sample", False), ("environment.py", False), ("keys.py", False),
    ("monkey.txt", False),
])
def test_secret_file_names(name, secret):
    assert is_secret_file(name) is secret


@pytest.mark.parametrize("command", [
    "echo ' & mkdir pwned & echo '",  # single quotes are plain characters in cmd.exe
    "type a.txt ^& del b.txt",
    "dir %USERPROFILE%",
    'echo "a" & del x',
])
def test_cmd_shell_is_classified_conservatively(command):
    assert classify(command, cmd_shell=True) == ASK


@pytest.mark.parametrize("command", [r"dir ..\..", r"dir \\attacker.example\share", "dir ../..", "dir sub\\..\\.."])
def test_cmd_backslashes_and_parent_paths_ask(command, tmp_path):
    # POSIX shlex drops backslashes, so `..\..` once reached the path check as `....`
    assert classify(command, tmp_path, cmd_shell=True) == ASK


@pytest.mark.parametrize("command", ["wc -c --files0-from=list.txt", "du --files0-from list.txt", "wc --files0=x"])
def test_reading_file_names_from_a_file_asks(box_ws, command):
    assert classify(command, box_ws) == ASK


@pytest.mark.parametrize("config", [
    "[core]\n\tfsmonitor = ./evil.sh\n", "[diff]\n\texternal = ./evil.sh\n",
    '[diff "bin"]\n\ttextconv = ./evil.sh\n', "[core]\n\tpager = ./evil.sh\n", "[include]\n\tpath = ../x\n",
])
def test_git_asks_when_the_repository_config_can_run_programs(tmp_path, config):
    ws = tmp_path.resolve()
    (ws / ".git").mkdir()
    (ws / ".git" / "config").write_text(config)
    assert classify("git status", ws) == ASK
    (ws / ".git" / "config").write_text("[core]\n\tbare = false\n\tautocrlf = input\n")
    assert classify("git status", ws) == ALLOW


@pytest.mark.parametrize("command", [
    "rg --color=always --hyperlink-format=file --hostname-bin=./evil.exe x src",  # ripgrep 14 runs it
    "rg --hostname-bin sh x", "rg --hyperlink-format=default x",
])
def test_ripgrep_options_that_run_programs_ask(box_ws, command):
    assert classify(command, box_ws) == ASK


def test_ripgrep_is_off_the_list_entirely(box_ws):
    # It has options that run programs (--pre, and --hostname-bin since 14), so by
    # the module's rule it asks, however plain the call; grep covers the same reads.
    assert classify("rg TODO src", box_ws) == ASK
    assert classify("grep -rn TODO src", box_ws) == ALLOW


@pytest.mark.parametrize("folder", ["build", "dist", "node_modules/pkg", ".venv", "__pycache__"])
def test_secrets_in_build_and_dependency_folders_still_count(box_ws, folder):
    (box_ws / folder).mkdir(parents=True)
    (box_ws / folder / ".env").write_text("HARNESS_API_KEY=sk")
    assert classify("grep -r API_KEY .", box_ws) == ASK  # grep does not skip these folders


@pytest.mark.parametrize("config", [
    '[diff "x"]\n\tcommand = ./evil.sh\n',  # with a .gitattributes the model can write
    "[gpg]\n\tprogram = ./evil.sh\n[log]\n\tshowSignature = true\n",
    "[extensions]\n\tworktreeConfig = true\n",
    "[core]\n\tbare = false\n\tsshCommand = ./evil.sh\n",
    "[alias]\n\tst = !./evil.sh\n",
    "not a config line\n",
])
def test_git_config_is_an_allowlist_of_plain_clone_keys(tmp_path, config):
    ws = tmp_path.resolve()
    (ws / ".git").mkdir()
    (ws / ".git" / "config").write_text(config)
    assert classify("git log", ws) == ASK


def test_a_worktree_config_file_makes_git_ask(tmp_path):
    ws = tmp_path.resolve()
    (ws / ".git").mkdir()
    (ws / ".git" / "config").write_text("[core]\n\tbare = false\n")
    (ws / ".git" / "config.worktree").write_text("[core]\n\tfsmonitor = ./evil.sh\n")
    assert classify("git status", ws) == ASK


def test_a_real_clone_config_passes(tmp_path):
    ws = tmp_path.resolve()
    (ws / ".git").mkdir()
    (ws / ".git" / "config").write_text(
        "[core]\n\trepositoryformatversion = 0\n\tfilemode = false\n\tbare = false\n\tlogallrefupdates = true\n"
        "\tsymlinks = false\n\tignorecase = true\n"
        '[remote "origin"]\n\turl = https://github.com/owner/repo.git\n\tfetch = +refs/heads/*:refs/remotes/origin/*\n'
        '[branch "main"]\n\tremote = origin\n\tmerge = refs/heads/main\n')
    assert classify("git log --oneline -5", ws) == ALLOW


def test_plain_commands_still_run_under_cmd():
    assert classify("dir", cmd_shell=True) == ALLOW
    assert classify("type notes.txt", cmd_shell=True) == ASK  # `type` is not on the list


@pytest.fixture
def box_ws(tmp_path):
    ws = (tmp_path / "ws").resolve()
    (ws / "src").mkdir(parents=True)
    (ws / "a.txt").write_text("a")
    (ws / "src" / "m.py").write_text("x")
    (tmp_path / "secret.txt").write_text("s")
    return ws


@pytest.mark.parametrize("command", [
    "cat a.txt|&rm a.txt",  # |& pipes stderr too; it was once read as an argument and rm ran
    "(rm a.txt)",
    "ls; (rm a.txt)",
    "cat {../secret.txt,a.txt}",  # brace expansion can produce any path
    "cat {/etc/passwd,x}",
    "grep -f../secret.txt x",  # path glued to a flag
    "grep -f/etc/passwd x",
    "cat .*",  # can match .. in some shells
    "ls .?*",
    "cat ../*.txt",
    "cat /et*/passwd",  # decided by the fixed prefix, whatever exists on this machine
])
def test_second_review_bypasses_now_ask(box_ws, command):
    assert classify(command, box_ws) == ASK


@pytest.mark.parametrize("command", [
    "cat *", "cat src/*.py", "grep -rn TODO src", "ls -la", "head -n 5 a.txt", "grep -c x a.txt",
    "wc -l src/m.py", "git log --oneline -5", "ls src/..", "find . -name '*.py'", "grep -E 'a|b' a.txt",
])
def test_everyday_commands_still_run_without_asking(box_ws, command):
    assert classify(command, box_ws) == ALLOW


def test_a_symlink_out_of_the_workspace_is_followed(box_ws):
    try:
        (box_ws / "link").symlink_to(box_ws.parent, target_is_directory=True)
    except OSError:
        pytest.skip("this account cannot create symlinks (Windows without developer mode)")
    assert classify("cat link/secret.txt", box_ws) == ASK
    assert classify("cat li*/secret.txt", box_ws) == ASK
    assert classify("cat a.txt", box_ws) == ALLOW


@pytest.mark.parametrize("command", [
    # global options hide the real subcommand
    "git --namespace log commit --allow-empty -m x",
    "git --namespace status config core.fsmonitor evil",
    "git --namespace diff push origin main",
    "git -C log commit -am x",
    "git --work-tree log checkout -- .",
    "git -C sub log",
    # pathspec magic and object paths reach past the workspace
    "git show HEAD:.env", "git show :id_rsa", "git show HEAD:../outside.txt", "git ls-files :/",
    "git diff HEAD -- :/",
    # abbreviated long options
    "git log --outp=x", "git diff --ext",
])
def test_git_is_allowed_only_in_its_plain_read_only_form(box_ws, command):
    assert classify(command, box_ws) == ASK


def test_git_asks_when_the_repository_is_bigger_than_the_workspace(tmp_path):
    (tmp_path / ".git").mkdir()
    ws = (tmp_path / "sub").resolve()
    ws.mkdir()
    assert classify("git log", ws) == ASK  # its history and diffs cover files outside ws
    (ws / ".git").mkdir()
    assert classify("git log", ws) == ALLOW


@pytest.mark.parametrize("command", [
    "date -f.env", "grep -f.env a.txt", "grep -fid_rsa a.txt",  # glued to a short option
    "cat prod.env", "cat .git-credentials", "cat key.ppk", "cat .envrc",  # names the list lacked
])
def test_glued_and_newly_listed_credentials_ask(box_ws, command):
    assert classify(command, box_ws) == ASK


def test_a_bracket_glob_cannot_reach_a_hidden_secret(box_ws):
    (box_ws / ".env").write_text("HARNESS_API_KEY=sk")
    assert classify("cat [.]env", box_ws) == ASK  # Python's glob skips hidden files unless told not to
    assert classify("cat .en?", box_ws) == ASK


@pytest.mark.parametrize("command", ["cat a.txt", "cat a.txt | head -3"])
def test_after_all_that_plain_reads_still_run(box_ws, command):
    assert classify(command, box_ws) == ALLOW


def test_recursive_readers_ask_only_when_the_workspace_holds_secrets(box_ws):
    for command in ["grep -rn API_KEY .", "grep -R x src", "diff -r . src", "grep --recursive x ."]:
        assert classify(command, box_ws) == ALLOW, command
    (box_ws / "src" / ".env").write_text("HARNESS_API_KEY=sk")
    for command in ["grep -rn API_KEY .", "grep -R x src", "rg KEY", "diff -r . src", "grep --recursive x .",
                    "grep -nr x ."]:
        assert classify(command, box_ws) == ASK, command
    assert classify("grep -n x a.txt", box_ws) == ALLOW  # not recursive


@pytest.mark.parametrize("command", [
    "grep -d recurse KEY .", "grep --directories recurse KEY .", "grep --dir=recurse KEY .",
    "grep --recur KEY .", "grep --rec KEY .", "grep --dereference-rec KEY .", "egrep -R KEY .", "fgrep -rl KEY .",
])
def test_every_spelling_of_grep_recursion_is_caught(box_ws, command):
    (box_ws / ".env").write_text("HARNESS_API_KEY=sk")
    assert classify(command, box_ws) == ASK


def _junction_out_of(ws: Path, target: Path) -> None:
    if os.name == "nt":
        import _winapi
        _winapi.CreateJunction(str(target), str(ws / "linked"))
    else:
        (ws / "linked").symlink_to(target, target_is_directory=True)


@pytest.mark.parametrize("command", ["ls -L linked", "find -L . -name x", "du -L", "grep -r x .", "rg x",
                                     "ls -laRL", "du --deref"])
def test_link_following_walkers_ask_when_a_link_leads_out(box_ws, tmp_path, command):
    outside = tmp_path / "elsewhere"
    outside.mkdir()
    try:
        _junction_out_of(box_ws, outside)
    except OSError:
        pytest.skip("cannot create a junction or symlink here")
    assert classify(command, box_ws) == ASK


def test_walkers_that_do_not_follow_links_still_run_beside_one(box_ws, tmp_path):
    outside = tmp_path / "elsewhere"
    outside.mkdir()
    try:
        _junction_out_of(box_ws, outside)
    except OSError:
        pytest.skip("cannot create a junction or symlink here")
    assert classify("ls -la", box_ws) == ALLOW
    assert classify("find . -name '*.txt'", box_ws) == ALLOW


@pytest.mark.parametrize("command", ["echo hi > -", "ls &> -", "echo hi > NUL", "cat a.txt <> b.txt"])
def test_only_dev_null_is_a_harmless_redirect(box_ws, command):
    assert classify(command, box_ws) == ASK


@pytest.mark.parametrize("command", ["grep -rn shutdown src", "git log --grep=reboot", "cat docs/halt.md",
                                     "grep -n mkfs a.txt"])
def test_dangerous_words_as_arguments_are_not_denied(box_ws, command):
    assert classify(command, box_ws) != DENY  # only as a command name, where they would run


@pytest.mark.skipif(os.name != "nt", reason="8.3 short names are a Windows feature")
def test_windows_short_names_are_resolved_before_the_check(box_ws):
    import ctypes
    (box_ws / ".env").write_text("x")
    buf = ctypes.create_unicode_buffer(260)
    ctypes.windll.kernel32.GetShortPathNameW(str(box_ws / ".env"), buf, 260)
    short = Path(buf.value).name
    if short.lower() == ".env":
        pytest.skip("8.3 names are disabled on this volume")
    assert classify(f"cat {short}", box_ws) == ASK


def test_rm_rf_on_a_subfolder_asks_rather_than_denies():
    assert classify("rm -rf ./build") == ASK
    assert classify("rm -rf /tmp/scratch") == ASK
