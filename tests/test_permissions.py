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
    "/usr/bin/ls",
    "tree",
])
def test_read_only_commands_run_at_once(command):
    assert classify(command) == ALLOW


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
    assert classify("sort words.txt | uniq -c") == ALLOW


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


def test_plain_commands_still_run_under_cmd():
    assert classify("dir", cmd_shell=True) == ALLOW
    assert classify("type notes.txt", cmd_shell=True) == ASK  # `type` is not on the list


def test_rm_rf_on_a_subfolder_asks_rather_than_denies():
    assert classify("rm -rf ./build") == ASK
    assert classify("rm -rf /tmp/scratch") == ASK
