import pytest

from harness.permissions import ALLOW, ASK, DENY, classify


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


def test_rm_rf_on_a_subfolder_asks_rather_than_denies():
    assert classify("rm -rf ./build") == ASK
    assert classify("rm -rf /tmp/scratch") == ASK
