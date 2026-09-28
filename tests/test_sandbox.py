import os
import subprocess

import pytest

from harness.sandbox import (
    CMD_RUNNER,
    COMMAND_VAR,
    POSIX_RUNNER,
    SEATBELT_PROFILE,
    Sandbox,
    SandboxUnavailable,
    find_shell,
    system_cmd,
    trusted_which,
)


def which_only(*names):
    return lambda n: os.path.abspath(f"/usr/bin/{n}") if n in names else None


def test_linux_with_bwrap_confines_writes_and_network(tmp_path):
    sb = Sandbox(tmp_path, "auto", system="Linux", which=which_only("bwrap"))
    argv = sb.wrap(["bash", "-c", POSIX_RUNNER])
    ws = str(tmp_path.resolve())
    assert sb.kind == "bwrap"
    assert argv[:5] == [os.path.abspath("/usr/bin/bwrap"), "--ro-bind", "/", "/", "--dev"]
    for flag in (
        "--unshare-net",
        "--unshare-pid",
        "--unshare-ipc",
        "--new-session",
        "--die-with-parent",
    ):
        assert flag in argv
    # /run holds the D-Bus and docker sockets, both ways out of a read-only bind
    assert argv[argv.index("/run") - 1] == "--tmpfs"
    # the workspace bind must come after both tmpfs mounts, or a workspace under /tmp is hidden
    assert max(i for i, a in enumerate(argv) if a == "--tmpfs") < argv.index("--bind")
    assert argv[argv.index("--bind") + 1 : argv.index("--bind") + 3] == [ws, ws]
    assert argv[-3:] == ["bash", "-c", POSIX_RUNNER]


def test_macos_uses_seatbelt_with_the_workspace_parameter(tmp_path):
    sb = Sandbox(tmp_path, "auto", system="Darwin", which=which_only("sandbox-exec"))
    argv = sb.wrap(["bash", "-c", POSIX_RUNNER])
    assert argv[:3] == [os.path.abspath("/usr/bin/sandbox-exec"), "-p", SEATBELT_PROFILE]
    assert argv[3:5] == ["-D", f"WORKSPACE={tmp_path.resolve()}"]
    assert "(deny network*)" in SEATBELT_PROFILE


def test_a_sandbox_program_inside_the_workspace_is_not_trusted(tmp_path):
    fake = str(tmp_path / "bwrap")
    assert Sandbox(tmp_path, "auto", system="Linux", which=lambda n: fake).kind == "none"


def test_windows_has_no_sandbox_and_says_so(tmp_path):
    sb = Sandbox(tmp_path, "auto", system="Windows", which=which_only())
    assert sb.kind == "none"
    assert sb.wrap(["x"]) == ["x"]
    assert "full user rights" in sb.describe()


def test_required_mode_refuses_to_run_unconfined(tmp_path):
    with pytest.raises(SandboxUnavailable):
        Sandbox(tmp_path, "required", system="Windows", which=which_only())


def test_none_mode_skips_an_available_sandbox(tmp_path):
    assert Sandbox(tmp_path, "none", system="Linux", which=which_only("bwrap")).kind == "none"


def test_bad_mode(tmp_path):
    with pytest.raises(ValueError):
        Sandbox(tmp_path, "sometimes")


def test_windows_prefers_git_bash_over_wsl_bash():
    argv, desc = find_shell(
        system="Windows",
        which=lambda n: r"C:\Windows\System32\bash.exe",
        exists=lambda p: p.endswith(r"Git\bin\bash.exe"),
    )
    assert argv[0].endswith(r"Git\bin\bash.exe") and argv[1:] == ["-c", POSIX_RUNNER]
    assert "Git Bash" in desc


def test_windows_never_picks_wsl_bash_and_falls_back_to_an_absolute_cmd():
    argv, _ = find_shell(
        system="Windows", which=lambda n: r"C:\Windows\System32\bash.exe", exists=lambda p: False
    )
    assert argv == [system_cmd(), "/d", "/c", CMD_RUNNER]
    assert argv[0].lower().endswith(r"system32\cmd.exe")  # never a cmd.exe from the current folder


def test_windows_finds_portable_git_next_to_git_exe():
    git = r"C:\Users\me\tools\PortableGit\cmd\git.exe"
    bash = r"C:\Users\me\tools\PortableGit\bin\bash.exe"
    argv, desc = find_shell(
        system="Windows", which=lambda n: git if n == "git" else None, exists=lambda p: p == bash
    )
    assert argv == [bash, "-c", POSIX_RUNNER] and "Git Bash" in desc


def test_windows_never_picks_the_windowsapps_wsl_alias():
    alias = r"C:\Users\me\AppData\Local\Microsoft\WindowsApps\bash.exe"
    argv, _ = find_shell(
        system="Windows", which=lambda n: alias if n == "bash" else None, exists=lambda p: False
    )
    assert argv == [system_cmd(), "/d", "/c", CMD_RUNNER]


def test_posix_uses_bash_then_sh():
    assert find_shell(system="Linux", which=which_only("bash"))[0] == [
        os.path.abspath("/usr/bin/bash"),
        "-c",
        POSIX_RUNNER,
    ]
    assert find_shell(system="Linux", which=which_only())[0] == ["/bin/sh", "-c", POSIX_RUNNER]


def test_configured_shell_wins():
    assert find_shell("/bin/zsh", system="Linux")[0] == ["/bin/zsh", "-c", POSIX_RUNNER]
    assert find_shell(r"C:\Windows\System32\cmd.exe", system="Windows")[0][1:] == [
        "/d",
        "/c",
        CMD_RUNNER,
    ]


# ---- where programs are looked up ---------------------------------------------


def test_a_relative_path_entry_is_never_trusted():
    # A `.` in PATH resolves against the current folder, which may be the repository.
    assert trusted_which("git", which=lambda n: os.path.join(".", "git.exe")) is None


def test_a_program_inside_the_workspace_is_never_trusted(tmp_path):
    inside = str(tmp_path / "tools" / "git.exe")
    assert trusted_which("git", tmp_path, which=lambda n: inside) is None
    outside = os.path.abspath("/opt/git/bin/git")
    assert trusted_which("git", tmp_path, which=lambda n: outside) == outside


def test_windows_git_bash_found_through_a_repo_shipped_git_is_refused(tmp_path):
    # A repository's own git.exe must not lead find_shell to the repository's bash.exe.
    git = str(tmp_path / "cmd" / "git.exe")
    argv, _ = find_shell(
        system="Windows",
        which=lambda n: git if n == "git" else None,
        exists=lambda p: p == str(tmp_path / "bin" / "bash.exe"),
        workspace=tmp_path,
    )
    assert argv[0] == system_cmd()


@pytest.mark.skipif(os.name != "nt", reason="the current-folder lookup is a Windows behaviour")
def test_windows_current_folder_lookup_is_off():
    assert os.environ.get("NODEFAULTCURRENTDIRECTORYINEXEPATH") == "1"  # names are case-blind


# ---- the command reaches the shell unchanged -----------------------------------


@pytest.mark.skipif(find_shell()[0][0] == system_cmd(), reason="needs a POSIX shell")
@pytest.mark.parametrize(
    "command",
    [
        "printf %s\\\\n a{b,c}",  # braces with no space: Git Bash's runtime used to expand them
        "printf %s\\\\n 'x*y'",  # a quoted glob
        "cat<'a.txt\n>important.py'",  # the reviewer's case: one quoted word to bash
        'printf %s\\\\n "q\'q"',
    ],
)
def test_the_shell_receives_exactly_the_checked_string(tmp_path, command):
    argv, _ = find_shell()
    (tmp_path / "a.txt").write_text("hello")
    (tmp_path / "important.py").write_text("keep me")
    # Ask the shell to echo back what it was given, before running anything.
    probe = argv[:-1] + [f'printf %s "${COMMAND_VAR}"']
    got = subprocess.run(
        probe,
        cwd=tmp_path,
        capture_output=True,
        env={**os.environ, COMMAND_VAR: command},
        encoding="utf-8",
        check=False,
    ).stdout
    assert got == command
    subprocess.run(
        argv,
        cwd=tmp_path,
        capture_output=True,
        env={**os.environ, COMMAND_VAR: command},
        check=False,  # only the file matters here, not the exit code
    )
    assert (tmp_path / "important.py").read_text() == "keep me"
