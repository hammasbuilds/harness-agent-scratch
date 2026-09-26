import pytest

from harness.sandbox import SEATBELT_PROFILE, Sandbox, SandboxUnavailable, find_shell


def which_only(*names):
    return lambda n: f"/usr/bin/{n}" if n in names else None


def test_linux_with_bwrap_confines_writes_and_network(tmp_path):
    sb = Sandbox(tmp_path, "auto", system="Linux", which=which_only("bwrap"))
    argv = sb.wrap(["bash", "-c", "ls"])
    ws = str(tmp_path.resolve())
    assert sb.kind == "bwrap"
    assert argv[:5] == ["bwrap", "--ro-bind", "/", "/", "--dev"]
    for flag in ("--unshare-net", "--unshare-pid", "--unshare-ipc", "--new-session", "--die-with-parent"):
        assert flag in argv
    # /run holds the D-Bus and docker sockets, both ways out of a read-only bind
    assert argv[argv.index("/run") - 1] == "--tmpfs"
    # the workspace bind must come after both tmpfs mounts, or a workspace under /tmp is hidden
    assert max(i for i, a in enumerate(argv) if a == "--tmpfs") < argv.index("--bind")
    assert argv[argv.index("--bind") + 1:argv.index("--bind") + 3] == [ws, ws]
    assert argv[-3:] == ["bash", "-c", "ls"]


def test_macos_uses_seatbelt_with_the_workspace_parameter(tmp_path):
    sb = Sandbox(tmp_path, "auto", system="Darwin", which=which_only("sandbox-exec"))
    argv = sb.wrap(["bash", "-c", "ls"])
    assert argv[:3] == ["sandbox-exec", "-p", SEATBELT_PROFILE]
    assert argv[3:5] == ["-D", f"WORKSPACE={tmp_path.resolve()}"]
    assert "(deny network*)" in SEATBELT_PROFILE


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
    argv, desc = find_shell(system="Windows", which=lambda n: r"C:\Windows\System32\bash.exe",
                            exists=lambda p: p.endswith(r"Git\bin\bash.exe"))
    assert argv[0].endswith(r"Git\bin\bash.exe") and argv[1] == "-c"
    assert "Git Bash" in desc


def test_windows_never_picks_wsl_bash_and_falls_back_to_cmd():
    argv, desc = find_shell(system="Windows", which=lambda n: r"C:\Windows\System32\bash.exe",
                            exists=lambda p: False)
    assert argv == ["cmd.exe", "/d", "/c"]


def test_windows_finds_portable_git_next_to_git_exe():
    git = r"C:\Users\me\tools\PortableGit\cmd\git.exe"
    bash = r"C:\Users\me\tools\PortableGit\bin\bash.exe"
    argv, desc = find_shell(system="Windows", which=lambda n: git if n == "git" else None,
                            exists=lambda p: p == bash)
    assert argv == [bash, "-c"] and "Git Bash" in desc


def test_windows_never_picks_the_windowsapps_wsl_alias():
    alias = r"C:\Users\me\AppData\Local\Microsoft\WindowsApps\bash.exe"
    argv, _ = find_shell(system="Windows", which=lambda n: alias if n == "bash" else None, exists=lambda p: False)
    assert argv == ["cmd.exe", "/d", "/c"]


def test_posix_uses_bash_then_sh():
    assert find_shell(system="Linux", which=which_only("bash"))[0] == ["/usr/bin/bash", "-c"]
    assert find_shell(system="Linux", which=which_only())[0] == ["/bin/sh", "-c"]


def test_configured_shell_wins():
    assert find_shell("/bin/zsh", system="Linux")[0] == ["/bin/zsh", "-c"]
    assert find_shell(r"C:\Windows\System32\cmd.exe", system="Windows")[0][1:] == ["/d", "/c"]
