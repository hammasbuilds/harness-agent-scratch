"""Which shell commands run at once, which ask the user, which never run.

The rule for running without asking is an allowlist, not a list of dangers:

- every command is a bare program name (no path: `./ls` could be a script the
  model just wrote) from READ_ONLY, a set chosen because none of those programs
  has an option that writes a file or runs another program; programs that do
  (`sort -o`, `file -C`, `tree -o`, `date -s`, `rg --pre`/`--hostname-bin`) are
  simply not on it, because bundled short options (`-uo`), abbreviated long ones
  (`--outp`) and options added in new versions make a blocklist of their flags
  unreliable. The one exception is find, whose actions (`-exec`, `-delete`, ...)
  are whole words that can be neither bundled nor abbreviated;
- git only as `git <read-only subcommand> ...`, with no global options, no
  `:` pathspecs or object paths, and only when the repository is the workspace
  and its config holds nothing but the keys a plain clone writes;
- no expansion the shell would perform on text this parser cannot see (`$`,
  backticks, process substitution, braces);
- every argument that could be a path (including one glued to a flag, or after
  a `:`) resolves inside the workspace, following globs and symlinks, and is
  not a credentials file; recursive readers (`grep -r`, `diff -r`) ask
  whenever the workspace holds one, and they and link-following walkers ask
  whenever a symlink or junction leads out of it;
- output redirection only to /dev/null, and no operator the parser does not
  model.

Anything else asks. This is still a convenience layer rather than a security
boundary: an approved command runs with the user's rights, and on Windows there
is no sandbox behind it. It errs towards asking: a grep pattern such as "/api/"
looks like an absolute path and asks.
"""

from __future__ import annotations

import glob
import os
import re
import shlex
from pathlib import Path

ALLOW, ASK, DENY = "allow", "ask", "deny"

READ_ONLY = {
    "ls", "dir", "pwd", "echo", "printf", "cat", "head", "tail", "wc", "grep", "egrep", "fgrep",
    "find", "which", "whoami", "stat", "du", "df", "basename", "dirname", "realpath", "true", "false",
    "nl", "cut", "diff", "uniq",
}
GIT_READ_ONLY = {"status", "log", "diff", "show", "rev-parse", "ls-files", "blame"}
# Options that write, execute or read indirectly, for the programs where such
# options are whole words (find) or are checked beside other rules (git, whose
# long options are also matched by prefix, since it accepts `--outp` for
# `--output`).
UNSAFE_FLAGS = {
    # -files0-from (GNU find 4.9+) takes its starting points from a file the
    # model can write, which could list anything outside the workspace.
    "find": {"-exec", "-execdir", "-delete", "-ok", "-okdir", "-fprint", "-fprint0", "-fprintf", "-fls",
             "-files0-from"},
    "git": {"--output", "--ext-diff", "--textconv", "--exec", "--upload-pack", "--open-files-in-pager", "-O"},
    # Read the names of the files to read from another file, which the model can
    # write without asking and fill with paths outside the workspace.
    "wc": {"--files0-from"},
    "du": {"--files0-from"},
}
# A repository's own config can make read-only git commands run programs
# (diff.external, diff.<driver>.command, core.fsmonitor, gpg.program with
# log.showSignature, textconv, a pager, ...). The model cannot write .git, but an
# unpacked archive can bring one. A list of dangerous keys missed three of them,
# so this is the keys a plain `git init`/`git clone` writes, and nothing else.
_GIT_PLAIN_KEYS = {
    "core": {"repositoryformatversion", "filemode", "bare", "logallrefupdates", "symlinks", "ignorecase",
             "autocrlf", "eol", "safecrlf", "precomposeunicode", "longpaths", "quotepath", "checkstat"},
    "remote": {"url", "pushurl", "fetch", "tagopt", "prune", "promisor", "partialclonefilter"},
    "branch": {"remote", "merge", "rebase", "pushremote", "description"},
    "user": {"name", "email"},
    "init": {"defaultbranch"},
    "lfs": {"repositoryformatversion"},
}
_GIT_SECTION = re.compile(r'^\[\s*([A-Za-z0-9.-]+)(?:\s+"[^"]*")?\s*\]\s*$')
_GIT_KEY = re.compile(r"^([A-Za-z][A-Za-z0-9-]*)\s*(=|$)")
RECURSIVE = {"grep": "rR", "egrep": "rR", "fgrep": "rR", "diff": "r"}  # short letters meaning "recurse"
NEVER_NAMES = re.compile(r"^(mkfs(\.\w+)?|shutdown|reboot|halt|poweroff|diskpart|format)$")
NEVER_RAW = [
    re.compile(r"(^|[;&|(\s])rm\s+(-\w*\s+)*-\w*[rR]\w*\s+(-\w*\s+)*(/|~|\$HOME|/\*|[A-Za-z]:[\\/]?)(\s|$)"),
    re.compile(r"(^|[;&|(\s])dd\s.*\bof=/dev/"),
    re.compile(r":\(\)\s*\{\s*:\s*\|\s*:\s*&\s*\}"),  # fork bomb
]

SEPARATORS = {"|", "||", "&", "&&", ";", ";;", "\n"}
_DRIVE = re.compile(r"^[A-Za-z]:[\\/]")
_GLOB = set("*?[")
_SECRET_FILE = re.compile(
    r"^(\.env(\.(?!example$|sample$|template$|dist$)[\w.-]+)?|.+\.env|\.envrc|\.netrc|_netrc|\.pgpass|\.npmrc|"
    r"\.pypirc|\.git-credentials|\.htpasswd|id_(rsa|dsa|ecdsa|ed25519)(\.pub)?|credentials(\.json)?|"
    r".*\.(pem|key|p12|pfx|keystore|jks|ppk|kdbx))$",
    re.I,
)
# cmd.exe searches the current folder before PATH and gives these characters
# meanings shlex does not model; only a plain `dir`/`echo` runs without asking.
# No backslashes (POSIX shlex would drop them, hiding `..\..` and `\\host\share`)
# and no `..` at all.
_CMD_PLAIN = re.compile(r"(dir|echo)( [\w.\-/]+)*")
# Only these are left out of the scan: grep -r does not skip build/, dist/ or
# node_modules, so a .env in any of them must count.
_SKIP_DIRS = {".git", ".harness"}
_SCAN_LIMIT = 50_000


def is_secret_file(name: str) -> bool:
    """Credentials by file name: .env files, SSH and PuTTY keys, .netrc, *.pem, ..."""
    return bool(_SECRET_FILE.match(re.split(r"[\\/]", name)[-1]))


def workspace_exposure(workspace: Path) -> tuple[bool, bool]:
    """(holds a credentials file, holds a symlink or junction leading outside).

    A recursive reader would read the first; one that follows links would walk
    out through the second. A tree too big to scan counts as both.
    """
    secrets = outward = False
    seen = 0
    for root, dirs, files in os.walk(workspace):
        dirs[:] = [d for d in dirs if d not in _SKIP_DIRS]
        seen += len(files) + len(dirs)
        if seen > _SCAN_LIMIT:
            return True, True
        secrets = secrets or any(is_secret_file(f) for f in files)
        for name in dirs + files:
            p = Path(root) / name
            if (p.is_symlink() or getattr(os.path, "isjunction", lambda _: False)(p)) and \
                    _outside_or_secret(str(p), workspace):
                outward = True
        if secrets and outward:
            break
    return secrets, outward


def workspace_has_secrets(workspace: Path) -> bool:
    return workspace_exposure(workspace)[0]


def _outside_or_secret(path: str, workspace: Path) -> bool:
    try:
        p = Path(path).expanduser()
        resolved = (p if p.is_absolute() else workspace / p).resolve()
    except (OSError, RuntimeError, ValueError):
        return True
    if not (resolved == workspace or resolved.is_relative_to(workspace)):
        return True
    # resolve() turns an 8.3 short name (ENV~1) into the real one (.env).
    return resolved.exists() and is_secret_file(resolved.name)


def unsafe_target(token: str, workspace: Path | None) -> bool:
    """True if `token`, read as a path, could leave the workspace or name a credentials file."""
    if token in ("-", "/dev/null"):
        return False
    if "{" in token or "}" in token or is_secret_file(token):
        return True
    if workspace is None:
        return False
    parts = re.split(r"[\\/]", token)
    pathlike = token.startswith(("/", "~", "\\\\")) or bool(_DRIVE.match(token)) or ".." in parts
    if _GLOB & set(token):
        if any(part.startswith(".") and _GLOB & set(part) for part in parts):
            return True  # a hidden-file glob can match ..
        fixed = re.split(r"[*?\[]", token, maxsplit=1)[0]
        if pathlike and _outside_or_secret(fixed or ".", workspace):
            return True
        p = Path(token).expanduser()
        pattern = str(p if p.is_absolute() else workspace / token)
        # Hidden files included: `[.]env` matches .env in a shell with dotglob set,
        # and Python's glob would otherwise skip it and report nothing to check.
        try:
            matches = glob.glob(pattern, include_hidden=True)
        except TypeError:  # Python 3.10
            matches = glob.glob(pattern) + [m for m in glob.glob(str(Path(pattern).parent / ".*"))
                                            if Path(m).name != ".."]
        return any(_outside_or_secret(m, workspace) for m in matches)
    if pathlike or os.path.lexists(workspace / token):
        return _outside_or_secret(token, workspace)
    return False


def _candidates(arg: str) -> list[str]:
    """The parts of one argument that could be read as paths."""
    if arg.startswith("--"):
        values = [arg.split("=", 1)[1]] if "=" in arg else []
    elif arg.startswith("-") and len(arg) > 2:
        values = [arg[2:]]  # a value glued to a short option: grep -f.env
    elif arg.startswith("-"):
        values = []
    else:
        values = [arg]
    out = []
    for v in values:
        out.append(v)
        if ":" in v and not _DRIVE.match(v):
            out.append(v.split(":", 1)[1])  # HEAD:.env, :/pathspec
    return [v for v in out if v]


def _flag_matches(arg: str, flags: set[str]) -> bool:
    key = arg.split("=", 1)[0]
    if key in flags:
        return True
    # GNU/git long options may be abbreviated to any unique prefix.
    return key.startswith("--") and len(key) > 2 and any(f.startswith(key) for f in flags if f.startswith("--"))


RECURSIVE_LONG = ("--recursive", "--dereference-recursive", "--directories")
# Options that make a directory walker follow symlinks out of the tree.
FOLLOW_LINKS = {"find": {"-L", "-H", "-follow"}, "ls": set("LH"), "dir": set("LH"), "du": set("LHD")}


def _recursive(name: str, args: list[str]) -> bool:
    """Does this command read file contents through a directory tree?

    grep has several spellings: -r, -R, -d recurse, --directories=recurse and
    any abbreviation getopt accepts (--rec, --dir). `-d` and --directories are
    treated as recursion whatever their value, which only ever over-asks.
    """
    letters = RECURSIVE.get(name)
    if not letters:
        return False
    if name != "diff":
        letters += "d"
    for a in args:
        key = a.split("=", 1)[0]
        if key.startswith("--") and len(key) >= 4 and any(opt.startswith(key) for opt in RECURSIVE_LONG):
            return True
        if a.startswith("-") and not a.startswith("--") and any(c in a[1:] for c in letters):
            return True
    return False


def _follows_links(name: str, args: list[str]) -> bool:
    flags = FOLLOW_LINKS.get(name)
    if not flags:
        return False
    for a in args:
        if name == "find":
            if a in flags:
                return True
        elif a.startswith("--"):
            if len(a) >= 5 and "--dereference".startswith(a.split("=", 1)[0]):
                return True
        elif a.startswith("-") and any(c in a[1:] for c in flags):
            return True
    return False


def _repo_is_workspace(workspace: Path) -> bool:
    if (workspace / ".git").exists():
        return True
    return not any((parent / ".git").exists() for parent in workspace.parents)


def repo_is_inert(workspace: Path) -> bool:
    """True if a read-only git command here can run nothing but git.

    A plain config is not enough on its own: `git status` rewrites the index and
    so fires .git/hooks/post-index-change, and it recurses into submodules,
    whose own configs (filters, fsmonitor) are not the one checked here.
    """
    if not repo_config_is_plain(workspace):
        return False
    git = workspace / ".git"
    hooks = git / "hooks"
    if hooks.is_dir() and any(not p.name.endswith(".sample") for p in hooks.iterdir()):
        return False
    if (git / "modules").exists() or (workspace / ".gitmodules").exists():
        return False
    return not _has_nested_repo(workspace)


def _has_nested_repo(workspace: Path) -> bool:
    """A repository inside the workspace (a gitlink with no .gitmodules): git
    status recurses into it and runs whatever its own config says."""
    seen = 0
    for root, dirs, files in os.walk(workspace):
        if Path(root) == workspace:
            dirs[:] = [d for d in dirs if d not in (".git", ".harness")]
        elif ".git" in dirs or ".git" in files:
            return True
        seen += len(dirs) + len(files)
        if seen > _SCAN_LIMIT:
            return True  # too big to be sure
    return False


def repo_config_is_plain(workspace: Path) -> bool:
    """True only for a real `.git` folder in the workspace whose config holds
    nothing but the keys a plain clone writes.

    No `.git` folder means False, not "nothing to check": the model can write
    HEAD, objects/, refs/ and a config (worktree = ., fsmonitor = ./hook.sh) at
    the workspace root, none of it guarded, and git then treats the root itself
    as the repository and runs the hook on `git status`.
    """
    git = workspace / ".git"
    if not git.is_dir() or git.is_symlink():
        return False  # missing, a gitdir pointer file, or a link elsewhere
    if (git / "commondir").exists() or (git / "config.worktree").exists():
        return False  # the config actually used lives elsewhere, or in a second file
    try:
        text = (git / "config").read_text(encoding="utf-8", errors="replace")
    except OSError:
        return False
    section = None
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith(("#", ";")):
            continue
        header = _GIT_SECTION.match(line)
        if header:
            section = header.group(1).lower()
            continue
        key = _GIT_KEY.match(line)
        if not key or section is None or key.group(1).lower() not in _GIT_PLAIN_KEYS.get(section, set()):
            return False  # unknown key, unknown section, or a line this parser cannot read
    return True


def _git_ok(args: list[str], workspace: Path | None) -> bool:
    i = 0
    while i < len(args) and args[i] == "--no-pager":
        i += 1
    if i >= len(args) or args[i] not in GIT_READ_ONLY:
        return False  # also rejects every global option: -C, -c, --namespace, --work-tree, ...
    rest = args[i + 1:]
    if any(":" in a for a in rest):
        return False  # pathspec magic (:/) and object paths (HEAD:../x) reach past the workspace
    if any(_flag_matches(a, UNSAFE_FLAGS["git"]) for a in rest):
        return False
    return workspace is None or (_repo_is_workspace(workspace) and repo_is_inert(workspace))


def classify(command: str, workspace: Path | None = None, *, cmd_shell: bool = False) -> str:
    """Return ALLOW, ASK or DENY for one shell command line."""
    if any(p.search(command) for p in NEVER_RAW):
        return DENY
    if any(s in command for s in ("$", "`", "<(", ">(")):
        return ASK
    if cmd_shell and (not _CMD_PLAIN.fullmatch(command.strip()) or ".." in command):
        return ASK
    try:
        lexer = shlex.shlex(command.replace("\n", " ; "), posix=True, punctuation_chars=True)
        lexer.whitespace_split = True
        # shlex would read `#` as a comment and hide everything after it; bash
        # does so only at the start of a word. `ls #\nrm x` must show the rm.
        lexer.commenters = ""
        tokens = list(lexer)
    except ValueError:  # unbalanced quotes
        return ASK
    if not tokens:
        return ASK

    segments: list[list[str]] = [[]]
    read_targets: list[str] = []
    i = 0
    while i < len(tokens):
        tok = tokens[i]
        target = tokens[i + 1] if i + 1 < len(tokens) else ""
        if tok in SEPARATORS:
            segments.append([])
        elif tok in (">", ">>", "&>", ">&", "&>>") or (tok.endswith(">") and set(tok) <= set("<>&|")):
            if not (target == "/dev/null" or (tok == ">&" and target.isdigit())):
                return ASK  # writes a file (`> -` creates one named "-")
            i += 1
        elif tok == "<":
            read_targets.append(target)
            i += 1
        elif tok == "<<<":
            i += 1  # a here-string's word is data, not a file
        elif set(tok) <= set("&|;<>()"):
            # Anything else (`|&`, `<>`, `<<`, `(`) is an operator this parser does
            # not model; reading `|&` as an argument once let `cat a|&rm a` run rm.
            return ASK
        else:
            segments[-1].append(tok)
        i += 1

    for seg in segments:
        if not seg:
            continue
        raw_name, args = seg[0], seg[1:]
        if "/" in raw_name or "\\" in raw_name or "=" in raw_name:
            return ASK  # ./ls may be the model's own script; FOO=1 cmd changes the environment
        name = raw_name.lower().removesuffix(".exe")
        if NEVER_NAMES.match(name):
            return DENY
        if name == "git":
            if not _git_ok(args, workspace):
                return ASK
        elif name not in READ_ONLY:
            return ASK
        elif any(_flag_matches(a, UNSAFE_FLAGS.get(name, set())) for a in args):
            return ASK
        # `uniq IN OUT` overwrites OUT.
        if name == "uniq" and len([a for a in args if not a.startswith("-") or a == "-"]) >= 2:
            return ASK
        reads_tree, follows = _recursive(name, args), _follows_links(name, args)
        if reads_tree or follows:
            if workspace is None:
                return ASK
            secrets, outward = workspace_exposure(workspace)
            if outward or (reads_tree and secrets):
                return ASK
        for a in args:
            read_targets += _candidates(a)

    if any(unsafe_target(t, workspace) for t in read_targets if t):
        return ASK
    return ALLOW
