"""Which shell commands run at once, which ask the user, which never run.

The rule for running without asking is an allowlist, not a list of dangers:

- every command is a bare program name (no path: `./ls` could be a script the
  model just wrote) from READ_ONLY, a set chosen because none of those programs
  has an option that writes a file or runs another program; programs that do
  (`sort -o`, `file -C`, `tree -o`, `date -s`) are simply not on it, because
  bundled short options (`-uo`) and abbreviated long ones (`--outp`) make a
  blocklist of their flags unreliable;
- git only as `git <read-only subcommand> ...`, with no global options, no
  `:` pathspecs or object paths, and only when the repository is the workspace;
- no expansion the shell would perform on text this parser cannot see (`$`,
  backticks, process substitution, braces);
- every argument that could be a path (including one glued to a flag, or after
  a `:`) resolves inside the workspace, following globs and symlinks, and is
  not a credentials file; recursive readers (`grep -r`, `rg`, `diff -r`) ask
  whenever the workspace holds one;
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
    "ls", "dir", "pwd", "echo", "printf", "cat", "head", "tail", "wc", "grep", "egrep", "fgrep", "rg",
    "find", "which", "whoami", "stat", "du", "df", "basename", "dirname", "realpath", "true", "false",
    "nl", "cut", "diff", "uniq",
}
GIT_READ_ONLY = {"status", "log", "diff", "show", "rev-parse", "ls-files", "blame"}
# The few options of READ_ONLY programs (and git) that write or execute. None of
# these programs bundles them or is sort/file/tree/date; long options are also
# matched by prefix, since git accepts `--outp` for `--output`.
UNSAFE_FLAGS = {
    "find": {"-exec", "-execdir", "-delete", "-ok", "-okdir", "-fprint", "-fprint0", "-fprintf", "-fls"},
    "rg": {"--pre", "--pre-glob", "--search-zip", "-z"},
    "git": {"--output", "--ext-diff", "--textconv", "--exec", "--upload-pack", "--open-files-in-pager", "-O"},
}
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
_CMD_PLAIN = re.compile(r"(dir|echo)( [\w.\-/\\]+)*")
_SKIP_DIRS = {".git", "node_modules", ".venv", "venv", "__pycache__", ".harness", ".tox", "dist", "build"}
_SCAN_LIMIT = 50_000


def is_secret_file(name: str) -> bool:
    """Credentials by file name: .env files, SSH and PuTTY keys, .netrc, *.pem, ..."""
    return bool(_SECRET_FILE.match(re.split(r"[\\/]", name)[-1]))


def workspace_has_secrets(workspace: Path) -> bool:
    """True if any credentials file is in the workspace (or the tree is too big to tell)."""
    seen = 0
    for root, dirs, files in os.walk(workspace):
        dirs[:] = [d for d in dirs if d not in _SKIP_DIRS]
        seen += len(files) + len(dirs)
        if any(is_secret_file(f) for f in files):
            return True
        if seen > _SCAN_LIMIT:
            return True
    return False


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


def _recursive(name: str, args: list[str]) -> bool:
    if name == "rg":
        return True
    letters = RECURSIVE.get(name)
    if not letters:
        return False
    for a in args:
        if a in ("--recursive", "--dereference-recursive") or a.startswith("--directories=r") or a.startswith("--recursive"):
            return True
        if a.startswith("-") and not a.startswith("--") and any(c in a[1:] for c in letters):
            return True
    return False


def _repo_is_workspace(workspace: Path) -> bool:
    if (workspace / ".git").exists():
        return True
    return not any((parent / ".git").exists() for parent in workspace.parents)


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
    return workspace is None or _repo_is_workspace(workspace)


def classify(command: str, workspace: Path | None = None, *, cmd_shell: bool = False) -> str:
    """Return ALLOW, ASK or DENY for one shell command line."""
    if any(p.search(command) for p in NEVER_RAW):
        return DENY
    if any(s in command for s in ("$", "`", "<(", ">(")):
        return ASK
    if cmd_shell and not _CMD_PLAIN.fullmatch(command.strip()):
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
        if _recursive(name, args) and (workspace is None or workspace_has_secrets(workspace)):
            return ASK
        for a in args:
            read_targets += _candidates(a)

    if any(unsafe_target(t, workspace) for t in read_targets if t):
        return ASK
    return ALLOW
