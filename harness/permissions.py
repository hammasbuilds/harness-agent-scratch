"""Which shell commands run at once, which ask the user, which never run.

A command runs without asking only if every part of it is on the read-only
list, uses no flag that makes that command write or execute something, names
no path outside the workspace and no credentials file, and contains no `$`
expansion. `cat ~/.ssh/id_rsa` is a read-only command, but it asks, exactly as
read_file does for the same file.

This is a convenience layer, not security: `python -c "import shutil; ..."` is
just "a command that asks". Real containment is the sandbox (sandbox.py), and
on Windows there is none, so the prompt is the only guard there.
"""

from __future__ import annotations

import re
import shlex
from pathlib import Path

ALLOW, ASK, DENY = "allow", "ask", "deny"

READ_ONLY = {
    "ls", "dir", "pwd", "echo", "printf", "cat", "head", "tail", "wc", "grep", "egrep", "rg",
    "find", "which", "whoami", "date", "file", "stat", "du", "df", "tree", "sort", "uniq",
    "cut", "diff", "basename", "dirname", "realpath", "true", "false", "nl",
}
GIT_READ_ONLY = {"status", "log", "diff", "show", "rev-parse", "ls-files", "blame"}
# Options that turn an otherwise read-only command into one that writes a file
# or runs another program. Matched on the part before any "=".
UNSAFE_FLAGS = {
    "find": {"-exec", "-execdir", "-delete", "-ok", "-okdir", "-fprint", "-fprint0", "-fprintf", "-fls"},
    "sort": {"-o", "--output", "--compress-program"},
    "rg": {"--pre", "--pre-glob"},
    "tree": {"-o", "-R"},
    "date": {"-s", "--set"},
    "file": {"-C", "--compile"},
    "git": {"-c", "--config-env", "--exec-path", "--output", "--ext-diff"},
}

NEVER = [
    re.compile(r"\brm\s+(-\w*\s+)*-\w*[rR]\w*\s+(-\w*\s+)*(/|~|\$HOME|/\*|[A-Za-z]:[\\/]?)(\s|$)"),
    re.compile(r"\bmkfs(\.\w+)?\b"),
    re.compile(r"\bdd\b.*\bof=/dev/"),
    re.compile(r":\(\)\s*\{\s*:\s*\|\s*:\s*&\s*\}"),  # fork bomb
    re.compile(r"\b(shutdown|reboot|halt|poweroff)\b"),
    re.compile(r"\bformat\s+[A-Za-z]:"),
    re.compile(r"\bdiskpart\b"),
]

SEPARATORS = {"|", "||", "&", "&&", ";", ";;", "\n"}
HARMLESS_PATHS = {"/dev/null", "NUL", "-"}
_DRIVE = re.compile(r"^[A-Za-z]:[\\/]")


def _unsafe_flag(name: str, args: list[str]) -> bool:
    flags = UNSAFE_FLAGS.get(name, set())
    for arg in args:
        key = arg.split("=", 1)[0]
        if key in flags:
            return True
        # sort's short options bundle: `sort -uo out.txt` writes out.txt.
        if name == "sort" and re.fullmatch(r"-[a-zA-Z]*o\S*", arg):
            return True
    return False


_SECRET_FILE = re.compile(
    r"^(\.env(\.(?!example$|sample$|template$|dist$)[\w.-]+)?|\.netrc|\.pgpass|\.npmrc|\.pypirc|"
    r"id_(rsa|dsa|ecdsa|ed25519)(\.pub)?|credentials(\.json)?|.*\.(pem|key|p12|pfx|keystore|jks))$",
    re.I,
)
# cmd.exe gives these meaning that shlex does not model (single quotes are plain
# characters there, so `echo ' & mkdir x & echo '` runs mkdir).
_CMD_SPECIAL = re.compile(r"[&|<>^%'\"()!\r\n]")


def is_secret_file(name: str) -> bool:
    """Credentials by file name: .env files, SSH keys, .netrc, *.pem and friends."""
    return bool(_SECRET_FILE.match(re.split(r"[\\/]", name)[-1]))


def points_outside(token: str, workspace: Path) -> bool:
    """True if `token` looks like a path and resolves outside the workspace.

    Only absolute paths, `~` paths and paths with a `..` part are checked; a
    bare word is relative to the workspace by construction. A grep pattern
    such as "/api/" also looks absolute, so it asks: a false alarm, not a leak.
    """
    if token in HARMLESS_PATHS:
        return False
    parts = re.split(r"[\\/]", token)
    if not (token.startswith(("/", "~", "\\\\")) or _DRIVE.match(token) or ".." in parts):
        return False
    try:
        p = Path(token).expanduser()
        resolved = (p if p.is_absolute() else workspace / p).resolve()
    except (OSError, RuntimeError, ValueError):
        return True
    return not (resolved == workspace or resolved.is_relative_to(workspace))


def classify(command: str, workspace: Path | None = None, *, cmd_shell: bool = False) -> str:
    """Return ALLOW, ASK or DENY for one shell command line.

    With `workspace`, any path argument that leaves it makes the command ask,
    and so does any argument naming a credentials file. `cmd_shell` is for
    cmd.exe, whose quoting rules differ from the POSIX ones parsed here.
    """
    if any(p.search(command) for p in NEVER):
        return DENY
    # `$` expands to text we cannot see here: $HOME/.ssh/id_rsa is a path outside
    # the workspace, $(...) runs a command. Backticks and process substitution too.
    if any(s in command for s in ("$", "`", "<(", ">(")):
        return ASK
    if cmd_shell and _CMD_SPECIAL.search(command):
        return ASK
    try:
        lexer = shlex.shlex(command.replace("\n", " ; "), posix=True, punctuation_chars=True)
        lexer.whitespace_split = True
        # shlex would treat `#` as the start of a comment and hide everything after
        # it; bash does so only at the start of a word. `ls #\nrm x` must see the rm.
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
        elif ">" in tok and set(tok) <= set("<>&|"):
            harmless = target in HARMLESS_PATHS or ("&" in tok and target.isdigit())
            if not harmless:
                return ASK  # writes a file
            i += 1  # skip the redirection target
        elif set(tok) <= set("<"):
            read_targets.append(target)  # `< file` reads it
            i += 1
        else:
            segments[-1].append(tok)
        i += 1

    for seg in segments:
        if not seg:
            continue
        name = seg[0].rsplit("/", 1)[-1].rsplit("\\", 1)[-1].lower().removesuffix(".exe")
        args = seg[1:]
        if name == "git":
            sub = next((t for t in args if not t.startswith("-")), "")
            if sub not in GIT_READ_ONLY:
                return ASK
        elif name not in READ_ONLY:
            return ASK
        if _unsafe_flag(name, args):
            return ASK
        # `uniq IN OUT` overwrites OUT.
        if name == "uniq" and len([a for a in args if not a.startswith("-") or a == "-"]) >= 2:
            return ASK
        read_targets += [a.split("=", 1)[1] if a.startswith("-") and "=" in a else a for a in args]

    targets = [t for t in read_targets if t]
    if any(is_secret_file(t) for t in targets):
        return ASK
    if workspace is not None and any(points_outside(t, workspace) for t in targets):
        return ASK
    return ALLOW
