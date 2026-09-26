"""Which shell commands run at once, which ask the user, which never run.

This is a convenience layer, not security: `python -c "import shutil; ..."` is
just "a command that asks". Real containment is the sandbox (sandbox.py), and
on Windows there is none, so the prompt is the only guard there.
"""

from __future__ import annotations

import re
import shlex

ALLOW, ASK, DENY = "allow", "ask", "deny"

READ_ONLY = {
    "ls", "dir", "pwd", "echo", "printf", "cat", "head", "tail", "wc", "grep", "egrep", "rg",
    "find", "which", "whoami", "date", "file", "stat", "du", "df", "tree", "sort", "uniq",
    "cut", "diff", "basename", "dirname", "realpath", "true", "false", "nl",
}
GIT_READ_ONLY = {"status", "log", "diff", "show", "rev-parse", "ls-files", "blame"}
# Flags that turn an otherwise read-only command into one that writes or runs things.
UNSAFE_FLAGS = {
    "find": {"-exec", "-execdir", "-delete", "-ok", "-okdir", "-fprint", "-fprintf", "-fls"},
    "sort": {"-o", "--output"},
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


def classify(command: str) -> str:
    """Return ALLOW, ASK or DENY for one shell command line."""
    if any(p.search(command) for p in NEVER):
        return DENY
    # Command substitution and process substitution run code we cannot see here.
    if any(s in command for s in ("$(", "`", "<(", ">(")):
        return ASK
    try:
        lexer = shlex.shlex(command.replace("\n", " ; "), posix=True, punctuation_chars=True)
        lexer.whitespace_split = True
        tokens = list(lexer)
    except ValueError:  # unbalanced quotes
        return ASK
    if not tokens:
        return ASK

    segments: list[list[str]] = [[]]
    i = 0
    while i < len(tokens):
        tok = tokens[i]
        if tok in SEPARATORS:
            segments.append([])
        elif ">" in tok and set(tok) <= set("<>&|"):
            target = tokens[i + 1] if i + 1 < len(tokens) else ""
            harmless = target in ("/dev/null", "NUL") or ("&" in tok and target.isdigit())
            if not harmless:
                return ASK  # writes a file
            i += 1  # skip the redirection target
        elif set(tok) <= set("<"):
            i += 1  # input redirection reads a file; skip its target
        else:
            segments[-1].append(tok)
        i += 1

    for seg in segments:
        if not seg:
            continue
        name = seg[0].rsplit("/", 1)[-1].rsplit("\\", 1)[-1].lower().removesuffix(".exe")
        if name == "git":
            sub = next((t for t in seg[1:] if not t.startswith("-")), "")
            if sub not in GIT_READ_ONLY:
                return ASK
            continue
        if name not in READ_ONLY:
            return ASK
        if UNSAFE_FLAGS.get(name, set()) & set(seg[1:]):
            return ASK
    return ALLOW
