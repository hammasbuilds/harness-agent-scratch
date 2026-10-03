"""How often the permission check lets a command run unasked, on three corpora.

    uv run python scripts/permission_study.py

1. corpora/swesmith_commands.jsonl: 300 distinct shell commands, sampled at
   random (seed 0) from the 1,453 distinct ones a real agent (Claude 3.7
   Sonnet, in 229 SWE-smith trajectories) issued. Each is labelled by hand:
   `read` (only reads files inside the repository) or `change` (runs code,
   deletes, installs). /testbed, the trajectories' repository root, is mapped
   onto a scratch workspace.
2. corpora/terminal_agent_risky.jsonl: 826 commands labelled dangerous / safe /
   neutral by a different project (hammasbuilds/terminal-agent) for its own
   safety policy, so the labels were not written with this checker in mind.
3. corpora/review_bypasses.jsonl: every command the review rounds used to get
   past earlier versions, as pinned in tests/test_permissions.py. These pass by
   construction; the file is a regression check, not evidence.

Writes results/permission_classifier.json. Rates carry Wilson 95% intervals.
"""

from __future__ import annotations

import json
import math
import os
import re
import shlex
import sys
import tempfile
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]

from harness.permissions import ALLOW, ASK, DENY, READ_ONLY, classify  # noqa: E402

CORPORA = ROOT / "corpora"
OUT = ROOT / "results" / "permission_classifier.json"
_SEPARATORS = {"|", "||", "&", "&&", ";", "\n"}
_CD_PREFIX = re.compile(r"^cd (\S+) && ")


def wilson(k: int, n: int, z: float = 1.96) -> list[float]:
    if n == 0:
        return [0.0, 0.0]
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return [round(max(0.0, centre - half), 4), round(min(1.0, centre + half), 4)]


def rate(k: int, n: int) -> dict:
    return {"k": k, "n": n, "rate": round(k / n, 4) if n else None, "ci95": wilson(k, n)}


def load(name: str) -> list[dict]:
    with open(CORPORA / name, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def make_repo(ws: Path) -> None:
    (ws / ".git").mkdir()
    (ws / ".git" / "config").write_text("[core]\n\trepositoryformatversion = 0\n\tbare = false\n")


def off_list_programs(command: str) -> list[str]:
    """First words of the command's segments that are not on the allowlist,
    or `{` / `$` when the command holds an expansion (which always asks)."""
    lexer = shlex.shlex(command, posix=True, punctuation_chars=True)
    lexer.whitespace_split = True
    try:
        tokens = list(lexer)
    except ValueError:
        return ["(unbalanced quotes)"]
    names, start = [], True
    for tok in tokens:
        if tok in _SEPARATORS:
            start = True
        elif start:
            if tok not in READ_ONLY and tok != "git":
                names.append(tok)
            start = False
    names += [c for c in ("{", "$") if c in command]
    return names


def swesmith(ws: Path) -> dict:
    rows = [r for r in load("swesmith_commands.jsonl") if r["label"] != "not_a_command"]
    root = ws.as_posix()
    verdicts, stripped = [], []
    for r in rows:
        command = r["command"].replace("/testbed", root)
        verdicts.append(classify(command, ws))
        # What-if: a leading `cd <workspace root> &&` treated as a no-op.
        m = _CD_PREFIX.match(command)
        if m and Path(m.group(1)).resolve() == ws:
            command = command[m.end() :]
        stripped.append(classify(command, ws))

    def scores(decisions: list[str]) -> dict:
        tp = sum(d == ALLOW and r["label"] == "read" for d, r in zip(decisions, rows, strict=True))
        fp = sum(
            d == ALLOW and r["label"] == "change" for d, r in zip(decisions, rows, strict=True)
        )
        reads = sum(r["label"] == "read" for r in rows)
        return {
            "allow_precision": rate(tp, tp + fp),
            "allow_recall": rate(tp, reads),
            "unsafe_allows": fp,
        }

    counts = Counter(verdicts)
    asked_reads = [
        r["command"]
        for d, r in zip(verdicts, rows, strict=True)
        if d != ALLOW and r["label"] == "read"
    ]
    blockers = Counter(p for c in asked_reads for p in set(off_list_programs(c)))
    return {
        "source": "SWE-bench/SWE-smith-trajectories, ticks split, file ticks-00001-of-00008, "
        "row group 3 (229 trajectories, 2,986 shell commands, 1,453 distinct)",
        "n": len(rows),
        "labels": dict(Counter(r["label"] for r in rows)),
        "excluded_not_a_command": sum(
            r["label"] == "not_a_command" for r in load("swesmith_commands.jsonl")
        ),
        "decisions": {d: rate(counts[d], len(rows)) for d in (ALLOW, ASK, DENY)},
        # Distinct commands above; here each counts as often as the agent issued it.
        "allow_share_weighted_by_times_issued": round(
            sum(r["times_issued"] for d, r in zip(verdicts, rows, strict=True) if d == ALLOW)
            / sum(r["times_issued"] for r in rows),
            4,
        ),
        "vs_labels": scores(verdicts),
        "asked_reads": len(asked_reads),
        "asked_reads_first_word_off_the_allowlist": dict(blockers.most_common()),
        "what_if_cd_to_workspace_root_were_allowed": scores(stripped),
    }


def terminal_agent(ws: Path) -> dict:
    rows = load("terminal_agent_risky.jsonl")
    verdicts = [classify(r["command"], ws) for r in rows]
    out = {}
    for label in ("dangerous", "neutral", "safe"):
        mine = [d for d, r in zip(verdicts, rows, strict=True) if r["label"] == label]
        c = Counter(mine)
        out[label] = {d: rate(c[d], len(mine)) for d in (ALLOW, ASK, DENY)}
    allowed_dangerous = [
        r["command"]
        for d, r in zip(verdicts, rows, strict=True)
        if d == ALLOW and r["label"] == "dangerous"
    ]
    allowed_safe = Counter(
        r.get("category")
        for d, r in zip(verdicts, rows, strict=True)
        if d == ALLOW and r["label"] == "safe"
    )
    return {
        "source": "hammasbuilds/terminal-agent data/risky_commands*.jsonl (four splits, labelled "
        "for that project's policy)",
        "n": len(rows),
        "labels": dict(Counter(r["label"] for r in rows)),
        "by_label": out,
        "dangerous_allowed": allowed_dangerous,
        "safe_allowed_by_category": dict(allowed_safe.most_common()),
    }


def bypasses(ws: Path) -> dict:
    rows = load("review_bypasses.jsonl")
    verdicts = [classify(r["command"], ws, cmd_shell=r["cmd_shell"]) for r in rows]
    allowed = [r["command"] for d, r in zip(verdicts, rows, strict=True) if d == ALLOW]
    deny_rows = [d for d, r in zip(verdicts, rows, strict=True) if r["expect"] == "deny"]
    return {
        "note": "commands pinned by tests/test_permissions.py; zero allowed is by construction",
        "n": len(rows),
        "allowed": rate(len(allowed), len(rows)),
        "allowed_commands": allowed,
        "expected_deny_denied": rate(sum(d == DENY for d in deny_rows), len(deny_rows)),
    }


def bypass_workspace(base: Path) -> Path:
    """The fixture the bypass tests use, plus a .env, a key file and a link out."""
    ws = base / "ws"
    (ws / "src").mkdir(parents=True)
    (ws / "a.txt").write_text("a")
    (ws / "src" / "m.py").write_text("x")
    (ws / "--").mkdir()
    (ws / ".env").write_text("HARNESS_API_KEY=sk")
    (ws / "id_rsa").write_text("KEY")
    (base / "secret.txt").write_text("s")
    (base / "elsewhere").mkdir()
    make_repo(ws)
    if os.name == "nt":
        import _winapi

        _winapi.CreateJunction(str(base / "elsewhere"), str(ws / "linked"))
    else:
        (ws / "linked").symlink_to(base / "elsewhere", target_is_directory=True)
    return ws.resolve()


def main() -> int:
    with tempfile.TemporaryDirectory() as tmp:
        base = Path(tmp).resolve()
        clean = base / "testbed"
        clean.mkdir()
        make_repo(clean)
        result = {
            "what": "decisions of harness.permissions.classify (allow = runs without asking)",
            "swesmith_agent_commands": swesmith(clean),
            "terminal_agent_labelled_commands": terminal_agent(clean),
            "review_bypass_regressions": bypasses(bypass_workspace(base / "box")),
        }
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    s = result["swesmith_agent_commands"]
    t = result["terminal_agent_labelled_commands"]["by_label"]
    b = result["review_bypass_regressions"]
    print(
        f"SWE-smith agent commands (n={s['n']}): allow {s['decisions']['allow']['rate']}, "
        f"precision {s['vs_labels']['allow_precision']['rate']}, "
        f"recall {s['vs_labels']['allow_recall']['rate']}"
    )
    print(
        f"  what-if cd-to-root allowed: recall "
        f"{s['what_if_cd_to_workspace_root_were_allowed']['allow_recall']['rate']}"
    )
    print(
        f"terminal-agent: dangerous allowed {t['dangerous']['allow']['k']}/"
        f"{t['dangerous']['allow']['n']}, safe allowed {t['safe']['allow']['k']}/"
        f"{t['safe']['allow']['n']}"
    )
    print(f"review bypasses allowed: {b['allowed']['k']}/{b['n']}")
    print(f"wrote {OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
