"""A small task suite for the model arm: fourteen coding tasks, each with a check.

Every task starts from a few files, asks for one change or one answer, and has
a check that decides pass or fail from the workspace (and the final answer)
alone, by running the code where it can. Each task also carries a reference
solution: tests apply it to prove the check passes on a right answer and fails
on the untouched start, so a model's score measures the model, not the check.
"""

from __future__ import annotations

import json
import subprocess
import sys
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

Check = Callable[[Path, str], tuple[bool, str]]


@dataclass(frozen=True)
class Edit:
    """One step of a reference solution: replace `old` by `new` in `path`, or
    (with old=None) write `new` as the whole file."""

    path: str
    new: str
    old: str | None = None


@dataclass(frozen=True)
class Task:
    name: str
    prompt: str
    files: dict[str, str | bytes]
    check: Check
    solution: tuple[Edit, ...] = ()
    answer: str = ""  # the reference final answer, for questions
    tags: tuple[str, ...] = field(default=())


def run_py(ws: Path, *args: str, stdin: str | None = None, timeout: float = 60) -> tuple[int, str]:
    """Run Python in the workspace; (exit code, stdout + stderr)."""
    try:
        p = subprocess.run(
            [sys.executable, *args],
            cwd=ws,
            capture_output=True,
            text=True,
            timeout=timeout,
            input=stdin,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return -1, "timed out"
    return p.returncode, (p.stdout + p.stderr).strip()


def _prints(script: str, expected: str, *args: str) -> Check:
    def check(ws: Path, answer: str) -> tuple[bool, str]:
        if not (ws / script).is_file():
            return False, f"{script} missing"
        code, out = run_py(ws, script, *args)
        return code == 0 and out == expected, f"exit {code}: {out[-200:]}"

    return check


def _tests_pass(test_file: str, unchanged: dict[str, str] | None = None) -> Check:
    def check(ws: Path, answer: str) -> tuple[bool, str]:
        for path, text in (unchanged or {}).items():
            if (ws / path).read_text(encoding="utf-8") != text:
                return False, f"{path} was changed (the fix belongs elsewhere)"
        code, out = run_py(ws, test_file)
        return code == 0, f"exit {code}: {out[-200:]}"

    return check


def _answer_has(*needles: str) -> Check:
    def check(ws: Path, answer: str) -> tuple[bool, str]:
        missing = [n for n in needles if n.lower() not in answer.lower()]
        return not missing, f"missing {missing}" if missing else "found"

    return check


# ---- the tasks ------------------------------------------------------------------

FIB_OUT = "0 1 1 2 3 5 8 13 21 34"

MATHUTIL = (
    "def total(xs):\n    return sum(xs[1:])\n\n\ndef mean(xs):\n    return total(xs) / len(xs)\n"
)
TEST_MATHUTIL = (
    "from mathutil import mean, total\n\n"
    "assert total([1, 2, 3]) == 6, total([1, 2, 3])\n"
    "assert mean([2, 4]) == 3\n"
    "print('ok')\n"
)

USERS = "def get_usr(uid):\n    return {'id': uid, 'name': f'user{uid}'}\n"
APP = "from users import get_usr\n\nprint(get_usr(7)['name'])\n"

GREET = (
    "import argparse\n\n"
    "p = argparse.ArgumentParser()\n"
    "p.add_argument('name')\n"
    "a = p.parse_args()\n"
    "print(f'hello {a.name}')\n"
)

CONFIG = {"host": "localhost", "port": 8000, "debug": False, "workers": 4}

SLUG = "import re\n\n\ndef slugify(text):\n    return re.sub(r'[^a-z0-9]+', '-', text).strip('-')\n"
TEST_SLUG = (
    "from slug import slugify\n\n"
    "assert slugify('Hello World') == 'hello-world', slugify('Hello World')\n"
    "assert slugify('  a  b ') == 'a-b'\n"
    "print('ok')\n"
)

CRLF_SETTINGS = "TIMEOUT = 30\r\nRETRIES = 3\r\nNAME = 'svc'\r\n"

WORDCOUNT = (
    "import sys\n\n"
    "counts = {}\n"
    "for word in sys.stdin.read().split():\n"
    "    counts[word] = counts.get(word, 0) + 1\n"
    "for word in sorted(counts):\n"
    "    print(word, counts[word])\n"
)

LONG_MODULE = "".join(f"CONST_{i} = {i}\n" for i in range(400))

PKG_PARSE = (
    "def parse_row(row):\n"
    "    if not row:\n"
    "        raise ValueError('empty row')\n"
    "    return row.split(',')\n"
)
PKG_FORMAT = "def format_row(cells):\n    return ','.join(cells)\n"

MAIN_IMPORT = "from helpers import double\n\nprint(double(21))\n"
HELPERS = "def double(x):\n    return 2 * x\n"

DATA_CSV = "id,value\n" + "".join(f"{i},{i * 3}\n" for i in range(1, 38))

EVEN = "def is_even(n):\n    return n % 2 == 0\n"


def _config_port(ws: Path, answer: str) -> tuple[bool, str]:
    try:
        cfg = json.loads((ws / "config.json").read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        return False, f"config.json unreadable: {e}"
    expected = {**CONFIG, "port": 8080}
    return cfg == expected, f"got {cfg}"


def _renamed(ws: Path, answer: str) -> tuple[bool, str]:
    text = "".join(p.read_text(encoding="utf-8") for p in ws.glob("*.py"))
    if "get_usr" in text or "get_user" not in text:
        return False, "get_usr still present or get_user missing"
    code, out = run_py(ws, "app.py")
    return code == 0 and out == "user7", f"exit {code}: {out[-200:]}"


def _verbose_flag(ws: Path, answer: str) -> tuple[bool, str]:
    code_a, out_a = run_py(ws, "greet.py", "ann", "--verbose")
    code_b, out_b = run_py(ws, "greet.py", "ann")
    ok = code_a == 0 and out_a == "verbose on\nhello ann" and code_b == 0 and out_b == "hello ann"
    return ok, f"with: {out_a[-120:]!r} without: {out_b[-120:]!r}"


def _crlf_kept(ws: Path, answer: str) -> tuple[bool, str]:
    raw = (ws / "settings.py").read_bytes()
    expected = CRLF_SETTINGS.replace("TIMEOUT = 30", "TIMEOUT = 60").encode()
    return raw == expected, f"got {raw!r}"


def _one_line_changed(ws: Path, answer: str) -> tuple[bool, str]:
    expected = LONG_MODULE.replace("CONST_350 = 350\n", "CONST_350 = 351\n")
    got = (ws / "consts.py").read_text(encoding="utf-8")
    return got == expected, "exact" if got == expected else "other lines changed or edit missing"


def _files_written(ws: Path, answer: str) -> tuple[bool, str]:
    want = {"a.txt": "1", "b.txt": "2", "c.txt": "3"}
    got = {
        n: (ws / n).read_text(encoding="utf-8").strip() if (ws / n).exists() else None for n in want
    }
    return got == want, f"got {got}"


def _wordcount(ws: Path, answer: str) -> tuple[bool, str]:
    code, out = run_py(ws, "wordcount.py", stdin="b a b\nc a b\n")
    return code == 0 and out == "b 3\na 2\nc 1", f"exit {code}: {out[-200:]!r}"


def _test_written(ws: Path, answer: str) -> tuple[bool, str]:
    test = ws / "test_even.py"
    if not test.is_file() or "assert" not in test.read_text(encoding="utf-8"):
        return False, "test_even.py missing or has no assert"
    code, out = run_py(ws, "test_even.py")
    if code != 0:
        return False, f"the new test fails: {out[-200:]}"
    # It must test something: against a broken is_even it has to fail.
    (ws / "even.py").write_text("def is_even(n):\n    return True\n", encoding="utf-8")
    try:
        broken, _ = run_py(ws, "test_even.py")
    finally:
        (ws / "even.py").write_text(EVEN, encoding="utf-8")
    return broken != 0, "catches a broken is_even" if broken else "passes on a broken is_even"


TASKS: tuple[Task, ...] = (
    Task(
        "create_fib",
        "Create fib.py that prints the first 10 Fibonacci numbers, starting 0 1, on one line "
        "separated by spaces. Run it to check.",
        {},
        _prints("fib.py", FIB_OUT),
        (
            Edit(
                "fib.py",
                "a, b = 0, 1\nout = []\nfor _ in range(10):\n    out.append(a)\n"
                "    a, b = b, a + b\nprint(*out)\n",
            ),
        ),
        tags=("create",),
    ),
    Task(
        "fix_off_by_one",
        "python test_mathutil.py fails. Fix the bug in mathutil.py (do not change the test).",
        {"mathutil.py": MATHUTIL, "test_mathutil.py": TEST_MATHUTIL},
        _tests_pass("test_mathutil.py", {"test_mathutil.py": TEST_MATHUTIL}),
        (Edit("mathutil.py", "    return sum(xs)\n", "    return sum(xs[1:])\n"),),
        tags=("debug",),
    ),
    Task(
        "rename_function",
        "Rename the function get_usr to get_user everywhere in this project. app.py must "
        "still run.",
        {"users.py": USERS, "app.py": APP},
        _renamed,
        (
            Edit("users.py", "def get_user(", "def get_usr("),
            Edit(
                "app.py",
                "from users import get_user\n\nprint(get_user(",
                "from users import get_usr\n\nprint(get_usr(",
            ),
        ),
        tags=("refactor",),
    ),
    Task(
        "add_verbose_flag",
        "Add a --verbose flag to greet.py. With it, print 'verbose on' on its own line before "
        "the greeting; without it, the output must not change.",
        {"greet.py": GREET},
        _verbose_flag,
        (
            Edit(
                "greet.py",
                "p.add_argument('name')\np.add_argument('--verbose', action='store_true')\n"
                "a = p.parse_args()\nif a.verbose:\n    print('verbose on')\n",
                "p.add_argument('name')\na = p.parse_args()\n",
            ),
        ),
        tags=("feature",),
    ),
    Task(
        "json_config_edit",
        "In config.json, change the port to 8080. Keep every other setting as it is.",
        {"config.json": json.dumps(CONFIG, indent=2) + "\n"},
        _config_port,
        (Edit("config.json", '"port": 8080', '"port": 8000'),),
        tags=("edit",),
    ),
    Task(
        "fix_slugify",
        "python test_slug.py fails. Fix slugify in slug.py so the test passes.",
        {"slug.py": SLUG, "test_slug.py": TEST_SLUG},
        _tests_pass("test_slug.py", {"test_slug.py": TEST_SLUG}),
        (Edit("slug.py", "text.lower()).strip('-')", "text).strip('-')"),),
        tags=("debug",),
    ),
    Task(
        "crlf_edit",
        "In settings.py, change TIMEOUT from 30 to 60. Change nothing else.",
        {"settings.py": CRLF_SETTINGS.encode()},
        _crlf_kept,
        (Edit("settings.py", "TIMEOUT = 60", "TIMEOUT = 30"),),
        tags=("edit", "line-endings"),
    ),
    Task(
        "sort_by_count",
        "wordcount.py prints each word and its count, sorted by word. Make it sort by count, "
        "highest first, and by word among equal counts.",
        {"wordcount.py": WORDCOUNT},
        _wordcount,
        (
            Edit(
                "wordcount.py",
                "for word in sorted(counts, key=lambda w: (-counts[w], w)):",
                "for word in sorted(counts):",
            ),
        ),
        tags=("feature",),
    ),
    Task(
        "edit_line_350",
        "In consts.py, set CONST_350 to 351. Leave the other 399 lines exactly as they are.",
        {"consts.py": LONG_MODULE},
        _one_line_changed,
        (Edit("consts.py", "CONST_350 = 351\n", "CONST_350 = 350\n"),),
        tags=("edit", "long-file"),
    ),
    Task(
        "three_files",
        "Create a.txt, b.txt and c.txt containing 1, 2 and 3 respectively. Keep a todo list.",
        {},
        _files_written,
        (Edit("a.txt", "1\n"), Edit("b.txt", "2\n"), Edit("c.txt", "3\n")),
        tags=("create", "todos"),
    ),
    Task(
        "fix_import",
        "python main.py fails with an import error. Fix main.py so it prints 42.",
        {"main.py": MAIN_IMPORT.replace("helpers", "helper"), "helpers.py": HELPERS},
        _prints("main.py", "42"),
        (Edit("main.py", "from helpers import", "from helper import"),),
        tags=("debug",),
    ),
    Task(
        "write_a_test",
        "Write test_even.py: a plain script (no pytest) that asserts is_even from even.py "
        "gives the right answer for 0, 1, 2 and -3, and exits non-zero if not.",
        {"even.py": EVEN},
        _test_written,
        (
            Edit(
                "test_even.py",
                "from even import is_even\n\nassert is_even(0)\n"
                "assert not is_even(1)\nassert is_even(2)\nassert not is_even(-3)\n",
            ),
        ),
        tags=("tests",),
    ),
    Task(
        "find_the_raiser",
        "Which function in the pkg folder raises ValueError on empty input? Answer with its "
        "name and file.",
        {"pkg/parse.py": PKG_PARSE, "pkg/format.py": PKG_FORMAT, "pkg/__init__.py": ""},
        _answer_has("parse_row", "parse.py"),
        answer="parse_row, in pkg/parse.py",
        tags=("explore",),
    ),
    Task(
        "count_rows",
        "How many data rows does data.csv have, not counting the header? Answer with the number.",
        {"data.csv": DATA_CSV},
        _answer_has("37"),
        answer="37",
        tags=("explore",),
    ),
)


def setup(task: Task, ws: Path) -> None:
    for rel, content in task.files.items():
        path = ws / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(content, bytes):
            path.write_bytes(content)
        else:
            path.write_bytes(content.encode("utf-8"))


def apply_solution(task: Task, ws: Path) -> str:
    """Apply the reference solution directly (no agent); returns the reference answer."""
    for e in task.solution:
        path = ws / e.path
        if e.old is None:
            path.write_bytes(e.new.encode("utf-8"))
        else:
            raw = path.read_bytes().decode("utf-8")
            crlf = "\r\n" in raw
            text = raw.replace("\r\n", "\n")
            assert text.count(e.old) == 1, (task.name, e.old)
            text = text.replace(e.old, e.new)
            path.write_bytes((text.replace("\n", "\r\n") if crlf else text).encode("utf-8"))
    return task.answer or "done"
