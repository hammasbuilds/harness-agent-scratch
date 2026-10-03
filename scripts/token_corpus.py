"""The texts the token estimate is calibrated on: 36 kinds, 5 samples each.

Every sample is 3,000 characters, the size of one capped tool result. Real
text comes from this repository's own history at pinned commits (so a rerun
reads the same bytes whatever the working tree holds); the rest is generated
from fixed seeds. Kinds were chosen to cover what reaches a coding agent's
context (prose, code, JSON, diffs, listings, logs, numbers) and the text the
estimator was previously wrong on (digits, DNA, symbols, three-byte scripts).
"""

from __future__ import annotations

import base64
import difflib
import hashlib
import json
import random
import subprocess
import uuid
from collections.abc import Callable
from functools import partial
from pathlib import Path

SIZE = 3000
SAMPLES = 5
ROOT = Path(__file__).resolve().parent.parent
# Commits of this repository the real-text kinds are read from.
PINNED = "83723b2344f664a8edf8a0180a8fe4265a837bee"
FIRST = "34d8198"


def git_show(rev: str, path: str) -> str:
    try:
        return subprocess.run(
            ["git", "show", f"{rev}:{path}"],
            cwd=ROOT,
            capture_output=True,
            check=True,
            encoding="utf-8",
        ).stdout
    except (OSError, subprocess.CalledProcessError) as e:
        raise SystemExit(
            f"cannot read {path} at {rev[:7]} ({e}); the calibration needs this repository's "
            "full git history (a shallow clone lacks it)"
        ) from None


def windows(text: str, i: int) -> str:
    """The i-th 3,000-character window of a long text, spread across it."""
    if len(text) < SIZE:
        text = (text + "\n") * (SIZE // max(1, len(text)) + 2)
    step = max(1, (len(text) - SIZE) // SAMPLES)
    return text[i * step : i * step + SIZE]


def repeat_shuffled(sentences: list[str], i: int, sep: str = " ") -> str:
    rng = random.Random(1000 + i)
    out: list[str] = []
    while sum(len(s) + 1 for s in out) < SIZE + 200:
        out.extend(rng.sample(sentences, len(sentences)))
    return sep.join(out)[:SIZE]


def random_text(alphabet: str, i: int, line: int = 0) -> str:
    rng = random.Random(2000 + i)
    chars = [rng.choice(alphabet) for _ in range(SIZE)]
    if line:
        for k in range(line, SIZE, line + 1):
            chars[k] = "\n"
    return "".join(chars)


# ---- generated kinds ----------------------------------------------------------


def seq_output(i: int) -> str:
    start = 1 + i * 20_000
    return "\n".join(str(n) for n in range(start, start + 5000))[:SIZE]


def csv_numeric(i: int) -> str:
    rows = (
        f"{(n + i) * 7919 % 100000:05d},{n * 104729 % 1000:03d}.{n % 97:02d}" for n in range(400)
    )
    return "\n".join(rows)[:SIZE]


def ls_la(i: int) -> str:
    rng = random.Random(3000 + i)
    names = ["agent.py", "cli.py", "README.md", "data.csv", "notes.txt", "setup.cfg", "main.rs"]
    lines = []
    for n in range(80):
        size = rng.randint(0, 999_999)
        day, hour, minute = rng.randint(1, 28), rng.randint(0, 23), rng.randint(0, 59)
        name = f"{rng.choice(names).split('.')[0]}_{n}.{rng.choice(names).split('.')[1]}"
        lines.append(
            f"-rw-r--r-- 1 dev 197121 {size:>7} Sep {day:>2} {hour:02d}:{minute:02d} {name}"
        )
    return "\n".join(lines)[:SIZE]


def ls_la_pinned(i: int) -> str:
    """The exact listing tests/test_compaction.py pins a real count for."""
    return ("-rw-r--r-- 1 dell 197121  48213 Sep 26 12:07 file_1.py\n" * 60)[:SIZE]


def json_records(i: int) -> str:
    rng = random.Random(4000 + i)
    records = [
        {
            "id": rng.randint(1, 10**6),
            "name": rng.choice(["alpha", "beta", "gamma", "delta"]) + f"-{n}",
            "active": rng.random() < 0.5,
            "score": round(rng.random() * 100, 3),
            "tags": rng.sample(["api", "db", "ui", "infra", "ml", "docs"], 2),
        }
        for n in range(60)
    ]
    return json.dumps(records, indent=2)[:SIZE]


def git_log(i: int) -> str:
    rng = random.Random(5000 + i)
    subjects = [
        line.strip("- *") for line in git_show(PINNED, "README.md").splitlines() if len(line) > 40
    ][:200]
    lines = [f"{rng.getrandbits(28):07x} {rng.choice(subjects)[:70]}" for _ in range(80)]
    return "\n".join(lines)[:SIZE]


def traceback_text(i: int) -> str:
    rng = random.Random(6000 + i)
    frames = []
    for n in range(40):
        mod = rng.choice(["agent", "tools", "llm", "compaction", "cli"])
        frames.append(
            f'  File "/home/dev/project/harness/{mod}.py", line {rng.randint(1, 900)}, in '
            f"{rng.choice(['send', 'call', 'chat', '_post', 'compact'])}\n"
            f"    result = self.{rng.choice(['toolbox', 'llm', 'agent'])}.run(args[{n}])"
        )
    return (
        "Traceback (most recent call last):\n" + "\n".join(frames) + "\nKeyError: 'tool_calls'"
    )[:SIZE]


def hex_hashes(i: int) -> str:
    return "\n".join(hashlib.sha256(f"{i}-{n}".encode()).hexdigest() for n in range(60))[:SIZE]


def base64_text(i: int) -> str:
    rng = random.Random(7000 + i)
    return base64.b64encode(rng.randbytes(2250)).decode()[:SIZE]


def base32_text(i: int) -> str:
    rng = random.Random(7500 + i)
    return base64.b32encode(rng.randbytes(1900)).decode()[:SIZE]


def uuids(i: int) -> str:
    rng = random.Random(8000 + i)
    return "\n".join(str(uuid.UUID(int=rng.getrandbits(128))) for _ in range(90))[:SIZE]


def file_paths(i: int) -> str:
    rng = random.Random(8500 + i)
    parts = ["src", "tests", "harness", "lib", "internal", "vendor", "docs", "api", "utils", "core"]
    exts = [".py", ".ts", ".go", ".md", ".json", ".yaml"]
    lines = [
        "/".join(rng.sample(parts, rng.randint(2, 5))) + f"/file_{n}" + rng.choice(exts)
        for n in range(120)
    ]
    return "\n".join(lines)[:SIZE]


def html(i: int) -> str:
    rng = random.Random(9000 + i)
    rows = "".join(
        f'<tr class="row-{n}"><td><a href="/items/{rng.randint(1, 9999)}">Item {n}</a></td>'
        f"<td>{rng.random():.2f}</td></tr>\n"
        for n in range(60)
    )
    return f"<html><body><table>\n{rows}</table></body></html>"[:SIZE]


def unified_diff(i: int) -> str:
    before = git_show(FIRST, "harness/tools.py").splitlines(keepends=True)
    after = git_show(PINNED, "harness/tools.py").splitlines(keepends=True)
    diff = "".join(difflib.unified_diff(before, after, "a/harness/tools.py", "b/harness/tools.py"))
    return windows(diff, i)


LONG_WORDS = [
    "internationalization",
    "Donaudampfschifffahrtsgesellschaft",
    "counterrevolutionaries",
    "incomprehensibilities",
    "Rechtsschutzversicherungsgesellschaften",
    "uncharacteristically",
    "electroencephalography",
    "Kraftfahrzeughaftpflichtversicherung",
    "antiestablishmentarian",
]

LANGUAGES = {
    "chinese": [
        "这是一个用于测试的中文句子。",
        "代理读取文件并运行测试。",
        "今天的天气很好，我们去公园散步。",
        "模型在循环中调用工具。",
    ],
    "japanese": [
        "これはテスト用の日本語の文です。",
        "エージェントがファイルを読み、テストを実行します。",
        "今日はとても良い天気ですね。",
    ],
    "korean": [
        "이것은 테스트를 위한 한국어 문장입니다.",
        "에이전트가 파일을 읽고 테스트를 실행합니다.",
        "오늘은 날씨가 아주 좋습니다.",
    ],
    "russian": [
        "Это предложение на русском языке для проверки.",
        "Агент читает файл и запускает тесты.",
        "Сегодня очень хорошая погода.",
    ],
    "hindi": [
        "नमस्ते दुनिया, यह एक परीक्षण है।",
        "एजेंट फ़ाइल पढ़ता है और परीक्षण चलाता है।",
        "आज मौसम बहुत अच्छा है।",
    ],
    "georgian": [
        "გამარჯობა მსოფლიო, ეს არის ტესტი.",
        "აგენტი კითხულობს ფაილს და უშვებს ტესტებს.",
        "დღეს ძალიან კარგი ამინდია.",
    ],
    "thai": ["สวัสดีชาวโลก นี่คือการทดสอบ", "เอเจนต์อ่านไฟล์และเรียกใช้การทดสอบ", "วันนี้อากาศดีมาก"],
    "amharic": [
        "ሰላም ልዑል ዓለም ይህ የሙከራ ጽሑፍ ነው።",
        "ወኪሉ ፋይሉን ያነባል እና ሙከራዎችን ያካሂዳል።",
        "ዛሬ የአየሩ ሁኔታ በጣም ጥሩ ነው።",
    ],
    "hebrew": [
        "שלום עולם, זהו מבחן.",
        "הסוכן קורא את הקובץ ומריץ את הבדיקות.",
        "היום מזג האוויר נעים מאוד.",
    ],
    "arabic": [
        "مرحبا بالعالم، هذا اختبار.",
        "الوكيل يقرأ الملف ويشغل الاختبارات.",
        "الطقس جميل جدا اليوم.",
    ],
}

SYMBOLS = {
    "math_symbols": [
        "∀x∈ℝ: x²≥0",
        "∑ᵢ aᵢ ≤ ∏ⱼ bⱼ",
        "f: A → B ⇒ g∘f",
        "∫₀^∞ e⁻ˣ dx = 1",
        "√2 ≈ 1.414 ≠ π",
    ],
    "braille_spinners": ["⠋ building", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"],
    "emoji": [
        "✅ tests passed 🎉",
        "❌ build failed 🔥",
        "🚀 deployed",
        "⚠️ warning 🐛",
        "📦 packaged ✨",
    ],
    "box_drawing": ["┌──────┬──────┐", "│ name │ size │", "├──────┼──────┤", "└──────┴──────┘"],
}


def build() -> dict[str, list[str]]:
    """Every kind with its samples, each exactly SIZE characters (or the text's full length)."""
    readme = git_show(PINNED, "README.md")
    code = git_show(PINNED, "harness/tools.py")
    prose = "\n\n".join(
        p
        for p in readme.split("\n\n")
        if p and not p.lstrip().startswith(("|", "<", "`", "#", "-"))
    )
    table = "\n".join(line for line in readme.splitlines() if line.startswith("|"))
    kinds: dict[str, Callable[[int], str]] = {
        "prose_readme": lambda i: windows(prose, i),
        "python_code": lambda i: windows(code, i),
        "markdown_table": lambda i: windows(table, i),
        "unified_diff": unified_diff,
        "json_records": json_records,
        "html": html,
        "traceback": traceback_text,
        "git_log_oneline": git_log,
        "ls_la": ls_la,
        "ls_la_pinned_in_tests": ls_la_pinned,
        "seq_output": seq_output,
        "csv_numeric": csv_numeric,
        "file_paths": file_paths,
        "hex_sha256": hex_hashes,
        "uuid": uuids,
        "base64": base64_text,
        "base32": base32_text,
        "dna": lambda i: random_text("ACGT", i, line=60),
        "protein": lambda i: random_text("ACDEFGHIKLMNPQRSTVWY", i, line=60),
        "random_lowercase": lambda i: random_text("abcdefghijklmnopqrstuvwxyz", i),
        "random_uppercase": lambda i: random_text("ABCDEFGHIJKLMNOPQRSTUVWXYZ", i),
        "long_words": lambda i: repeat_shuffled(LONG_WORDS, i),
        **{f"lang_{name}": partial(repeat_shuffled, s) for name, s in LANGUAGES.items()},
        **{name: partial(repeat_shuffled, s, sep="\n") for name, s in SYMBOLS.items()},
    }
    return {name: [make(i) for i in range(SAMPLES)] for name, make in kinds.items()}
