"""Measure the harness's token estimate against the real Qwen2.5 tokenizer.

    uv run --group calibrate python scripts/calibrate_tokens.py

Writes results/token_estimate_calibration.json. For every kind of text in
scripts/token_corpus.py: characters, the real token count, the estimate, and
real / estimate. The estimate must never be below the real count (Ollama drops
the start of an overlong prompt without an error), so the number that matters
is the largest ratio; the mean says how much context the pessimism wastes.
The flat "characters / 3" rule the estimator replaced is scored the same way,
as the baseline.

The tokenizer is only a tokenizer (no model, no GPU). It is looked for in the
local Hugging Face cache, then downloaded once into data/ (gitignored) at a
pinned revision and checked against its git blob hash.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]

from harness.compaction import text_tokens  # noqa: E402
from scripts import token_corpus  # noqa: E402

REPO = "Qwen/Qwen2.5-7B-Instruct"
REVISION = "a09a35458c702b33eeacc393d103063234e8bc28"
BLOB_SHA1 = "443909a61d429dff23010e5bddd28ff530edda00"  # git blob hash, as the Hub reports it
SIZE = 7_031_645
CHUNK = 5 * 1024 * 1024
OUT = ROOT / "results" / "token_estimate_calibration.json"
BOOTSTRAP = 10_000


def blob_sha1(data: bytes) -> str:
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


def find_tokenizer() -> Path:
    local = ROOT / "data" / "qwen2.5-tokenizer.json"
    hub = Path.home() / ".cache" / "huggingface" / "hub" / "models--Qwen--Qwen2.5-7B-Instruct"
    for candidate in [local, *sorted(hub.glob("snapshots/*/tokenizer.json"))]:
        if candidate.is_file() and blob_sha1(candidate.read_bytes()) == BLOB_SHA1:
            return candidate
    # Whole-file downloads from the Hub stall on this network; 5 MB ranges do not.
    url = f"https://huggingface.co/{REPO}/resolve/{REVISION}/tokenizer.json"
    parts = []
    for start in range(0, SIZE, CHUNK):
        end = min(start + CHUNK, SIZE) - 1
        req = urllib.request.Request(url, headers={"Range": f"bytes={start}-{end}"})
        with urllib.request.urlopen(req, timeout=300) as r:
            parts.append(r.read())
        print(f"  tokenizer.json: {end + 1:,} of {SIZE:,} bytes", file=sys.stderr)
    data = b"".join(parts)
    if blob_sha1(data) != BLOB_SHA1:
        raise SystemExit("the downloaded tokenizer.json does not match the pinned revision")
    local.parent.mkdir(exist_ok=True)
    local.write_bytes(data)
    return local


def bootstrap_mean_ci(values: list[float], seed: int = 0) -> tuple[float, float]:
    rng = random.Random(seed)
    n = len(values)
    means = sorted(sum(rng.choice(values) for _ in range(n)) / n for _ in range(BOOTSTRAP))
    return means[int(0.025 * BOOTSTRAP)], means[int(0.975 * BOOTSTRAP) - 1]


def summarise(ratios_by_kind: dict[str, list[float]]) -> dict:
    kind_means = [sum(r) / len(r) for r in ratios_by_kind.values()]
    every = [x for r in ratios_by_kind.values() for x in r]
    lo, hi = bootstrap_mean_ci(kind_means)
    worst = max(ratios_by_kind, key=lambda k: max(ratios_by_kind[k]))
    return {
        "max_ratio": round(max(every), 4),
        "max_ratio_kind": worst,
        "mean_ratio_over_kinds": round(sum(kind_means) / len(kind_means), 4),
        "mean_ratio_ci95": [round(lo, 4), round(hi, 4)],
        "samples_underestimated": sum(x > 1 for x in every),
        "kinds_underestimated": sum(max(r) > 1 for r in ratios_by_kind.values()),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()
    try:
        from tokenizers import Tokenizer
    except ImportError:
        print(
            "needs the tokenizers package: uv run --group calibrate python "
            "scripts/calibrate_tokens.py",
            file=sys.stderr,
        )
        return 2
    path = find_tokenizer()
    tok = Tokenizer.from_file(str(path))
    corpus = token_corpus.build()

    kinds, harness, baseline = {}, {}, {}
    for name, samples in corpus.items():
        rows = []
        for text in samples:
            real = len(tok.encode(text, add_special_tokens=False).ids)
            est = text_tokens(text)
            base = math.ceil(len(text) / 3)
            rows.append(
                {
                    "chars": len(text),
                    "real": real,
                    "estimate": est,
                    "ratio": round(real / est, 4),
                    "baseline_chars_div_3": base,
                    "baseline_ratio": round(real / base, 4),
                }
            )
        harness[name] = [r["ratio"] for r in rows]
        baseline[name] = [r["baseline_ratio"] for r in rows]
        kinds[name] = {
            "samples": len(rows),
            "chars": sum(r["chars"] for r in rows) // len(rows),
            "real_tokens_mean": round(sum(r["real"] for r in rows) / len(rows), 1),
            "estimate_mean": round(sum(r["estimate"] for r in rows) / len(rows), 1),
            "ratio_mean": round(sum(harness[name]) / len(rows), 4),
            "ratio_max": max(harness[name]),
            "baseline_ratio_max": max(baseline[name]),
            "per_sample": rows,
        }

    result = {
        "what": "real Qwen2.5 token count / harness estimate (text_tokens) per 3,000-character "
        "sample; above 1.0 means the estimate was too low",
        "tokenizer": {
            "repo": REPO,
            "revision": REVISION,
            "file": "tokenizer.json",
            "git_blob_sha1": BLOB_SHA1,
        },
        "corpus": {
            "kinds": len(corpus),
            "samples_per_kind": token_corpus.SAMPLES,
            "samples": sum(len(s) for s in corpus.values()),
            "chars_per_sample": token_corpus.SIZE,
            "real_text_from_commit": token_corpus.PINNED,
        },
        "bootstrap": {"resampled_unit": "kind", "replicates": BOOTSTRAP, "seed": 0},
        "harness_estimate": summarise(harness),
        "baseline_chars_div_3": summarise(baseline),
        "kinds": kinds,
    }
    args.out.parent.mkdir(exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    h, b = result["harness_estimate"], result["baseline_chars_div_3"]
    print(f"{len(corpus)} kinds x {token_corpus.SAMPLES} samples")
    print(
        f"harness estimate: max real/estimate {h['max_ratio']} ({h['max_ratio_kind']}), mean "
        f"{h['mean_ratio_over_kinds']} CI95 {h['mean_ratio_ci95']}, "
        f"{h['kinds_underestimated']} kinds underestimated"
    )
    print(
        f"chars/3 baseline: max {b['max_ratio']} ({b['max_ratio_kind']}), mean "
        f"{b['mean_ratio_over_kinds']}, {b['kinds_underestimated']} kinds underestimated"
    )
    print(f"wrote {args.out.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
