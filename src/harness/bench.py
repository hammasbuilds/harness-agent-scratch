"""The model arm: run the task suite against a real model and write the result.

    python -m harness.bench --model qwen2.5:14b-instruct --num-gpu 99
    python -m harness.bench --dry-run            # the job list and call count only

Each task runs in its own fresh workspace under runs/<model>/<task>/, with every
permission prompt approved (the workspaces are throwaway) and a fixed date, so
that a rerun sends byte-identical requests. Every generation is cached on disk,
keyed by (model, a hash of the request, the generation options); an interrupted
run resumes from the cache instead of paying for its calls again.

Per task the result records pass/fail from the task's own check, model calls
(steps), wall time, model time, and how much of each prompt Ollama's prefix
cache let it skip. Ollama reports prompt_eval_count, the prompt tokens it had
to evaluate, not the prompt's length; the first call of a task has nothing
cached beyond a few words of the system prompt (the workspace path differs per
task), so its evaluated count over the harness's estimate gives this model's
real-to-estimate ratio, and each later call's full length is its estimate
times that ratio. Reuse = 1 - evaluated / full, over the calls after the first.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import shutil
import statistics
import sys
import time
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, field
from datetime import date
from pathlib import Path

from .agent import Agent
from .bench_tasks import TASKS, Task, setup
from .compaction import estimate_tokens
from .config import Config
from .llm import LLM, LLMError, Reply, ToolCall, extract_inline_tool_calls, make_llm
from .sandbox import Sandbox

MODELS = ("qwen2.5:7b-instruct", "qwen2.5:14b-instruct", "qwen2.5-coder:14b")
FIXED_DATE = date(2026, 9, 28)
TYPICAL_CALLS_PER_TASK = 6  # read, edit, run, answer, and a retry or two
CACHE_VERSION = 1


def slug(model: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", model)


def generation_options(cfg: Config) -> dict:
    """What, besides the messages and tools, changes a generation."""
    return {
        "backend": cfg.backend,
        "temperature": cfg.temperature,
        "num_ctx": cfg.num_ctx,
        "max_output_tokens": cfg.max_output_tokens,
    }


def request_key(model: str, messages: list[dict], tools: list[dict] | None, options: dict) -> str:
    blob = json.dumps(
        {
            "v": CACHE_VERSION,
            "model": model,
            "options": options,
            "messages": messages,
            "tools": tools,
        },
        sort_keys=True,
        ensure_ascii=False,
    )
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def _reply_to_json(reply: Reply) -> dict:
    return {
        "content": reply.content,
        "tool_calls": [asdict(c) for c in reply.tool_calls],
        "usage": reply.usage,
        "truncated": reply.truncated,
    }


def _reply_from_json(data: dict) -> Reply:
    return Reply(
        content=data["content"],
        tool_calls=[ToolCall(**c) for c in data["tool_calls"]],
        usage=data["usage"],
        truncated=data["truncated"],
    )


@dataclass
class CallRecord:
    estimated_prompt_tokens: int
    prompt_eval_count: int | None
    completion_tokens: int | None
    seconds: float
    from_cache: bool


@dataclass
class CachedLLM:
    """Wraps a model client; stores every reply on disk under its request key."""

    inner: LLM
    cache_dir: Path
    model: str
    options: dict
    calls: list[CallRecord] = field(default_factory=list)

    def chat(self, messages: list[dict], tools: list[dict] | None) -> Reply:
        key = request_key(self.model, messages, tools, self.options)
        path = self.cache_dir / key[:2] / f"{key}.json"
        estimate = estimate_tokens(messages) + (estimate_tokens(tools) if tools else 0)
        if path.is_file():
            stored = json.loads(path.read_text(encoding="utf-8"))
            reply, seconds, hit = _reply_from_json(stored["reply"]), stored["seconds"], True
            # The cache holds replies as the backend parsed them at the time. Re-run
            # the text fallback so a parser fix also reaches replies cached before it.
            if not reply.tool_calls and tools:
                names = {t["function"]["name"] for t in tools}
                calls = extract_inline_tool_calls(reply.content, names)
                if calls:
                    for j, c in enumerate(calls):
                        c.id = f"call_{len(self.calls)}_{j}"
                    reply = Reply(
                        content="", tool_calls=calls, usage=reply.usage, truncated=reply.truncated
                    )
        else:
            start = time.monotonic()
            reply = self.inner.chat(messages, tools)
            seconds, hit = time.monotonic() - start, False
            # Call ids come from a per-client counter, so a resumed run would number
            # them differently and could repeat one already in the transcript.
            # Numbered by position instead, a rerun sends the same bytes.
            for j, c in enumerate(reply.tool_calls):
                c.id = f"call_{len(self.calls)}_{j}"
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(".tmp")
            tmp.write_text(
                json.dumps(
                    {"reply": _reply_to_json(reply), "seconds": seconds, "model": self.model},
                    ensure_ascii=False,
                ),
                "utf-8",
            )
            tmp.replace(path)  # a run killed mid-write leaves no half entry behind
        self.calls.append(
            CallRecord(
                estimate,
                reply.usage.get("prompt_tokens"),
                reply.usage.get("completion_tokens"),
                seconds,
                hit,
            )
        )
        return reply


def prompt_cache_reuse(calls: Sequence[CallRecord]) -> float | None:
    """Share of prompt tokens after the first call that Ollama did not re-evaluate."""
    if len(calls) < 2 or not calls[0].prompt_eval_count or not calls[0].estimated_prompt_tokens:
        return None
    if any(c.prompt_eval_count is None for c in calls):
        return None
    ratio = calls[0].prompt_eval_count / calls[0].estimated_prompt_tokens
    full = sum(c.estimated_prompt_tokens * ratio for c in calls[1:])
    evaluated = sum(c.prompt_eval_count for c in calls[1:])
    return max(0.0, 1 - evaluated / full) if full else None


def wilson(k: int, n: int, z: float = 1.96) -> list[float]:
    if n == 0:
        return [0.0, 0.0]
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return [round(max(0.0, centre - half), 4), round(min(1.0, centre + half), 4)]


def run_task(
    task: Task,
    cfg_for: Callable[[Path], Config],
    client: LLM,
    cache_dir: Path,
    model: str,
    runs_dir: Path,
) -> dict:
    ws = (runs_dir / task.name / "workspace").resolve()
    shutil.rmtree(ws, ignore_errors=True)
    ws.mkdir(parents=True)
    setup(task, ws)
    cfg = cfg_for(ws)
    llm = CachedLLM(client, cache_dir, model, generation_options(cfg))
    agent = Agent(
        cfg, llm, lambda question: True, today=lambda: FIXED_DATE, sandbox=Sandbox(ws, cfg.sandbox)
    )
    start = time.monotonic()
    error = None
    try:
        answer = agent.send(task.prompt)
    except LLMError as e:
        answer, error = "", str(e)
    wall = time.monotonic() - start
    passed, detail = task.check(ws, answer)
    return {
        "task": task.name,
        "tags": list(task.tags),
        "passed": bool(passed and error is None),
        "check": detail,
        "error": error,
        "steps": len(llm.calls),
        "wall_seconds": round(wall, 2),
        "model_seconds": round(sum(c.seconds for c in llm.calls), 2),
        "generation_cache_hits": sum(c.from_cache for c in llm.calls),
        "prompt_eval_tokens": sum(c.prompt_eval_count or 0 for c in llm.calls),
        "estimated_prompt_tokens": sum(c.estimated_prompt_tokens for c in llm.calls),
        "prompt_cache_reuse": prompt_cache_reuse(llm.calls),
        "answer": answer[-500:],
    }


def summarise(model: str, options: dict, rows: list[dict]) -> dict:
    n, k = len(rows), sum(r["passed"] for r in rows)
    reuse = [r["prompt_cache_reuse"] for r in rows if r["prompt_cache_reuse"] is not None]
    calls = sum(r["steps"] for r in rows)
    return {
        "model": model,
        "options": options,
        "tasks": n,
        "passed": k,
        "pass_rate": round(k / n, 4) if n else None,
        "pass_rate_ci95": wilson(k, n),
        "mean_steps": round(statistics.mean(r["steps"] for r in rows), 2) if rows else None,
        "median_wall_seconds": round(statistics.median(r["wall_seconds"] for r in rows), 2)
        if rows
        else None,
        "total_model_seconds": round(sum(r["model_seconds"] for r in rows), 1),
        "model_calls": calls,
        "generation_cache_hit_rate": round(sum(r["generation_cache_hits"] for r in rows) / calls, 4)
        if calls
        else None,
        "median_prompt_cache_reuse": round(statistics.median(reuse), 4) if reuse else None,
        "rows": rows,
    }


def run_suite(
    model: str,
    cfg_for: Callable[[Path], Config],
    client: LLM,
    tasks: Sequence[Task],
    out_dir: Path,
    cache_dir: Path,
    runs_dir: Path,
    log: Callable[[str], None] = print,
) -> dict:
    rows = []
    for i, task in enumerate(tasks, 1):
        row = run_task(task, cfg_for, client, cache_dir, model, runs_dir / slug(model))
        verdict = "PASS" if row["passed"] else "FAIL"
        log(
            f"[{i}/{len(tasks)}] {task.name}: {verdict} in {row['steps']} calls, "
            f"{row['wall_seconds']}s ({row['generation_cache_hits']} cached)"
        )
        rows.append(row)
    options = generation_options(cfg_for(runs_dir))
    result = summarise(model, options, rows)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{slug(model)}.json"
    path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    log(f"{model}: {result['passed']}/{result['tasks']} passed; wrote {path}")
    return result


def plan(models: Sequence[str], tasks: Sequence[Task], max_steps: int) -> dict:
    jobs = [(m, t.name) for m in models for t in tasks]
    return {
        "jobs": jobs,
        "min_calls": len(jobs),
        "typical_calls": len(jobs) * TYPICAL_CALLS_PER_TASK,
        # The main loop's cap; compaction and subagent calls come on top of it.
        "main_loop_cap": len(jobs) * max_steps,
    }


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        prog="python -m harness.bench", description=__doc__.split("\n\n")[0]
    )
    ap.add_argument("--model", action="append", help=f"repeatable (default: {', '.join(MODELS)})")
    ap.add_argument("--tasks", help="comma-separated task names (default: all)")
    ap.add_argument("--num-gpu", type=int, default=0, help="Ollama layers on the GPU (99 = all)")
    ap.add_argument("--num-ctx", type=int, default=8192)
    ap.add_argument("--max-steps", type=int, default=20)
    ap.add_argument("--base-url", default="http://127.0.0.1:11434")
    ap.add_argument("--out", type=Path, default=Path("results") / "model_runs")
    ap.add_argument("--cache", type=Path, default=Path(".bench-cache"))
    ap.add_argument("--runs", type=Path, default=Path("runs"))
    ap.add_argument(
        "--dry-run", action="store_true", help="print the jobs and call count, run nothing"
    )
    return ap.parse_args(argv)


def select(names: str | None) -> list[Task]:
    if not names:
        return list(TASKS)
    known = {t.name: t for t in TASKS}
    unknown = [n for n in names.split(",") if n not in known]
    if unknown:
        raise SystemExit(f"unknown task(s): {', '.join(unknown)}; known: {', '.join(known)}")
    return [known[n] for n in names.split(",")]


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    models = args.model or list(MODELS)
    tasks = select(args.tasks)
    p = plan(models, tasks, args.max_steps)
    if args.dry_run:
        for model, task in p["jobs"]:
            print(f"job  {model:24s} {task}")
        print(
            f"{len(p['jobs'])} jobs ({len(models)} models x {len(tasks)} tasks); about "
            f"{p['typical_calls']} model calls (at least {p['min_calls']}; the main loop "
            f"stops at {p['main_loop_cap']}, plus any compaction and subagent calls)"
        )
        return 0

    def cfg_for(ws: Path, model: str) -> Config:
        return Config(
            workspace=ws,
            model=model,
            base_url=args.base_url,
            num_gpu=args.num_gpu,
            num_ctx=args.num_ctx,
            context_limit=args.num_ctx,
            temperature=0.0,
            max_steps=args.max_steps,
            subagent_max_steps=args.max_steps,
            max_output_tokens=min(2048, args.num_ctx // 4),
            skills_dirs=[],
            sandbox="auto",
        )

    for model in models:
        client = make_llm(cfg_for(Path.cwd(), model))
        run_suite(
            model,
            lambda ws, m=model: cfg_for(ws, m),
            client,
            tasks,
            args.out,
            args.cache,
            args.runs,
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
