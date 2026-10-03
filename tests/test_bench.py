"""The model arm, driven by a deterministic fake Ollama. No model, no network."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from harness import bench
from harness.bench_tasks import TASKS, apply_solution, setup
from harness.config import Config
from harness.llm import LLMError, OllamaBackend
from harness.sandbox import find_shell

ROOT = Path(__file__).resolve().parent.parent
# Tasks whose checks start no Python process: the cache and resume tests run the
# suite several times, and a process start costs a second or two on a busy box.
QUICK = tuple(
    t
    for t in TASKS
    if t.name
    in {
        "json_config_edit",
        "crlf_edit",
        "edit_line_350",
        "three_files",
        "find_the_raiser",
        "count_rows",
    }
)


class FakeOllama:
    """Answers /api/chat like Ollama would, by replaying each task's reference
    solution as tool calls: read_file then str_replace for an edit, write_file
    for a new file, then the final answer. prompt_eval_count imitates Ollama's
    prefix cache: only the part of the prompt after the longest prefix shared
    with the previous request is counted as evaluated."""

    def __init__(self, solve: bool = True):
        self.solve = solve
        self.posts = 0
        self.previous = ""

    def plan(self, prompt: str) -> list[dict]:
        task = next(t for t in TASKS if t.prompt == prompt)
        steps = []
        for e in task.solution if self.solve else ():
            if e.old is None:
                steps.append(
                    {"name": "write_file", "arguments": {"path": e.path, "content": e.new}}
                )
            else:
                steps.append({"name": "read_file", "arguments": {"path": e.path}})
                steps.append(
                    {
                        "name": "str_replace",
                        "arguments": {"path": e.path, "old_string": e.old, "new_string": e.new},
                    }
                )
        return steps + [{"final": (task.answer or "done") if self.solve else "I cannot."}]

    def __call__(self, url, body, headers, timeout):
        self.posts += 1
        assert url.endswith("/api/chat") and body["options"]["temperature"] == 0.0
        messages = body["messages"]
        user = next(m["content"] for m in messages if m["role"] == "user")
        done = sum(1 for m in messages if m["role"] == "assistant")
        step = self.plan(user)[done]
        text = json.dumps(messages, sort_keys=True)
        shared = len(os.path.commonprefix([text, self.previous]))
        self.previous = text
        usage = {"prompt_eval_count": max(1, (len(text) - shared) // 4), "eval_count": 20}
        if "final" in step:
            return {"message": {"role": "assistant", "content": step["final"]}, **usage}
        return {
            "message": {"role": "assistant", "content": "", "tool_calls": [{"function": step}]},
            **usage,
        }


def fake_client(fake: FakeOllama, tmp_path: Path) -> OllamaBackend:
    return OllamaBackend(Config(workspace=tmp_path, temperature=0.0), post=fake)


def cfg_for(ws: Path) -> Config:
    return Config(
        workspace=ws,
        temperature=0.0,
        skills_dirs=[],
        sandbox="none",
        max_steps=12,
        max_output_tokens=512,
    )


def run(tmp_path: Path, fake: FakeOllama, tasks=TASKS, model: str = "fake:1b") -> dict:
    return bench.run_suite(
        model,
        cfg_for,
        fake_client(fake, tmp_path),
        tasks,
        tmp_path / "results",
        tmp_path / "cache",
        tmp_path / "runs",
        log=lambda line: None,
    )


@pytest.mark.parametrize("task", TASKS, ids=lambda t: t.name)
def test_every_check_fails_on_the_start_and_passes_on_the_reference(task, tmp_path):
    setup(task, tmp_path)
    assert task.check(tmp_path, "")[0] is False
    answer = apply_solution(task, tmp_path)
    ok, detail = task.check(tmp_path, answer)
    assert ok, detail


def test_the_suite_has_the_intended_size_and_unique_names():
    assert 10 <= len(TASKS) <= 20
    assert len({t.name for t in TASKS}) == len(TASKS)


def test_a_model_that_solves_everything_passes_through_the_real_harness(tmp_path):
    fake = FakeOllama()
    result = run(tmp_path, fake)
    failed = [(r["task"], r["check"]) for r in result["rows"] if not r["passed"]]
    assert failed == []
    assert result["pass_rate"] == 1.0 and result["tasks"] == len(TASKS)
    assert result["model_calls"] == fake.posts  # every call went to the model once
    for row in result["rows"]:
        assert row["steps"] >= 1 and row["wall_seconds"] >= 0
    # The fake re-evaluates only what changed after the shared prefix: reuse is
    # real and below 1 (each step adds new text).
    reuse = [r["prompt_cache_reuse"] for r in result["rows"] if r["prompt_cache_reuse"] is not None]
    assert reuse and all(0 < x < 1 for x in reuse)
    saved = json.loads((tmp_path / "results" / "fake_1b.json").read_text(encoding="utf-8"))
    assert saved["passed"] == len(TASKS) and saved["pass_rate_ci95"][1] == 1.0


def test_a_rerun_is_served_entirely_from_the_generation_cache(tmp_path):
    first = run(tmp_path, FakeOllama(), tasks=QUICK)
    second_fake = FakeOllama()
    second = run(tmp_path, second_fake, tasks=QUICK)
    assert second_fake.posts == 0
    assert second["generation_cache_hit_rate"] == 1.0
    assert [r["passed"] for r in second["rows"]] == [r["passed"] for r in first["rows"]]
    assert [r["steps"] for r in second["rows"]] == [r["steps"] for r in first["rows"]]


def test_an_interrupted_run_resumes_where_it_stopped(tmp_path):
    run(tmp_path, FakeOllama(), tasks=QUICK[:3])
    fake = FakeOllama()
    result = run(tmp_path, fake, tasks=QUICK)
    hits = [r["generation_cache_hits"] for r in result["rows"]]
    assert all(h == r["steps"] for h, r in zip(hits[:3], result["rows"][:3], strict=True))
    assert all(h == 0 for h in hits[3:]) and fake.posts == sum(
        r["steps"] for r in result["rows"][3:]
    )


def test_a_model_that_does_nothing_fails_the_checks(tmp_path):
    result = run(tmp_path, FakeOllama(solve=False), tasks=QUICK)
    assert result["passed"] == 0 and result["pass_rate_ci95"][0] == 0.0


def test_a_model_error_is_recorded_as_a_failure(tmp_path):
    def broken(url, body, headers, timeout):
        raise LLMError("HTTP 500 from fake")

    client = OllamaBackend(Config(workspace=tmp_path, retries=0), post=broken)
    result = bench.run_suite(
        "fake:1b",
        cfg_for,
        client,
        TASKS[:1],
        tmp_path / "r",
        tmp_path / "c",
        tmp_path / "w",
        log=lambda line: None,
    )
    row = result["rows"][0]
    assert row["passed"] is False and "HTTP 500" in row["error"]


def test_the_cache_key_covers_model_prompt_and_options():
    msgs = [{"role": "user", "content": "hi"}]
    base = bench.request_key("m", msgs, None, {"temperature": 0.0})
    assert base == bench.request_key("m", [dict(msgs[0])], None, {"temperature": 0.0})
    assert base != bench.request_key("m2", msgs, None, {"temperature": 0.0})
    assert base != bench.request_key(
        "m", [{"role": "user", "content": "hi!"}], None, {"temperature": 0.0}
    )
    assert base != bench.request_key("m", msgs, None, {"temperature": 0.2})
    assert base != bench.request_key("m", msgs, [{"type": "function"}], {"temperature": 0.0})


def test_prompt_cache_reuse_arithmetic():
    rec = bench.CallRecord
    # The first call calibrates: 500 evaluated for an estimate of 1000 -> ratio 0.5.
    calls = [
        rec(1000, 500, 10, 1.0, False),
        rec(1200, 100, 10, 1.0, False),
        rec(1400, 100, 10, 1.0, False),
    ]
    # Later calls' full lengths: 600 + 700 = 1300; evaluated 200 -> reuse 1 - 200/1300.
    assert bench.prompt_cache_reuse(calls) == pytest.approx(1 - 200 / 1300)
    assert bench.prompt_cache_reuse(calls[:1]) is None
    assert bench.prompt_cache_reuse([rec(1000, None, 1, 1.0, False)] * 2) is None


def test_dry_run_lists_every_job_and_the_call_count(capsys):
    assert bench.main(["--dry-run"]) == 0
    out = capsys.readouterr().out
    jobs = [line for line in out.splitlines() if line.startswith("job ")]
    assert len(jobs) == len(bench.MODELS) * len(TASKS)
    assert f"about {len(jobs) * bench.TYPICAL_CALLS_PER_TASK} model calls" in out


def test_unknown_task_names_are_refused():
    with pytest.raises(SystemExit, match="unknown task"):
        bench.main(["--dry-run", "--tasks", "create_fib,nope"])


def _bash() -> str | None:
    argv = find_shell()[0]
    return argv[0] if argv[0].lower().endswith(("bash", "bash.exe")) else None


@pytest.mark.skipif(_bash() is None, reason="needs bash")
def test_run_models_sh_dry_run_runs_no_model():
    env = {**os.environ, "PYTHON": sys.executable, "MODELS": "a:1b b:2b"}
    out = subprocess.run(
        [_bash(), "scripts/run_models.sh", "--dry-run"],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
        timeout=120,
    )
    assert out.returncode == 0, out.stderr
    assert f"{2 * len(TASKS)} jobs (2 models x {len(TASKS)} tasks)" in out.stdout


@pytest.mark.skipif(_bash() is None, reason="needs bash")
def test_run_models_sh_rejects_unknown_arguments():
    out = subprocess.run(
        [_bash(), "scripts/run_models.sh", "--go"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
        timeout=60,
    )
    assert out.returncode == 2 and "usage" in out.stderr
