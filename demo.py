"""Run the agent on one small task in a throwaway folder.

    python demo.py             # a real model (Ollama on the CPU by default; see .env.example)
    python demo.py --scripted  # no model at all: pre-written replies, real tools

The scripted mode shows the harness mechanics (todos, writing, running,
editing, a subagent, the permission prompt) on a machine with no model
available. Its replies are written by hand, so it says nothing about how well
any model does the task.
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import tempfile
from pathlib import Path

from harness.agent import Agent
from harness.cli import Printer
from harness.config import Config, load_dotenv
from harness.llm import Reply, ScriptedLLM, call, make_llm

TASK = (
    "Create fib.py that prints the first 10 Fibonacci numbers on one line, run it, "
    "then change it to print 15 and run it again. Keep a todo list."
)

FIB = (
    "a, b = 0, 1\nout = []\nfor _ in range(10):\n"
    "    out.append(a)\n    a, b = b, a + b\nprint(*out)\n"
)


def scripted_model() -> ScriptedLLM:
    def todo(*states: str):
        return call(
            "write_todos",
            todos=[
                {"content": "write fib.py", "status": states[0]},
                {"content": "run it", "status": states[1]},
                {"content": "change to 15 and rerun", "status": states[2]},
            ],
        )

    return ScriptedLLM(
        [
            Reply("", [todo("in_progress", "pending", "pending")]),
            Reply("", [call("write_file", path="fib.py", content=FIB)]),
            Reply(
                "",
                [
                    todo("completed", "in_progress", "pending"),
                    call("bash", command="python fib.py"),
                ],
            ),
            Reply(
                "",
                [
                    todo("completed", "completed", "in_progress"),
                    call(
                        "str_replace", path="fib.py", old_string="range(10)", new_string="range(15)"
                    ),
                ],
            ),
            Reply("", [call("bash", command="python fib.py")]),
            Reply(
                "",
                [
                    call(
                        "task",
                        prompt="Read fib.py in the workspace and say in one sentence "
                        "what it prints.",
                    )
                ],
            ),
            Reply("", [call("read_file", path="fib.py")]),  # the subagent
            Reply("fib.py prints the first 15 Fibonacci numbers, space-separated, on one line."),
            Reply("", [todo("completed", "completed", "completed")]),
            Reply(
                "Created fib.py, ran it (0 1 1 2 3 5 8 13 21 34), changed range(10) to "
                "range(15) and ran it again (... 144 233 377). All three todos are done."
            ),
        ]
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--scripted", action="store_true", help="use pre-written replies instead of a model"
    )
    ap.add_argument("--keep", action="store_true", help="keep the throwaway workspace afterwards")
    args = ap.parse_args()

    workspace = Path(tempfile.mkdtemp(prefix="harness-demo-")).resolve()
    try:
        return run(args, workspace)
    finally:
        if args.keep:
            print(f"\nworkspace kept at {workspace}")
        else:
            shutil.rmtree(workspace, ignore_errors=True)


def run(args, workspace: Path) -> int:
    env = dict(os.environ)
    load_dotenv(Path(__file__).parent / ".env", env)
    cfg = Config.from_env(workspace, env)
    cfg.skills_dirs = []
    llm = scripted_model() if args.scripted else make_llm(cfg)

    def approve(question: str) -> bool:
        print(f"  [permission] {question} -> approved by demo.py")
        return True

    agent = Agent(cfg, llm, approve, on_event=Printer())
    source = "scripted replies (no model)" if args.scripted else f"{cfg.model} via {cfg.backend}"
    print(f"model: {source}\nworkspace: {workspace}\nsandbox: {agent.sandbox.describe()}\n")
    print(f"task: {TASK}\n")
    answer = agent.send(TASK)
    print(f"\nfinal answer:\n{answer}\n")
    print(f"todos:\n{agent.todos.render()}")
    fib = workspace / "fib.py"
    print(f"\nfib.py on disk:\n{fib.read_text() if fib.exists() else '(not created)'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
