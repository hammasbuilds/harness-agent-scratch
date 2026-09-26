"""The terminal front end: `harness` for a session, `harness -p "..."` for one task."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from .agent import Agent
from .config import Config, load_dotenv
from .llm import LLMError, make_llm
from .sandbox import SandboxUnavailable

RESULT_PREVIEW = 400


class Printer:
    def __init__(self, stream=sys.stdout):
        self.out = stream
        color = stream.isatty() and "NO_COLOR" not in os.environ
        if color and os.name == "nt":
            os.system("")  # turns on ANSI escape handling in the Windows console
        self.c = (lambda code, s: f"\033[{code}m{s}\033[0m") if color else (lambda code, s: s)

    def __call__(self, kind: str, data) -> None:
        c, p = self.c, lambda s: print(s, file=self.out, flush=True)
        if kind == "assistant_text":
            p(data)
        elif kind == "tool_call":
            p(c("36", f"> {data.name} {data.arguments}"))
        elif kind == "subagent_tool_call":
            p(c("36", f"  [subagent] > {data.name} {data.arguments}"))
        elif kind == "tool_result":
            _, result = data
            shown = result if len(result) <= RESULT_PREVIEW else result[:RESULT_PREVIEW] + f" ... ({len(result)} chars)"
            p(c("2", "  " + shown.replace("\n", "\n  ")))
        elif kind == "usage" and data:
            cached = f", {data['cached_tokens']} cached" if data.get("cached_tokens") else ""
            p(c("2", f"  [{data.get('prompt_tokens', 0)} prompt tokens{cached}, {data.get('completion_tokens', 0)} out]"))
        elif kind == "subagent_start":
            p(c("35", "  [subagent started]"))
        elif kind == "subagent_end":
            p(c("35", "  [subagent finished]"))
        elif kind == "compacted":
            before, after = data
            p(c("33", f"  [compacted the transcript: ~{before} -> ~{after} tokens]"))


def ask_user(question: str) -> bool:
    prompt = f"Allow? {question} [y/N] "
    if sys.stdout.isatty() and "NO_COLOR" not in os.environ:
        prompt = f"\033[33m{prompt}\033[0m"
    try:
        return input(prompt).strip().lower() in ("y", "yes")
    except EOFError:
        return False


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(prog="harness", description="A minimal coding agent.")
    ap.add_argument("-p", "--prompt", help="run one task and exit")
    ap.add_argument("-w", "--workspace", default=".", help="folder the agent works in (default: current)")
    ap.add_argument("--backend", choices=["ollama", "openai"])
    ap.add_argument("--model")
    ap.add_argument("--base-url")
    ap.add_argument("--sandbox", choices=["auto", "none", "required"])
    return ap.parse_args(argv)


def build_config(args: argparse.Namespace, env: dict) -> Config:
    workspace = Path(args.workspace).resolve()
    load_dotenv(workspace / ".env", env)
    if args.backend:
        env["HARNESS_BACKEND"] = args.backend
    cfg = Config.from_env(workspace, env)
    cfg.model = args.model or cfg.model
    cfg.base_url = args.base_url or cfg.base_url
    cfg.sandbox = args.sandbox or cfg.sandbox
    return cfg


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        cfg = build_config(args, dict(os.environ))
        printer = Printer()
        agent = Agent(cfg, make_llm(cfg), ask_user, on_event=printer)
    except (ValueError, SandboxUnavailable) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2

    where = f"{cfg.backend} {cfg.base_url}"
    if cfg.backend == "ollama":
        where += f" (num_gpu={cfg.num_gpu}, num_ctx={cfg.num_ctx}, threads={cfg.num_thread})"
    print(f"harness | model {cfg.model} via {where}")
    print(f"workspace {cfg.workspace} | sandbox {agent.sandbox.describe()} | {len(agent.skills)} skills")

    if args.prompt:
        try:
            print(agent.send(args.prompt))
        except LLMError as e:
            print(f"error: {e}", file=sys.stderr)
            return 1
        return 0

    print("Type a task. /todos, /tokens, /clear, /exit.")
    while True:
        try:
            text = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        if not text:
            continue
        if text in ("/exit", "/quit"):
            return 0
        if text == "/todos":
            print(agent.todos.render())
            continue
        if text == "/tokens":
            print(f"~{agent.context_tokens()} of {cfg.context_limit} tokens in context")
            continue
        if text == "/clear":
            agent.reset()
            print("started a fresh conversation")
            continue
        try:
            print("\n" + agent.send(text))
        except LLMError as e:
            print(f"error: {e}", file=sys.stderr)
        except KeyboardInterrupt:
            print("\n[interrupted]")
