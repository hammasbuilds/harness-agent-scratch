"""The terminal front end: `harness` for a session, `harness -p "..."` for one task."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from .agent import Agent
from .config import PROJECT_SETTINGS, Config, load_dotenv, user_config_file
from .llm import LLMError, make_llm
from .sandbox import SandboxUnavailable
from .tools import printable

RESULT_PREVIEW = 400

EPILOG = """\
settings, strongest first: these flags, HARNESS_* environment variables,
~/.config/harness/.env (or the file HARNESS_CONFIG names), then the workspace's
own .env, which may set only HARNESS_TEMPERATURE, HARNESS_COMPACT_AT and
HARNESS_COMPACT_TO. See .env.example for every setting.

examples:
  harness                                   a session in the current folder
  harness -p "add a --verbose flag" -w app  one task in ./app, then exit
  harness --num-ctx 4096 --max-steps 20     a smaller context and step budget
"""


def _screen_safe(text: str) -> str:
    """Escape every control character except newlines and tabs."""
    return "\n".join(
        "\t".join(printable(cell) for cell in line.split("\t")) for line in text.split("\n")
    )


def _plain(code: str, s: str) -> str:
    return s


def _ansi(code: str, s: str) -> str:
    return f"\033[{code}m{s}\033[0m"


class Printer:
    def __init__(self, stream=sys.stdout):
        self.out = stream
        color = stream.isatty() and "NO_COLOR" not in os.environ
        if color and os.name == "nt":
            os.system("")  # turns on ANSI escape handling in the Windows console
        self.paint = _ansi if color else _plain

    def _show(self, code: str, text: str) -> None:
        # Model text and file contents may hold escape sequences or \r that would
        # erase or rewrite lines on screen, hiding what happened before a prompt.
        # Cleaned before our own colour codes go around it.
        print(self.paint(code, _screen_safe(text)), file=self.out, flush=True)

    def __call__(self, kind: str, data) -> None:
        if kind == "assistant_text":
            print(_screen_safe(data), file=self.out, flush=True)
        elif kind == "tool_call":
            self._show("36", f"> {data.name} {data.arguments}")
        elif kind == "subagent_tool_call":
            self._show("36", f"  [subagent] > {data.name} {data.arguments}")
        elif kind == "tool_result":
            _, result = data
            shown = result
            if len(result) > RESULT_PREVIEW:
                shown = f"{result[:RESULT_PREVIEW]} ... ({len(result)} chars)"
            self._show("2", "  " + shown.replace("\n", "\n  "))
        elif kind == "usage" and data:
            cached = f", {data['cached_tokens']} cached" if data.get("cached_tokens") else ""
            prompt, out = data.get("prompt_tokens", 0), data.get("completion_tokens", 0)
            self._show("2", f"  [{prompt} prompt tokens{cached}, {out} out]")
        elif kind == "subagent_start":
            self._show("35", "  [subagent started]")
        elif kind == "subagent_end":
            self._show("35", "  [subagent finished]")
        elif kind == "compacted":
            before, after = data
            self._show("33", f"  [compacted the transcript: ~{before} -> ~{after} tokens]")
        elif kind == "squeezed":
            tokens, budget = data
            self._show(
                "33", f"  [cut this turn's tool outputs to fit: ~{tokens} tokens, budget {budget}]"
            )


def ask_user(question: str) -> bool:
    prompt = f"Allow? {question} [y/N] "
    if sys.stdout.isatty() and "NO_COLOR" not in os.environ:
        prompt = f"\033[33m{prompt}\033[0m"
    try:
        return input(prompt).strip().lower() in ("y", "yes")
    except EOFError:
        return False


def _positive(text: str) -> int:
    try:
        value = int(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"not a whole number: {text!r}") from None
    if value < 1:
        raise argparse.ArgumentTypeError(f"must be at least 1, got {value}")
    return value


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        prog="harness",
        description="A minimal coding agent: a model called in a loop, with tools, "
        "permissions and compaction. With no -p it opens an interactive session.",
        epilog=EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("-p", "--prompt", metavar="TASK", help="run one task, print the answer, exit")
    ap.add_argument(
        "-w",
        "--workspace",
        default=".",
        metavar="DIR",
        help="existing folder the agent works in (default: the current one)",
    )
    ap.add_argument(
        "--backend",
        choices=["ollama", "openai"],
        help="ollama: native /api/chat (default); openai: any /chat/completions API",
    )
    ap.add_argument("--model", help="model name (default: qwen2.5:7b-instruct, or HARNESS_MODEL)")
    ap.add_argument(
        "--base-url",
        metavar="URL",
        help="server address (default: http://localhost:11434 for ollama, "
        "https://openrouter.ai/api/v1 for openai)",
    )
    ap.add_argument(
        "--sandbox",
        choices=["auto", "none", "required"],
        help="auto: bubblewrap/Seatbelt when present (default); none: never; "
        "required: refuse to start without one",
    )
    ap.add_argument(
        "--num-ctx",
        type=_positive,
        metavar="TOKENS",
        help="Ollama's context window, also where compaction aims (default 8192)",
    )
    ap.add_argument(
        "--max-steps",
        type=_positive,
        metavar="N",
        help="model calls allowed per task before stopping (default 40)",
    )
    return ap.parse_args(argv)


def build_config(
    args: argparse.Namespace, env: dict, warn=lambda msg: print(msg, file=sys.stderr)
) -> Config:
    """Settings, strongest first: command-line flags, the real environment, the
    user's own config file, then the workspace .env, which may set only
    PROJECT_SETTINGS (harmless tuning). A workspace that is not an existing
    folder is an error here, before any model is contacted."""
    workspace = Path(args.workspace).expanduser().resolve()
    if not workspace.exists():
        raise ValueError(f"workspace {workspace} does not exist")
    if not workspace.is_dir():
        raise ValueError(f"workspace {workspace} is not a folder")
    load_dotenv(user_config_file(env), env)
    ignored = load_dotenv(workspace / ".env", env, only=PROJECT_SETTINGS)
    if ignored:
        warn(
            f"warning: ignored {', '.join(ignored)} from {workspace / '.env'}; a project's .env "
            f"may set only {', '.join(sorted(PROJECT_SETTINGS))}. Put the rest in "
            f"{user_config_file(env)} or the environment."
        )
    # Flags that decide validated sizes go in before the settings are built.
    for flag, key in (("backend", "BACKEND"), ("num_ctx", "NUM_CTX"), ("max_steps", "MAX_STEPS")):
        if getattr(args, flag, None) is not None:
            env[f"HARNESS_{key}"] = str(getattr(args, flag))
    cfg = Config.from_env(workspace, env)
    cfg.model = args.model or cfg.model
    cfg.base_url = args.base_url or cfg.base_url
    cfg.sandbox = args.sandbox or cfg.sandbox
    return cfg


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    # Redirected output on Windows is cp1252: one "✓" in a tool result would
    # raise UnicodeEncodeError and end the session.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(errors="replace")
    if args.prompt is not None and not args.prompt.strip():
        print("error: -p needs a task; leave -p out for an interactive session", file=sys.stderr)
        return 2
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
    skills = len(agent.skills)
    print(f"workspace {cfg.workspace} | sandbox {agent.sandbox.describe()} | {skills} skills")

    if args.prompt is not None:
        try:
            print(_screen_safe(agent.send(args.prompt)))
        except LLMError as e:
            print(_screen_safe(f"error: {e}"), file=sys.stderr)
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
            print(_screen_safe(agent.todos.render()))
            continue
        if text == "/tokens":
            print(f"~{agent.context_tokens()} of {cfg.context_limit} tokens in context")
            continue
        if text == "/clear":
            agent.reset()
            print("started a fresh conversation")
            continue
        try:
            print("\n" + _screen_safe(agent.send(text)))
        except LLMError as e:
            print(_screen_safe(f"error: {e}"), file=sys.stderr)
        except KeyboardInterrupt:
            print("\n[interrupted]")
