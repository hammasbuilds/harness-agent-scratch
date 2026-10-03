"""Four harness behaviours, each run for real with scripted model replies.

    python examples/scenarios.py               # all four
    python examples/scenarios.py compaction    # one of: refusal compaction squeeze subagent

The harness, the tools, the permission check, compaction and the subagent all
really run; only the model's replies are written in advance (no inference). The
README's Input / Output section quotes this script's output.
"""

from __future__ import annotations

import shutil
import sys
import tempfile
from collections.abc import Callable
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]

from harness.agent import Agent  # noqa: E402
from harness.cli import Printer  # noqa: E402
from harness.config import Config  # noqa: E402
from harness.llm import Reply, call  # noqa: E402
from harness.sandbox import Sandbox  # noqa: E402

SUMMARY = (
    "Goal: collect the report parts the user pastes, one per message. Done: parts "
    "received so far were noted, all regions present. Next: note each new part."
)


class ScenarioLLM:
    """Replays replies in order; answers a compaction request with a fixed note."""

    def __init__(self, replies: list[Reply]):
        self.replies = list(replies)

    def chat(self, messages: list[dict], tools: list[dict] | None) -> Reply:
        if messages[0]["content"].startswith("You are compacting"):
            return Reply(SUMMARY)
        return self.replies.pop(0)


def _agent(ws: Path, replies: list[Reply], approve: Callable[[str], bool], **cfg) -> Agent:
    config = Config(workspace=ws, skills_dirs=[], sandbox="none", **cfg)
    return Agent(
        config,
        ScenarioLLM(replies),
        approve,
        on_event=Printer(),
        today=lambda: date(2026, 9, 28),
        sandbox=Sandbox(ws, "none"),
    )


def _say(question: str, answer: bool) -> bool:
    print(f"  Allow? {question} [y/N] {'y' if answer else 'n'}   (typed by this script)")
    return answer


def refusal(ws: Path) -> None:
    (ws / "notes.txt").write_text("keep me\n")
    agent = _agent(
        ws,
        [
            Reply("", [call("bash", command="rm notes.txt")]),
            Reply("You declined, so notes.txt is still there. Delete it yourself if you want."),
        ],
        lambda q: _say(q, False),
    )
    print("> task: delete notes.txt")
    print(agent.send("delete notes.txt"))
    print(f"notes.txt still on disk: {(ws / 'notes.txt').exists()}")


def compaction(ws: Path) -> None:
    replies = [Reply(f"Noted part {n}: {40 * n} rows, all regions present.") for n in (1, 2, 3, 4)]
    agent = _agent(
        ws, replies, lambda q: True, num_ctx=4096, context_limit=4096, max_output_tokens=512
    )
    print(
        f"fixed overhead (system prompt, tool schemas, reminder): ~{agent.overhead_tokens()} "
        f"of {agent.cfg.context_limit} tokens"
    )
    for n in (1, 2, 3, 4):
        pasted = "".join(f"region-{n}-{i},{i * 37 % 1000}\n" for i in range(60))
        print(f"> task: here is part {n} of the report ({len(pasted)} characters pasted)")
        print(agent.send(f"Here is part {n} of the report:\n{pasted}"))
        print(f"  [~{agent.context_tokens()} tokens in context]")
    print(f"\nhandoff note now in the system prompt:\n{agent.summary}")


def squeeze(ws: Path) -> None:
    for n in (1, 2, 3):
        (ws / f"log{n}.txt").write_text(
            "".join(f"{n}:{i} 200 GET /api/items/{i * 7}\n" for i in range(300))
        )
    agent = _agent(
        ws,
        [
            Reply("", [call("read_file", path="log1.txt")]),
            Reply("", [call("read_file", path="log2.txt")]),
            Reply("", [call("read_file", path="log3.txt")]),
            Reply("All three logs show only 200 responses."),
        ],
        lambda q: True,
        num_ctx=4096,
        context_limit=4096,
        max_output_tokens=512,
    )
    print("> task: check the three logs for errors")
    print(agent.send("check the three logs for errors"))


def subagent(ws: Path) -> None:
    (ws / "README.md").write_text("# svc\nA small service.\n")
    agent = _agent(
        ws,
        [
            Reply(
                "",
                [call("task", prompt="What is this project? Tidy README.md while you are there.")],
            ),
            Reply("", [call("write_file", path="README.md", content="# svc\n")]),  # the subagent
            Reply("", [call("bash", command="rm README.md")]),  # the subagent
            Reply("", [call("read_file", path="README.md")]),  # the subagent
            Reply("It is 'svc', a small service. I could not tidy README.md: I can only read."),
            Reply("The subagent says this is svc, a small service; it was not allowed to edit."),
        ],
        lambda q: _say(q, True),
    )
    print("> task: explore this project")
    print(agent.send("explore this project"))
    unchanged = (ws / "README.md").read_text() == "# svc\nA small service.\n"
    print(f"README.md unchanged: {unchanged}")


SCENARIOS = {"refusal": refusal, "compaction": compaction, "squeeze": squeeze, "subagent": subagent}


def main(argv: list[str]) -> int:
    names = argv or list(SCENARIOS)
    unknown = [n for n in names if n not in SCENARIOS]
    if unknown:
        print(f"unknown scenario(s) {unknown}; choose from {list(SCENARIOS)}", file=sys.stderr)
        return 2
    for name in names:
        print(f"===== {name} (scripted model replies; everything else really runs) =====")
        ws = Path(tempfile.mkdtemp(prefix=f"harness-{name}-")).resolve()
        try:
            SCENARIOS[name](ws)
        finally:
            shutil.rmtree(ws, ignore_errors=True)
        print()
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
