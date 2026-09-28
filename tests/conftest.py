import os
from datetime import date
from pathlib import Path

import pytest

from harness.agent import Agent
from harness.config import Config
from harness.sandbox import Sandbox


@pytest.fixture(autouse=True)
def no_harness_env(monkeypatch):
    """A HARNESS_* variable in the shell running the suite (a model, a context
    size, a base URL) must not change what the tests see."""
    for key in [k for k in os.environ if k.upper().startswith("HARNESS_")]:
        monkeypatch.delenv(key, raising=False)


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    ws = tmp_path / "ws"
    ws.mkdir()
    return ws.resolve()


@pytest.fixture
def cfg(workspace: Path) -> Config:
    # skills_dirs=[] keeps the real ~/.agents/skills out of every test. Tests
    # shrink context_limit to a few thousand tokens to force compaction, so the
    # reply reservation is scaled down with it.
    return Config(workspace=workspace, skills_dirs=[], sandbox="none", max_output_tokens=256)


class Approvals:
    """Records every permission question; answers with a fixed verdict."""

    def __init__(self, answer: bool):
        self.answer = answer
        self.asked: list[str] = []

    def __call__(self, question: str) -> bool:
        self.asked.append(question)
        return self.answer


@pytest.fixture
def deny() -> Approvals:
    return Approvals(False)


@pytest.fixture
def allow() -> Approvals:
    return Approvals(True)


@pytest.fixture
def make_agent(cfg):
    def make(llm, approve=None, **kwargs):
        return Agent(
            cfg,
            llm,
            approve or Approvals(False),
            today=lambda: date(2026, 9, 26),
            sandbox=Sandbox(cfg.workspace, "none"),
            **kwargs,
        )

    return make
