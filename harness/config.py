"""Settings, read once from environment variables and .env files."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

DEFAULT_BASE_URLS = {
    "ollama": "http://localhost:11434",
    "openai": "https://openrouter.ai/api/v1",
}


# The only settings a project's own .env may change. It may come from a cloned,
# untrusted repository, or be written by the model itself, so it is an
# allowlist of harmless tuning: not where requests go (a base URL there would
# send the user's real API key elsewhere), not what runs commands or how they
# are confined, not the model, and not num_gpu, which keeps a model that loads
# onto the GPU from pushing a training job out of memory.
# Step counts and the context limit are left out too: on a paid API a cloned
# repository could raise them to multiply what a session costs.
PROJECT_SETTINGS = frozenset({"HARNESS_TEMPERATURE", "HARNESS_COMPACT_AT", "HARNESS_COMPACT_TO"})

_QUOTED = re.compile(r"""(["'])(.*?)\1\s*(#.*)?$""")


def user_config_file(env: Mapping[str, str]) -> Path:
    return Path(env.get("HARNESS_CONFIG") or Path.home() / ".config" / "harness" / ".env")


def _dotenv_value(raw: str) -> str:
    """`"a # b"` keeps its hash; `a  # note` loses the comment; one pair of
    matching quotes is removed, never more."""
    raw = raw.strip()
    quoted = _QUOTED.match(raw)
    if quoted:
        return quoted.group(2)
    return re.split(r"\s+#", raw, maxsplit=1)[0].strip()


def load_dotenv(path: Path, environ: dict | None = None, only: frozenset[str] | None = None) -> list[str]:
    """Fill unset variables from a KEY=VALUE file. Variables already set win.

    With `only`, other keys are not loaded; the HARNESS_ ones among them are
    returned so the caller can say they were ignored.
    """
    environ = os.environ if environ is None else environ
    if not path.is_file():
        return []
    ignored = []
    # utf-8-sig: Notepad's BOM would otherwise become part of the first key.
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip().removeprefix("export ").strip()
        if only is not None and key not in only:
            if key.startswith("HARNESS_"):
                ignored.append(key)
            continue
        environ.setdefault(key, _dotenv_value(value))
    return ignored


@dataclass
class Config:
    workspace: Path
    backend: str = "ollama"  # "ollama" = native /api/chat, "openai" = any /chat/completions
    base_url: str = DEFAULT_BASE_URLS["ollama"]
    api_key: str = ""
    model: str = "qwen2.5:7b-instruct"
    temperature: float = 0.2
    # Ollama only. num_gpu=0 keeps every layer on the CPU, so a training job that
    # owns the GPU is never pushed out of memory by the agent.
    num_gpu: int = 0
    num_thread: int = 6
    num_ctx: int = 8192
    keep_alive: str = "30m"  # keep the model loaded so its prompt cache survives between turns
    request_timeout: float = 900.0  # CPU inference is slow; one call can take minutes
    retries: int = 2  # extra attempts after a rate limit, 5xx or dropped connection
    context_limit: int = 8192  # tokens the transcript may grow to before compaction
    compact_at: float = 0.85
    compact_to: float = 0.35
    output_cap: int = 3000  # characters of one tool result the model sees
    max_steps: int = 40
    subagent_max_steps: int = 20
    shell: str | None = None
    skills_dirs: list[Path] | None = None  # None = ~/.agents/skills and <workspace>/.agents/skills
    sandbox: str = "auto"  # auto | none | required

    def __post_init__(self):
        problems = []
        if not 0 < self.compact_to < self.compact_at <= 1:
            problems.append(f"need 0 < compact_to ({self.compact_to}) < compact_at ({self.compact_at}) <= 1")
        for name, low in (("num_ctx", 512), ("context_limit", 512), ("output_cap", 200), ("max_steps", 1),
                          ("subagent_max_steps", 1), ("num_thread", 1), ("retries", 0)):
            if getattr(self, name) < low:
                problems.append(f"{name} must be at least {low}, got {getattr(self, name)}")
        if not self.request_timeout > 0:
            problems.append("request_timeout must be positive")
        # nan passes every < and > comparison, and serialises as NaN, which is not JSON.
        if not 0 <= self.temperature <= 2:
            problems.append(f"temperature must be between 0 and 2, got {self.temperature}")
        if not (0 < self.compact_at <= 1 and 0 < self.compact_to < 1):
            problems.append("compact_at and compact_to must be numbers between 0 and 1")
        if self.backend == "ollama" and self.context_limit > self.num_ctx:
            # Ollama silently drops the start of a prompt longer than num_ctx, so
            # compaction must trigger below it.
            problems.append(f"context_limit ({self.context_limit}) is above num_ctx ({self.num_ctx}); "
                            "Ollama would cut the prompt before compaction ever ran")
        if problems:
            raise ValueError("invalid settings: " + "; ".join(problems))

    @classmethod
    def from_env(cls, workspace: Path, env: Mapping[str, str]) -> "Config":
        def get(key: str, default: str) -> str:
            return env.get(f"HARNESS_{key}", default)

        def number(key: str, default, kind):
            raw = env.get(f"HARNESS_{key}")
            if raw is None or raw == "":
                return default
            try:
                return kind(raw)
            except ValueError:
                raise ValueError(f"HARNESS_{key}={raw!r} is not a valid {kind.__name__}") from None

        backend = get("BACKEND", "ollama").lower()
        if backend not in DEFAULT_BASE_URLS:
            raise ValueError(f"HARNESS_BACKEND must be one of {sorted(DEFAULT_BASE_URLS)}, got {backend!r}")
        num_ctx = number("NUM_CTX", 8192, int)
        skills_raw = env.get("HARNESS_SKILLS_DIRS")
        return cls(
            workspace=Path(workspace).resolve(),
            backend=backend,
            base_url=get("BASE_URL", DEFAULT_BASE_URLS[backend]),
            api_key=get("API_KEY", ""),
            model=get("MODEL", "qwen2.5:7b-instruct"),
            temperature=number("TEMPERATURE", 0.2, float),
            num_gpu=number("NUM_GPU", 0, int),
            num_thread=number("NUM_THREAD", 6, int),
            num_ctx=num_ctx,
            keep_alive=get("KEEP_ALIVE", "30m"),
            request_timeout=number("REQUEST_TIMEOUT", 900.0, float),
            retries=number("RETRIES", 2, int),
            # With Ollama the real ceiling is num_ctx; past it Ollama drops the
            # start of the prompt without telling anyone.
            context_limit=number("CONTEXT_LIMIT", num_ctx if backend == "ollama" else 64000, int),
            compact_at=number("COMPACT_AT", 0.85, float),
            compact_to=number("COMPACT_TO", 0.35, float),
            output_cap=number("OUTPUT_CAP", 3000, int),
            max_steps=number("MAX_STEPS", 40, int),
            subagent_max_steps=number("SUBAGENT_MAX_STEPS", 20, int),
            shell=env.get("HARNESS_SHELL") or None,
            # An override, not an extra path: when set, only these folders are searched.
            skills_dirs=[Path(p) for p in skills_raw.split(os.pathsep) if p] if skills_raw is not None else None,
            sandbox=get("SANDBOX", "auto"),
        )
