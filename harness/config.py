"""Settings, read once from environment variables and .env files."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

DEFAULT_BASE_URLS = {
    "ollama": "http://localhost:11434",
    "openai": "https://openrouter.ai/api/v1",
}


# Settings that decide where requests go, what runs commands and what is
# confined. A project's own .env (possibly from a cloned, untrusted repo, or
# written by the model itself) must not be able to set these: a base URL there
# would send the user's real API key to someone else's server.
TRUSTED_ONLY = frozenset({
    "HARNESS_BACKEND", "HARNESS_BASE_URL", "HARNESS_API_KEY",
    "HARNESS_SHELL", "HARNESS_SANDBOX", "HARNESS_SKILLS_DIRS",
})


def user_config_file(env: Mapping[str, str]) -> Path:
    return Path(env.get("HARNESS_CONFIG") or Path.home() / ".config" / "harness" / ".env")


def load_dotenv(path: Path, environ: dict | None = None, skip: frozenset[str] = frozenset()) -> list[str]:
    """Fill unset variables from a KEY=VALUE file. Variables already set win.

    Keys in `skip` are not loaded; their names are returned so the caller can
    say they were ignored.
    """
    environ = os.environ if environ is None else environ
    if not path.is_file():
        return []
    skipped = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip().removeprefix("export ").strip()
        if key in skip:
            skipped.append(key)
            continue
        environ.setdefault(key, value.strip().strip('"').strip("'"))
    return skipped


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
        if self.request_timeout <= 0:
            problems.append("request_timeout must be positive")
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
            output_cap=number("OUTPUT_CAP", 3000, int),
            max_steps=number("MAX_STEPS", 40, int),
            shell=env.get("HARNESS_SHELL") or None,
            # An override, not an extra path: when set, only these folders are searched.
            skills_dirs=[Path(p) for p in skills_raw.split(os.pathsep) if p] if skills_raw is not None else None,
            sandbox=get("SANDBOX", "auto"),
        )
