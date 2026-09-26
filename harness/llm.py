"""Talking to the model: two HTTP backends that return the same Reply.

Messages are kept in the OpenAI chat format everywhere in the harness. The
Ollama backend translates them to Ollama's native /api/chat on the way out,
because that endpoint accepts per-request `options` and the OpenAI-compatible
/v1 endpoint does not. `num_gpu: 0` is what keeps the model on the CPU while
another job owns the GPU, and `num_ctx` is what stops Ollama cutting the prompt
at its small default context.

No SDK: one POST with urllib per call.
"""

from __future__ import annotations

import copy
import itertools
import json
import re
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Callable, Protocol

from .config import Config


class LLMError(RuntimeError):
    """The endpoint failed, or answered with something the harness cannot use."""


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: str  # JSON text exactly as the model wrote it; the toolbox parses it


@dataclass
class Reply:
    content: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    usage: dict[str, int] = field(default_factory=dict)

    def to_message(self) -> dict:
        msg: dict = {"role": "assistant", "content": self.content}
        if self.tool_calls:
            msg["tool_calls"] = [
                {"id": c.id, "type": "function", "function": {"name": c.name, "arguments": c.arguments}}
                for c in self.tool_calls
            ]
        return msg


class LLM(Protocol):
    def chat(self, messages: list[dict], tools: list[dict] | None) -> Reply: ...


Post = Callable[[str, dict, dict, float], dict]


def http_post_json(url: str, body: dict, headers: dict, timeout: float) -> dict:
    request = urllib.request.Request(
        url,
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json", **headers},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", "replace")[:2000]
        raise LLMError(f"HTTP {e.code} from {url}: {detail}") from None
    except urllib.error.URLError as e:
        raise LLMError(f"cannot reach {url}: {e.reason}") from None
    except (TimeoutError, json.JSONDecodeError) as e:
        raise LLMError(f"bad response from {url}: {e}") from None


_TAGGED_CALL = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.S)
_FENCED = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.S)


def extract_inline_tool_calls(content: str, known: set[str]) -> list[ToolCall]:
    """Recover tool calls a model wrote as text instead of in the tool_calls field.

    qwen2.5-coder under Ollama is the known offender: it prints
    {"name": "bash", "arguments": {...}} as its whole reply. Only a reply that is
    nothing but such JSON (or <tool_call> blocks) counts, so a model explaining
    JSON in prose is never mistaken for a call.
    """
    candidates = _TAGGED_CALL.findall(content)
    if not candidates:
        text = content.strip()
        fenced = _FENCED.fullmatch(text)
        if fenced:
            text = fenced.group(1)
        if text.startswith("{") and text.endswith("}"):
            candidates = [text]
    calls = []
    for i, text in enumerate(candidates):
        try:
            obj = json.loads(text)
        except json.JSONDecodeError:
            continue
        if not isinstance(obj, dict):
            continue
        name = obj.get("name")
        args = obj.get("arguments", obj.get("parameters", {}))
        if name in known and isinstance(args, dict):
            calls.append(ToolCall(id=f"inline_{i}", name=name, arguments=json.dumps(args)))
    return calls


def _tool_names(tools: list[dict] | None) -> set[str]:
    return {t["function"]["name"] for t in tools or []}


def _args_object(text: str) -> dict:
    try:
        obj = json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return {}
    return obj if isinstance(obj, dict) else {}


class _Backend:
    def __init__(self, cfg: Config, post: Post = http_post_json):
        self.cfg = cfg
        self.post = post
        self._ids = itertools.count()

    def _finish(self, content: str, calls: list[ToolCall], usage: dict, tools: list[dict] | None) -> Reply:
        if not calls and tools:
            calls = extract_inline_tool_calls(content, _tool_names(tools))
            if calls:
                content = ""
        # Some servers omit ids or reuse them per reply; tool results are matched
        # by id, so every call in the conversation gets a unique one.
        for c in calls:
            c.id = f"{c.id or 'call'}_{next(self._ids)}"
        return Reply(content=content, tool_calls=calls, usage=usage)


class OllamaBackend(_Backend):
    def request_body(self, messages: list[dict], tools: list[dict] | None) -> dict:
        body = {
            "model": self.cfg.model,
            "messages": [self._message(m) for m in messages],
            "stream": False,
            "keep_alive": self.cfg.keep_alive,
            "options": {
                "num_gpu": self.cfg.num_gpu,
                "num_thread": self.cfg.num_thread,
                "num_ctx": self.cfg.num_ctx,
                "temperature": self.cfg.temperature,
            },
        }
        if tools:
            body["tools"] = tools
        return body

    @staticmethod
    def _message(msg: dict) -> dict:
        if msg["role"] == "tool":
            out = {"role": "tool", "content": msg.get("content") or ""}
            if msg.get("name"):
                out["tool_name"] = msg["name"]
            return out
        out = {"role": msg["role"], "content": msg.get("content") or ""}
        if msg.get("tool_calls"):
            # Ollama wants arguments as an object, not JSON text.
            out["tool_calls"] = [
                {"function": {"name": tc["function"]["name"], "arguments": _args_object(tc["function"]["arguments"])}}
                for tc in msg["tool_calls"]
            ]
        return out

    def chat(self, messages: list[dict], tools: list[dict] | None) -> Reply:
        base = self.cfg.base_url.rstrip("/").removesuffix("/v1")
        data = self.post(f"{base}/api/chat", self.request_body(messages, tools), {}, self.cfg.request_timeout)
        if "error" in data:
            raise LLMError(f"ollama: {data['error']}")
        msg = data.get("message") or {}
        calls = []
        for tc in msg.get("tool_calls") or []:
            fn = tc.get("function") or {}
            args = fn.get("arguments", {})
            calls.append(ToolCall(
                id=tc.get("id") or "call",
                name=fn.get("name", ""),
                arguments=args if isinstance(args, str) else json.dumps(args),
            ))
        # prompt_eval_count counts only prompt tokens Ollama had to evaluate, so a
        # low number on a long transcript means the prefix cache was hit.
        usage = {"prompt_tokens": data.get("prompt_eval_count", 0), "completion_tokens": data.get("eval_count", 0)}
        return self._finish(msg.get("content") or "", calls, usage, tools)


class OpenAIBackend(_Backend):
    def request_body(self, messages: list[dict], tools: list[dict] | None) -> dict:
        body = {"model": self.cfg.model, "messages": [self._message(m) for m in messages],
                "temperature": self.cfg.temperature}
        if tools:
            body["tools"] = tools
        return body

    @staticmethod
    def _message(msg: dict) -> dict:
        if msg["role"] == "tool":
            # The harness keeps the tool name on tool messages for Ollama; the
            # OpenAI schema has no such field.
            return {"role": "tool", "tool_call_id": msg["tool_call_id"], "content": msg.get("content") or ""}
        out = {"role": msg["role"], "content": msg.get("content") or ""}
        if msg.get("tool_calls"):
            out["tool_calls"] = copy.deepcopy(msg["tool_calls"])
            out["content"] = msg.get("content") or None
        return out

    def chat(self, messages: list[dict], tools: list[dict] | None) -> Reply:
        headers = {"Authorization": f"Bearer {self.cfg.api_key}"} if self.cfg.api_key else {}
        url = f"{self.cfg.base_url.rstrip('/')}/chat/completions"
        data = self.post(url, self.request_body(messages, tools), headers, self.cfg.request_timeout)
        if data.get("error"):
            raise LLMError(f"{url}: {data['error']}")
        try:
            msg = data["choices"][0]["message"]
        except (KeyError, IndexError, TypeError):
            raise LLMError(f"{url}: no choices in response: {str(data)[:500]}") from None
        calls = []
        for tc in msg.get("tool_calls") or []:
            fn = tc.get("function") or {}
            args = fn.get("arguments", "{}")
            calls.append(ToolCall(id=tc.get("id") or "call", name=fn.get("name", ""),
                                  arguments=args if isinstance(args, str) else json.dumps(args)))
        raw = data.get("usage") or {}
        usage = {
            "prompt_tokens": raw.get("prompt_tokens", 0),
            "completion_tokens": raw.get("completion_tokens", 0),
            "cached_tokens": (raw.get("prompt_tokens_details") or {}).get("cached_tokens", 0),
        }
        return self._finish(msg.get("content") or "", calls, usage, tools)


def make_llm(cfg: Config, post: Post = http_post_json) -> LLM:
    return OllamaBackend(cfg, post) if cfg.backend == "ollama" else OpenAIBackend(cfg, post)


class ScriptedLLM:
    """A stand-in model that replays pre-written replies. Used by the tests and
    by `demo.py --scripted`; the tools it calls still really run."""

    def __init__(self, replies: list[Reply | Callable[[list[dict]], Reply]]):
        self.replies = list(replies)
        self.requests: list[tuple[list[dict], list[dict] | None]] = []
        self._ids = itertools.count()

    def chat(self, messages: list[dict], tools: list[dict] | None) -> Reply:
        self.requests.append((copy.deepcopy(messages), copy.deepcopy(tools)))
        if not self.replies:
            raise LLMError("scripted model ran out of replies")
        nxt = self.replies.pop(0)
        reply = nxt(messages) if callable(nxt) else copy.deepcopy(nxt)
        for c in reply.tool_calls:
            c.id = c.id or f"call_{next(self._ids)}"
        return reply


def call(tool: str, /, **arguments) -> ToolCall:
    """Shorthand for scripted tool calls. `tool` is positional-only so a tool
    argument may itself be called `name` (read_skill's is)."""
    return ToolCall(id="", name=tool, arguments=json.dumps(arguments))
