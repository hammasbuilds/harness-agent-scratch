"""Talking to the model: two HTTP backends that return the same Reply.

Messages are kept in the OpenAI chat format everywhere in the harness. The
Ollama backend translates them to Ollama's native /api/chat on the way out,
because that endpoint accepts per-request `options` and the OpenAI-compatible
/v1 endpoint does not. `num_gpu: 0` is what keeps the model on the CPU while
another job owns the GPU, and `num_ctx` is what stops Ollama cutting the prompt
at its small default context.

No SDK: one POST with http.client per call.
"""

from __future__ import annotations

import copy
import http.client
import itertools
import json
import re
import socket
import threading
import time
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from typing import Callable, Protocol

from .config import Config


class LLMError(RuntimeError):
    """The endpoint failed, or answered with something the harness cannot use.

    `retryable` marks failures worth one more attempt: rate limits, 5xx, a
    dropped connection. A refused connection (server not running) or a timeout
    is not: on a CPU a timeout already cost many minutes, and repeating the
    same request would cost them again.
    """

    def __init__(self, message: str, retryable: bool = False, retry_after: float | None = None):
        super().__init__(message)
        self.retryable = retryable
        self.retry_after = retry_after  # seconds the server asked for, from Retry-After


RETRY_STATUS = {408, 409, 429, 500, 502, 503, 504}
MAX_RESPONSE_BYTES = 20_000_000
MAX_ERROR_BYTES = 65_536
MAX_RETRY_WAIT = 60.0


class ContextOverflow(LLMError):
    """The request cannot be made to fit the model's context."""


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
    truncated: bool = False  # the model hit its output-token limit mid-reply

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


def _read_body(stream, deadline: float, cap: int, url: str, truncate: bool = False) -> bytes:
    """Read a response within the deadline and size cap.

    read1 returns whatever has arrived; read(n) would block until n bytes or the
    end, so the deadline would never be checked while a server trickles.
    """
    read = getattr(stream, "read1", None) or stream.read
    chunks, size = [], 0
    while chunk := read(65536):
        chunks.append(chunk)
        size += len(chunk)
        if size > cap:
            if truncate:
                break
            raise LLMError(f"{url} sent more than {cap:,} bytes; giving up")
        if time.monotonic() > deadline:
            if truncate:
                break
            raise TimeoutError
    return b"".join(chunks)[:cap]


def _retry_after(headers) -> float | None:
    """Seconds from a Retry-After header, when it is a number (dates are ignored)."""
    try:
        return float(headers.get("Retry-After")) if headers and headers.get("Retry-After") else None
    except (TypeError, ValueError):
        return None


def _connection(url: str, timeout: float) -> tuple[http.client.HTTPConnection, str]:
    """A connection for `url` (through the environment's proxy, if one applies)
    and the request target to send on it."""
    parts = urllib.parse.urlsplit(url)
    if parts.scheme not in ("http", "https") or not parts.hostname:
        raise LLMError(f"not an http(s) URL: {url}")
    cls = http.client.HTTPSConnection if parts.scheme == "https" else http.client.HTTPConnection
    port = parts.port or (443 if parts.scheme == "https" else 80)
    target = urllib.parse.urlunsplit(("", "", parts.path or "/", parts.query, ""))
    proxy = urllib.request.getproxies().get(parts.scheme)
    # The local Ollama is always reached directly: a proxy from the environment
    # would otherwise receive every prompt, and Python only bypasses localhost
    # when NO_PROXY says so.
    loopback = parts.hostname in ("localhost", "::1") or parts.hostname.startswith("127.")
    if proxy and not loopback and not urllib.request.proxy_bypass(parts.hostname):
        p = urllib.parse.urlsplit(proxy if "://" in proxy else f"http://{proxy}")
        # The proxy's own scheme decides how to talk to it and its default port;
        # an HTTPS target through an http:// proxy is a CONNECT over plain HTTP.
        proxy_cls = http.client.HTTPSConnection if p.scheme == "https" else http.client.HTTPConnection
        conn = proxy_cls(p.hostname, p.port or (443 if p.scheme == "https" else 80), timeout=timeout)
        if parts.scheme == "https":
            conn.set_tunnel(parts.hostname, port)
        else:
            target = url  # a plain-HTTP proxy wants the absolute URL
        return conn, target
    return cls(parts.hostname, port, timeout=timeout), target


def http_post_json(url: str, body: dict, headers: dict, timeout: float) -> dict:
    """One JSON POST, bounded as a whole.

    - A timer closes the socket at the deadline, so a server trickling its
      status line, its headers or its body cannot stretch the request; the
      per-read socket timeout alone never trips on one byte a second.
    - Redirects are not followed. urllib would follow a 302 as a GET and send
      the Authorization header, the API key, to whatever host it named.
    """
    data = json.dumps(body).encode("utf-8")
    deadline = time.monotonic() + timeout
    conn, target = _connection(url, timeout)
    expired = threading.Event()

    def expire():
        expired.set()
        sock = getattr(conn, "sock", None)
        if sock is not None:
            try:
                sock.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            sock.close()

    timer = threading.Timer(timeout, expire)
    timer.daemon = True
    timer.start()
    try:
        conn.request("POST", target, body=data,
                     headers={"Content-Type": "application/json", "Content-Length": str(len(data)), **headers})
        response = conn.getresponse()
        status, reply_headers = response.status, response.headers
        if status >= 300:
            raw = _read_body(response, deadline, MAX_ERROR_BYTES, url, truncate=True)
        else:
            raw = _read_body(response, deadline, MAX_RESPONSE_BYTES, url)
    except LLMError:
        raise
    except ConnectionRefusedError as e:
        raise LLMError(f"cannot reach {url}: {e}") from None  # not running; retrying will not start it
    except (OSError, http.client.HTTPException) as e:
        if expired.is_set() or isinstance(e, TimeoutError):
            raise LLMError(f"timed out after {timeout:.0f}s waiting for {url}") from None
        raise LLMError(f"connection to {url} failed: {e!r}", retryable=True) from None  # reset, dropped
    finally:
        timer.cancel()
        conn.close()

    # On Linux, closing the socket at the deadline reads as a clean end of data,
    # not an error: without this check a timeout came back as "no usable JSON",
    # and a cut-off body could even parse. An error status that arrived in time
    # is still reported as itself, with its Retry-After; only its body was cut.
    if expired.is_set() and status < 300:
        raise LLMError(f"timed out after {timeout:.0f}s waiting for {url}")
    if 300 <= status < 400:
        raise LLMError(f"HTTP {status} from {url}: redirect to {reply_headers.get('Location')!r} not followed "
                       "(it would carry the API key to another address)")
    if status >= 400:
        raise LLMError(f"HTTP {status} from {url}: {raw.decode('utf-8', 'replace')[:2000]}",
                       retryable=status in RETRY_STATUS, retry_after=_retry_after(reply_headers))
    try:
        return json.loads(raw.decode("utf-8"))
    # ValueError also covers an integer longer than Python's 4,300-digit limit.
    except (ValueError, RecursionError) as e:
        raise LLMError(f"{url} did not return usable JSON ({type(e).__name__}): {raw[:300]!r}") from None


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
    if candidates and _TAGGED_CALL.sub("", content).strip():
        return []  # <tool_call> blocks inside prose are quoted examples, not calls
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
        if isinstance(args, str):  # OpenAI-style: the arguments object as JSON text
            try:
                args = json.loads(args)
            except json.JSONDecodeError:
                continue
        if name in known and isinstance(args, dict):
            calls.append(ToolCall(id=f"inline_{i}", name=name, arguments=json.dumps(args)))
    return calls


def _tool_names(tools: list[dict] | None) -> set[str]:
    return {t["function"]["name"] for t in tools or []}


def _args_object(text: str) -> dict:
    try:
        obj = json.loads(text)
    except (ValueError, TypeError, RecursionError):
        return {}
    return obj if isinstance(obj, dict) else {}


class _Backend:
    def __init__(self, cfg: Config, post: Post = http_post_json, sleep: Callable[[float], None] = time.sleep):
        self.cfg = cfg
        self.post = post
        self.sleep = sleep
        self._ids = itertools.count()

    def _post(self, url: str, body: dict, headers: dict) -> dict:
        """POST with retries for transient failures: 1s, then 3s, or longer if the
        server's Retry-After asks for it (capped at a minute), then give up."""
        for attempt in range(self.cfg.retries + 1):
            try:
                return self.post(url, body, headers, self.cfg.request_timeout)
            except LLMError as e:
                if not e.retryable or attempt == self.cfg.retries:
                    raise
                self.sleep(min(MAX_RETRY_WAIT, max(3 ** attempt, e.retry_after or 0)))
        raise AssertionError("unreachable")

    def chat(self, messages: list[dict], tools: list[dict] | None) -> Reply:
        url, headers = self._endpoint()
        data = self._post(url, self.request_body(messages, tools), headers)
        # Valid JSON in the wrong shape ([] or {"message": null}) must fail as an
        # LLMError the CLI reports, not as an AttributeError that ends the session.
        try:
            if not isinstance(data, dict):
                raise TypeError(f"expected a JSON object, got {type(data).__name__}")
            return self._parse(data, tools, url)
        except LLMError:
            raise
        except (AttributeError, TypeError, KeyError, IndexError, ValueError, RecursionError) as e:
            raise LLMError(f"{url} answered in an unexpected shape ({e}): {str(data)[:300]}") from None

    def _endpoint(self) -> tuple[str, dict]:
        raise NotImplementedError

    def _parse(self, data: dict, tools: list[dict] | None, url: str) -> Reply:
        raise NotImplementedError

    def request_body(self, messages: list[dict], tools: list[dict] | None) -> dict:
        raise NotImplementedError

    def _finish(self, content: str, calls: list[ToolCall], usage: dict, tools: list[dict] | None,
                truncated: bool = False) -> Reply:
        # Valid JSON with the wrong types ("content": 123, "name": ["bash"]) would
        # otherwise surface far away, as a crash in the printer or the toolbox.
        if not isinstance(content, str):
            raise TypeError(f"content is {type(content).__name__}, not text")
        for c in calls:
            if not isinstance(c.name, str) or not isinstance(c.arguments, str):
                raise TypeError(f"tool call has a {type(c.name).__name__} name or {type(c.arguments).__name__} arguments")
        if not isinstance(usage, dict) or not all(isinstance(v, int) for v in usage.values()):
            usage = {}
        if not calls and tools:
            calls = extract_inline_tool_calls(content, _tool_names(tools))
            if calls:
                content = ""
        # Some servers omit ids or reuse them per reply; tool results are matched
        # by id, so every call in the conversation gets a unique one.
        for c in calls:
            c.id = f"{c.id or 'call'}_{next(self._ids)}"
        return Reply(content=content, tool_calls=calls, usage=usage, truncated=truncated)


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
                "num_predict": self.cfg.max_output_tokens,
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

    def _endpoint(self) -> tuple[str, dict]:
        return f"{self.cfg.base_url.rstrip('/').removesuffix('/v1')}/api/chat", {}

    def _parse(self, data: dict, tools: list[dict] | None, url: str) -> Reply:
        if "error" in data:
            raise LLMError(f"ollama: {data['error']}")
        msg = data.get("message")
        if not isinstance(msg, dict):
            raise TypeError("no message object in the reply")
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
        return self._finish(msg.get("content") or "", calls, usage, tools,
                            truncated=data.get("done_reason") == "length")


class OpenAIBackend(_Backend):
    def request_body(self, messages: list[dict], tools: list[dict] | None) -> dict:
        body = {"model": self.cfg.model, "messages": [self._message(m) for m in messages],
                "temperature": self.cfg.temperature, "max_tokens": self.cfg.max_output_tokens}
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

    def _endpoint(self) -> tuple[str, dict]:
        headers = {"Authorization": f"Bearer {self.cfg.api_key}"} if self.cfg.api_key else {}
        return f"{self.cfg.base_url.rstrip('/')}/chat/completions", headers

    def _parse(self, data: dict, tools: list[dict] | None, url: str) -> Reply:
        if data.get("error"):
            raise LLMError(f"{url}: {data['error']}")
        try:
            choice = data["choices"][0]
            msg = choice["message"]
        except (KeyError, IndexError, TypeError):
            raise LLMError(f"{url}: no choices in response: {str(data)[:500]}") from None
        content = msg.get("content") or ""
        if isinstance(content, list):  # some providers send content as typed parts
            content = "".join(p.get("text", "") for p in content if isinstance(p, dict))
        calls = []
        for tc in msg.get("tool_calls") or []:
            fn = tc.get("function") or {}
            args = fn.get("arguments", "{}")
            calls.append(ToolCall(id=tc.get("id") or "call", name=fn.get("name", ""),
                                  arguments=args if isinstance(args, str) else json.dumps(args)))
        # Usage is bookkeeping: a malformed one must not throw away a good answer.
        raw = data.get("usage")
        raw = raw if isinstance(raw, dict) else {}
        details = raw.get("prompt_tokens_details")
        usage = {
            "prompt_tokens": raw.get("prompt_tokens", 0),
            "completion_tokens": raw.get("completion_tokens", 0),
            "cached_tokens": details.get("cached_tokens", 0) if isinstance(details, dict) else 0,
        }
        return self._finish(content, calls, usage, tools, truncated=choice.get("finish_reason") == "length")


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
