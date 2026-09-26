import http.client
import json
import socket
import threading
import time
import urllib.request
from http.server import BaseHTTPRequestHandler, HTTPServer, ThreadingHTTPServer

import pytest

from harness.config import Config
from harness.llm import (LLMError, OllamaBackend, OpenAIBackend, Reply, ScriptedLLM, ToolCall, call,
                         extract_inline_tool_calls, http_post_json, make_llm)

TOOLS = [{"type": "function", "function": {"name": "bash", "description": "", "parameters": {}}}]

HISTORY = [
    {"role": "system", "content": "sys"},
    {"role": "user", "content": "hi"},
    {"role": "assistant", "content": "", "tool_calls": [
        {"id": "c1", "type": "function", "function": {"name": "bash", "arguments": '{"command": "ls"}'}}]},
    {"role": "tool", "tool_call_id": "c1", "name": "bash", "content": "a.txt"},
]


class FakePost:
    def __init__(self, response):
        self.response = response
        self.calls = []

    def __call__(self, url, body, headers, timeout):
        self.calls.append((url, body, headers, timeout))
        return self.response


def cfg(tmp_path, **kw):
    return Config(workspace=tmp_path, **kw)


def test_ollama_request_forces_cpu_and_sets_context(tmp_path):
    post = FakePost({"message": {"role": "assistant", "content": "done"}})
    OllamaBackend(cfg(tmp_path), post).chat(HISTORY, TOOLS)
    url, body, _, timeout = post.calls[0]
    assert url == "http://localhost:11434/api/chat"
    assert body["options"]["num_gpu"] == 0
    assert body["options"]["num_ctx"] == 8192
    assert body["keep_alive"] == "30m"
    assert body["stream"] is False
    assert timeout == 900.0


def test_reply_length_is_capped_on_both_backends(tmp_path):
    post = FakePost({"message": {"content": "ok"}, "choices": [{"message": {"content": "ok"}}]})
    OllamaBackend(cfg(tmp_path, max_output_tokens=1024), post).chat(HISTORY, None)
    OpenAIBackend(cfg(tmp_path, backend="openai", max_output_tokens=1024), post).chat(HISTORY, None)
    assert post.calls[0][1]["options"]["num_predict"] == 1024
    assert post.calls[1][1]["max_tokens"] == 1024


def test_ollama_translates_history_to_native_shapes(tmp_path):
    post = FakePost({"message": {"content": "ok"}})
    OllamaBackend(cfg(tmp_path), post).chat(HISTORY, TOOLS)
    msgs = post.calls[0][1]["messages"]
    assert msgs[2]["tool_calls"] == [{"function": {"name": "bash", "arguments": {"command": "ls"}}}]
    assert msgs[3] == {"role": "tool", "content": "a.txt", "tool_name": "bash"}


def test_ollama_base_url_with_v1_suffix_still_hits_native_api(tmp_path):
    post = FakePost({"message": {"content": "ok"}})
    OllamaBackend(cfg(tmp_path, base_url="http://localhost:11434/v1/"), post).chat(HISTORY, None)
    assert post.calls[0][0] == "http://localhost:11434/api/chat"


def test_ollama_reply_is_normalised(tmp_path):
    post = FakePost({"message": {"content": "", "tool_calls": [
        {"function": {"name": "bash", "arguments": {"command": "pwd"}}},
        {"function": {"name": "bash", "arguments": {"command": "ls"}}}]},
        "prompt_eval_count": 120, "eval_count": 9})
    reply = OllamaBackend(cfg(tmp_path), post).chat(HISTORY, TOOLS)
    assert [json.loads(c.arguments) for c in reply.tool_calls] == [{"command": "pwd"}, {"command": "ls"}]
    assert len({c.id for c in reply.tool_calls}) == 2  # ids are unique even when the server sends none
    assert reply.usage == {"prompt_tokens": 120, "completion_tokens": 9}


def test_ollama_error_field_raises(tmp_path):
    with pytest.raises(LLMError, match="model not found"):
        OllamaBackend(cfg(tmp_path), FakePost({"error": "model not found"})).chat(HISTORY, None)


def test_openai_request_drops_harness_only_fields(tmp_path):
    post = FakePost({"choices": [{"message": {"content": "hi"}}]})
    OpenAIBackend(cfg(tmp_path, backend="openai", base_url="https://x/api/v1", api_key="sk"), post).chat(HISTORY, TOOLS)
    url, body, headers, _ = post.calls[0]
    assert url == "https://x/api/v1/chat/completions"
    assert headers == {"Authorization": "Bearer sk"}
    assert body["messages"][3] == {"role": "tool", "tool_call_id": "c1", "content": "a.txt"}
    assert body["messages"][2]["content"] is None
    assert "options" not in body


def test_openai_reply_with_cached_tokens(tmp_path):
    post = FakePost({"choices": [{"message": {"content": None, "tool_calls": [
        {"id": "x", "type": "function", "function": {"name": "bash", "arguments": '{"command": "ls"}'}}]}}],
        "usage": {"prompt_tokens": 3624, "completion_tokens": 40, "prompt_tokens_details": {"cached_tokens": 3328}}})
    reply = OpenAIBackend(cfg(tmp_path, backend="openai"), post).chat(HISTORY, TOOLS)
    assert reply.tool_calls[0].name == "bash"
    assert reply.content == ""
    assert reply.usage["cached_tokens"] == 3328


def test_openai_without_choices_raises(tmp_path):
    with pytest.raises(LLMError, match="no choices"):
        OpenAIBackend(cfg(tmp_path, backend="openai"), FakePost({"id": "x"})).chat(HISTORY, None)


def test_make_llm_picks_backend(tmp_path):
    assert isinstance(make_llm(cfg(tmp_path)), OllamaBackend)
    assert isinstance(make_llm(cfg(tmp_path, backend="openai")), OpenAIBackend)


@pytest.mark.parametrize("content", [
    '{"name": "bash", "arguments": {"command": "ls"}}',
    '```json\n{"name": "bash", "arguments": {"command": "ls"}}\n```',
    '<tool_call>\n{"name": "bash", "arguments": {"command": "ls"}}\n</tool_call>',
    '\n  <tool_call>{"name": "bash", "arguments": {"command": "ls"}}</tool_call>\n',
])
def test_inline_tool_calls_are_recovered(content):
    calls = extract_inline_tool_calls(content, {"bash"})
    assert [(c.name, json.loads(c.arguments)) for c in calls] == [("bash", {"command": "ls"})]


@pytest.mark.parametrize("content", [
    'Call it like this: {"name": "bash", "arguments": {"command": "ls"}}',  # prose around JSON
    '{"name": "rm_everything", "arguments": {}}',  # not a known tool
    '{"name": "bash", "arguments": "ls"}',  # arguments not an object
    "{not json}",
    # a tagged block quoted in prose is an example, not a call (it was once run)
    'Never do this: <tool_call>{"name": "bash", "arguments": {"command": "rm -rf build"}}</tool_call> ok?',
    'Sure.\n<tool_call>\n{"name": "bash", "arguments": {"command": "ls"}}\n</tool_call>',
])
def test_prose_and_unknown_tools_are_not_calls(content):
    assert extract_inline_tool_calls(content, {"bash"}) == []


def test_backend_uses_inline_fallback_only_when_tools_were_offered(tmp_path):
    text = '{"name": "bash", "arguments": {"command": "ls"}}'
    backend = OllamaBackend(cfg(tmp_path), FakePost({"message": {"content": text}}))
    assert backend.chat(HISTORY, TOOLS).tool_calls[0].name == "bash"
    assert backend.chat(HISTORY, TOOLS).content == ""
    assert backend.chat(HISTORY, None).content == text


def test_reply_to_message():
    msg = Reply("x", [ToolCall("id1", "bash", "{}")]).to_message()
    assert msg == {"role": "assistant", "content": "x", "tool_calls": [
        {"id": "id1", "type": "function", "function": {"name": "bash", "arguments": "{}"}}]}
    assert Reply("y").to_message() == {"role": "assistant", "content": "y"}


def test_scripted_llm_assigns_ids_and_records_requests():
    llm = ScriptedLLM([Reply("", [call("bash", command="ls")]), Reply("done")])
    first = llm.chat([{"role": "user", "content": "a"}], TOOLS)
    llm.chat([], None)
    assert first.tool_calls[0].id == "call_0"
    assert llm.requests[0][0] == [{"role": "user", "content": "a"}]
    with pytest.raises(LLMError, match="ran out"):
        llm.chat([], None)


class _Handler(BaseHTTPRequestHandler):
    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        if body.get("model") == "missing":
            self.send_response(404)
            self.end_headers()
            self.wfile.write(b'{"error": "model \\"missing\\" not found"}')
            return
        payload = json.dumps({"message": {"content": f"echo {body['messages'][-1]['content']}"}}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, *args):
        pass


@pytest.fixture
def local_server():
    server = HTTPServer(("127.0.0.1", 0), _Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_port}"
    server.shutdown()


def test_real_http_round_trip_against_a_local_stub(tmp_path, local_server):
    backend = OllamaBackend(cfg(tmp_path, base_url=local_server))
    assert backend.chat([{"role": "user", "content": "ping"}], None).content == "echo ping"


def test_http_errors_carry_the_server_message(tmp_path, local_server):
    with pytest.raises(LLMError, match="HTTP 404.*not found"):
        OllamaBackend(cfg(tmp_path, base_url=local_server, model="missing")).chat([{"role": "user", "content": "x"}], None)


class _TrickleHandler(BaseHTTPRequestHandler):
    """Sends a valid response one byte every 0.3 s: each read is quick, the whole is not."""

    def do_POST(self):
        self.rfile.read(int(self.headers["Content-Length"]))
        body = b'{"message": {"content": "slow"}}'
        self.send_response(200)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        for byte in body:
            try:
                self.wfile.write(bytes([byte]))
                self.wfile.flush()
            except OSError:
                return
            time.sleep(0.3)

    def log_message(self, *args):
        pass


def test_the_timeout_covers_the_whole_response_not_each_read():
    server = ThreadingHTTPServer(("127.0.0.1", 0), _TrickleHandler)
    server.daemon_threads = True
    threading.Thread(target=server.serve_forever, daemon=True).start()
    start = time.monotonic()
    try:
        with pytest.raises(LLMError, match="timed out"):
            http_post_json(f"http://127.0.0.1:{server.server_port}/api/chat", {}, {}, 1.5)
        elapsed = time.monotonic() - start
    finally:
        server.shutdown()
    assert elapsed < 4  # the full trickle takes ~10 s; each single read is well under the timeout


class _SlowErrorHandler(BaseHTTPRequestHandler):
    """HTTP 500 whose body trickles in: the error path needs the deadline too."""

    def do_POST(self):
        self.rfile.read(int(self.headers["Content-Length"]))
        body = b'{"error": "internal failure, please retry later"}'
        self.send_response(500)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Retry-After", "7")
        self.end_headers()
        for byte in body:
            try:
                self.wfile.write(bytes([byte]))
                self.wfile.flush()
            except OSError:
                return
            time.sleep(0.2)

    def log_message(self, *args):
        pass


def test_a_slow_error_body_is_cut_at_the_deadline():
    server = ThreadingHTTPServer(("127.0.0.1", 0), _SlowErrorHandler)
    server.daemon_threads = True
    threading.Thread(target=server.serve_forever, daemon=True).start()
    start = time.monotonic()
    try:
        with pytest.raises(LLMError, match="HTTP 500") as err:
            http_post_json(f"http://127.0.0.1:{server.server_port}/api/chat", {}, {}, 1.0)
        elapsed = time.monotonic() - start
    finally:
        server.shutdown()
    assert elapsed < 3  # the whole body takes ~10 s
    assert err.value.retryable and err.value.retry_after == 7.0


def test_retry_after_lengthens_the_wait_up_to_a_cap(tmp_path):
    post = FlakyPost([LLMError("429", retryable=True, retry_after=20), LLMError("429", retryable=True, retry_after=500)],
                     {"message": {"content": "ok"}})
    waits = []
    OllamaBackend(cfg(tmp_path), post, sleep=waits.append).chat(HISTORY, None)
    assert waits == [20, 60]  # the server's 20 s honoured; 500 s capped at a minute


def test_inline_call_with_arguments_as_json_text():
    calls = extract_inline_tool_calls('{"name": "bash", "arguments": "{\\"command\\": \\"ls\\"}"}', {"bash"})
    assert [(c.name, json.loads(c.arguments)) for c in calls] == [("bash", {"command": "ls"})]


def _closed_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def test_refused_connection_is_an_llm_error_and_not_retried():
    with pytest.raises(LLMError, match="cannot reach") as err:
        http_post_json(f"http://127.0.0.1:{_closed_port()}/api/chat", {}, {}, 5)
    assert err.value.retryable is False  # the server is not running; retrying will not start it


class FakeResponse:
    def __init__(self, body: bytes, status: int = 200, headers=None):
        self.chunks, self.status, self.headers = [body], status, headers or {}

    def read1(self, n=-1):
        return self.chunks.pop() if self.chunks else b""


class FakeConnection:
    """Stands in for http.client: raises `error` on request, or returns `response`."""

    def __init__(self, error=None, response=None):
        self.error, self.response, self.sock = error, response, None

    def request(self, *args, **kwargs):
        if self.error:
            raise self.error

    def getresponse(self):
        return self.response

    def close(self):
        pass


def fake_connection(monkeypatch, **kwargs):
    monkeypatch.setattr("harness.llm._connection", lambda url, timeout: (FakeConnection(**kwargs), "/api/chat"))


@pytest.mark.parametrize("error, retryable", [
    (ConnectionResetError(10054, "reset"), True),
    (http.client.RemoteDisconnected("gone"), True),
    (http.client.IncompleteRead(b"par"), True),
    (TimeoutError("read timed out"), False),
    (ConnectionRefusedError(10061, "refused"), False),
])
def test_network_failures_never_escape_as_raw_exceptions(monkeypatch, error, retryable):
    fake_connection(monkeypatch, error=error)
    with pytest.raises(LLMError) as err:
        http_post_json("http://x/api/chat", {}, {}, 1)
    assert err.value.retryable is retryable


def test_non_json_body_is_an_llm_error(monkeypatch):
    fake_connection(monkeypatch, response=FakeResponse(b"<html>502 Bad Gateway</html>"))
    with pytest.raises(LLMError, match="did not return usable JSON"):
        http_post_json("http://x/api/chat", {}, {}, 1)


def test_an_integer_past_pythons_digit_limit_is_an_llm_error(monkeypatch):
    fake_connection(monkeypatch, response=FakeResponse(b'{"eval_count": 1' + b"0" * 5000 + b"}"))
    with pytest.raises(LLMError, match="did not return usable JSON"):
        http_post_json("http://x/api/chat", {}, {}, 1)


def test_a_proxy_without_a_port_gets_its_own_schemes_default(monkeypatch):
    from harness.llm import _connection
    monkeypatch.setattr(urllib.request, "getproxies", lambda: {"https": "http://proxy.example"})
    monkeypatch.setattr(urllib.request, "proxy_bypass", lambda host: False)
    conn, target = _connection("https://openrouter.ai/api/v1/chat/completions", 5)
    assert (conn.host, conn.port) == ("proxy.example", 80)  # was 443, the target's default
    assert type(conn) is http.client.HTTPConnection and target == "/api/v1/chat/completions"


@pytest.mark.parametrize("url", ["http://localhost:11434/api/chat", "http://127.0.0.1:11434/api/chat",
                                 "http://[::1]:11434/api/chat"])
def test_the_local_model_is_never_reached_through_a_proxy(monkeypatch, url):
    # A proxy from the environment would otherwise receive every prompt.
    from harness.llm import _connection
    monkeypatch.setattr(urllib.request, "getproxies", lambda: {"http": "http://proxy.example:3128"})
    monkeypatch.setattr(urllib.request, "proxy_bypass", lambda host: False)
    conn, target = _connection(url, 5)
    assert conn.host != "proxy.example" and target == "/api/chat"


def test_deeply_nested_json_is_an_llm_error_not_a_recursion_crash(monkeypatch):
    fake_connection(monkeypatch, response=FakeResponse(b"[" * 200_000 + b"]" * 200_000))
    with pytest.raises(LLMError, match="RecursionError"):
        http_post_json("http://x/api/chat", {}, {}, 1)


def test_redirects_are_refused_so_the_key_never_travels(monkeypatch):
    fake_connection(monkeypatch, response=FakeResponse(b"", status=302, headers={"Location": "http://evil.example/"}))
    with pytest.raises(LLMError, match="redirect to 'http://evil.example/' not followed") as err:
        http_post_json("http://x/api/chat", {}, {"Authorization": "Bearer sk-secret"}, 1)
    assert not err.value.retryable


class _RedirectHandler(BaseHTTPRequestHandler):
    seen: list = []

    def do_POST(self):
        self.rfile.read(int(self.headers["Content-Length"]))
        self.send_response(302)
        self.send_header("Location", f"http://localhost:{self.server.server_port}/elsewhere")
        self.send_header("Content-Length", "0")
        self.end_headers()

    def do_GET(self):  # urllib turned the POST into this GET, carrying the key
        _RedirectHandler.seen.append(self.headers.get("Authorization"))
        self.send_response(200)
        self.end_headers()

    def log_message(self, *args):
        pass


def test_a_real_redirect_never_reaches_the_second_address():
    server = ThreadingHTTPServer(("127.0.0.1", 0), _RedirectHandler)
    server.daemon_threads = True
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        with pytest.raises(LLMError, match="HTTP 302"):
            http_post_json(f"http://127.0.0.1:{server.server_port}/api/chat", {}, {"Authorization": "Bearer sk"}, 5)
    finally:
        server.shutdown()
    assert _RedirectHandler.seen == []


class _SlowHeadersHandler(BaseHTTPRequestHandler):
    def do_POST(self):
        self.rfile.read(int(self.headers["Content-Length"]))
        self.wfile.write(b"HTTP/1.1 200 OK\r\n")
        for i in range(40):
            try:
                self.wfile.write(f"X-Slow-{i}: y\r\n".encode())
                self.wfile.flush()
            except OSError:
                return
            time.sleep(0.2)

    def log_message(self, *args):
        pass


def test_the_deadline_also_covers_a_trickled_status_line_and_headers():
    server = ThreadingHTTPServer(("127.0.0.1", 0), _SlowHeadersHandler)
    server.daemon_threads = True
    threading.Thread(target=server.serve_forever, daemon=True).start()
    start = time.monotonic()
    try:
        with pytest.raises(LLMError, match="timed out"):
            http_post_json(f"http://127.0.0.1:{server.server_port}/api/chat", {}, {}, 1.0)
        elapsed = time.monotonic() - start
    finally:
        server.shutdown()
    assert elapsed < 3  # 40 headers at 0.2 s would take 8 s


@pytest.mark.parametrize("backend", [OllamaBackend, OpenAIBackend])
@pytest.mark.parametrize("message", [
    {"content": 123, "tool_calls": [{"function": {"name": "bash", "arguments": {}}}]},
    {"content": "", "tool_calls": [{"function": {"name": ["bash"], "arguments": {}}}]},
])
def test_wrong_typed_fields_are_an_llm_error(tmp_path, backend, message):
    kind = "openai" if backend is OpenAIBackend else "ollama"
    body = {"message": message, "choices": [{"message": message}]}
    with pytest.raises(LLMError, match="unexpected shape"):
        backend(cfg(tmp_path, backend=kind), FakePost(body)).chat(HISTORY, TOOLS)


class FlakyPost:
    def __init__(self, failures, final):
        self.failures = list(failures)
        self.final = final
        self.calls = 0

    def __call__(self, url, body, headers, timeout):
        self.calls += 1
        if self.failures:
            raise self.failures.pop(0)
        return self.final


def test_transient_failures_are_retried_with_backoff(tmp_path):
    post = FlakyPost([LLMError("503", retryable=True), LLMError("reset", retryable=True)],
                     {"message": {"content": "finally"}})
    waits = []
    backend = OllamaBackend(cfg(tmp_path), post, sleep=waits.append)
    assert backend.chat(HISTORY, None).content == "finally"
    assert post.calls == 3 and waits == [1, 3]


def test_retries_give_up_and_permanent_errors_fail_at_once(tmp_path):
    post = FlakyPost([LLMError("503", retryable=True)] * 5, {})
    with pytest.raises(LLMError, match="503"):
        OllamaBackend(cfg(tmp_path, retries=2), post, sleep=lambda s: None).chat(HISTORY, None)
    assert post.calls == 3
    post = FlakyPost([LLMError("HTTP 401", retryable=False)], {})
    with pytest.raises(LLMError, match="401"):
        OllamaBackend(cfg(tmp_path), post, sleep=lambda s: None).chat(HISTORY, None)
    assert post.calls == 1


def test_http_status_decides_retryability(tmp_path, local_server):
    with pytest.raises(LLMError) as err:
        OllamaBackend(cfg(tmp_path, base_url=local_server, model="missing"), sleep=lambda s: None).chat(
            [{"role": "user", "content": "x"}], None)
    assert err.value.retryable is False  # 404: asking again will not help


def test_truncated_replies_are_flagged(tmp_path):
    ollama = OllamaBackend(cfg(tmp_path), FakePost({"message": {"content": "half"}, "done_reason": "length"}))
    assert ollama.chat(HISTORY, None).truncated is True
    openai = OpenAIBackend(cfg(tmp_path, backend="openai"),
                           FakePost({"choices": [{"message": {"content": "half"}, "finish_reason": "length"}]}))
    assert openai.chat(HISTORY, None).truncated is True
    done = OllamaBackend(cfg(tmp_path), FakePost({"message": {"content": "all"}, "done_reason": "stop"}))
    assert done.chat(HISTORY, None).truncated is False


@pytest.mark.parametrize("backend", [OllamaBackend, OpenAIBackend])
@pytest.mark.parametrize("body", [
    [], "text", 42, None,
    {"message": None, "choices": [{"message": None}]},
    {"message": {"tool_calls": ["x"]}, "choices": [{"message": {"tool_calls": ["x"]}}]},
    {"message": {"tool_calls": [{"function": "bash"}]}, "choices": [{"message": {"tool_calls": [{"function": "bash"}]}}]},
    {"choices": "none"},
])
def test_wrong_shaped_json_is_an_llm_error(tmp_path, backend, body):
    kind = "openai" if backend is OpenAIBackend else "ollama"
    with pytest.raises(LLMError):
        backend(cfg(tmp_path, backend=kind), FakePost(body)).chat(HISTORY, TOOLS)


@pytest.mark.parametrize("usage", [["x"], "lots", {"prompt_tokens_details": "n/a"}, None])
def test_a_malformed_usage_field_does_not_throw_away_a_good_answer(tmp_path, usage):
    post = FakePost({"choices": [{"message": {"content": "the answer"}}], "usage": usage})
    assert OpenAIBackend(cfg(tmp_path, backend="openai"), post).chat(HISTORY, None).content == "the answer"


def test_openai_content_as_typed_parts(tmp_path):
    post = FakePost({"choices": [{"message": {"content": [{"type": "text", "text": "a"}, {"type": "text", "text": "b"}]}}]})
    assert OpenAIBackend(cfg(tmp_path, backend="openai"), post).chat(HISTORY, None).content == "ab"
