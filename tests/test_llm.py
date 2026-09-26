import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

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
    'Sure.\n<tool_call>\n{"name": "bash", "arguments": {"command": "ls"}}\n</tool_call>',
])
def test_inline_tool_calls_are_recovered(content):
    calls = extract_inline_tool_calls(content, {"bash"})
    assert [(c.name, json.loads(c.arguments)) for c in calls] == [("bash", {"command": "ls"})]


@pytest.mark.parametrize("content", [
    'Call it like this: {"name": "bash", "arguments": {"command": "ls"}}',  # prose around JSON
    '{"name": "rm_everything", "arguments": {}}',  # not a known tool
    '{"name": "bash", "arguments": "ls"}',  # arguments not an object
    "{not json}",
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


def test_unreachable_server_is_an_llm_error():
    with pytest.raises(LLMError, match="cannot reach"):
        http_post_json("http://127.0.0.1:9/api/chat", {}, {}, 2)
