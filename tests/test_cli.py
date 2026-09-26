import io

import pytest

from harness import cli
from harness.llm import Reply, ScriptedLLM, ToolCall


def test_args_override_env(tmp_path):
    args = cli.parse_args(["-w", str(tmp_path), "--backend", "openai", "--model", "m", "--sandbox", "none"])
    env = {"HARNESS_MODEL": "from-env"}
    cfg = cli.build_config(args, env)
    assert cfg.backend == "openai" and cfg.model == "m" and cfg.sandbox == "none"
    assert cfg.workspace == tmp_path.resolve()


def test_dotenv_in_workspace_is_read(tmp_path):
    (tmp_path / ".env").write_text("HARNESS_MODEL=qwen2.5:14b-instruct\n")
    cfg = cli.build_config(cli.parse_args(["-w", str(tmp_path)]), {})
    assert cfg.model == "qwen2.5:14b-instruct"


def test_bad_config_exits_2(tmp_path, capsys):
    (tmp_path / ".env").write_text("HARNESS_NUM_GPU=many\n")
    assert cli.main(["-w", str(tmp_path)]) == 2
    assert "HARNESS_NUM_GPU" in capsys.readouterr().err


def test_one_shot_prompt(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(cli, "make_llm", lambda cfg: ScriptedLLM([Reply("all done")]))
    monkeypatch.setenv("HARNESS_SKILLS_DIRS", "")
    assert cli.main(["-w", str(tmp_path), "--sandbox", "none", "-p", "do it"]) == 0
    out = capsys.readouterr().out
    assert "num_gpu=0" in out and out.rstrip().endswith("all done")


def test_one_shot_reports_an_unreachable_model(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(cli, "make_llm", lambda cfg: ScriptedLLM([]))
    monkeypatch.setenv("HARNESS_SKILLS_DIRS", "")
    assert cli.main(["-w", str(tmp_path), "--sandbox", "none", "-p", "x"]) == 1
    assert "error:" in capsys.readouterr().err


def test_repl_commands(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(cli, "make_llm", lambda cfg: ScriptedLLM([Reply("hi there")]))
    monkeypatch.setenv("HARNESS_SKILLS_DIRS", "")
    lines = iter(["", "/todos", "hello", "/tokens", "/clear", "/exit"])
    monkeypatch.setattr("builtins.input", lambda prompt="": next(lines))
    assert cli.main(["-w", str(tmp_path), "--sandbox", "none"]) == 0
    out = capsys.readouterr().out
    assert "(no todos)" in out and "hi there" in out and "tokens in context" in out and "fresh conversation" in out


def test_ask_user(monkeypatch):
    monkeypatch.setattr("builtins.input", lambda prompt="": "y")
    assert cli.ask_user("run: rm x") is True
    monkeypatch.setattr("builtins.input", lambda prompt="": "")
    assert cli.ask_user("run: rm x") is False

    def eof(prompt=""):
        raise EOFError
    monkeypatch.setattr("builtins.input", eof)
    assert cli.ask_user("run: rm x") is False


def test_printer_formats_events():
    out = io.StringIO()
    p = cli.Printer(out)
    p("tool_call", ToolCall("1", "bash", '{"command": "ls"}'))
    p("tool_result", ("bash", "x" * 1000))
    p("usage", {"prompt_tokens": 3624, "completion_tokens": 40, "cached_tokens": 3328})
    p("compacted", (7000, 2500))
    text = out.getvalue()
    assert '> bash {"command": "ls"}' in text
    assert "(1000 chars)" in text
    assert "3624 prompt tokens, 3328 cached, 40 out" in text
    assert "~7000 -> ~2500" in text
