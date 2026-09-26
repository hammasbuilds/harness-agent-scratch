from pathlib import Path

import pytest

from harness.config import PROJECT_SETTINGS, Config, load_dotenv


def test_defaults_keep_ollama_on_the_cpu(tmp_path):
    cfg = Config.from_env(tmp_path, {})
    assert cfg.backend == "ollama"
    assert cfg.num_gpu == 0
    assert cfg.base_url == "http://localhost:11434"
    assert cfg.context_limit == cfg.num_ctx == 8192


def test_openai_backend_gets_its_own_default_url_and_a_larger_limit(tmp_path):
    cfg = Config.from_env(tmp_path, {"HARNESS_BACKEND": "openai", "HARNESS_API_KEY": "k"})
    assert cfg.base_url == "https://openrouter.ai/api/v1"
    assert cfg.api_key == "k"
    assert cfg.context_limit == 64000


def test_context_limit_follows_num_ctx_for_ollama(tmp_path):
    cfg = Config.from_env(tmp_path, {"HARNESS_NUM_CTX": "16384"})
    assert cfg.context_limit == 16384


def test_bad_numbers_name_the_variable(tmp_path):
    with pytest.raises(ValueError, match="HARNESS_NUM_GPU"):
        Config.from_env(tmp_path, {"HARNESS_NUM_GPU": "lots"})


def test_unknown_backend_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="HARNESS_BACKEND"):
        Config.from_env(tmp_path, {"HARNESS_BACKEND": "gemini"})


def test_skills_dirs_env_is_an_override(tmp_path):
    cfg = Config.from_env(tmp_path, {"HARNESS_SKILLS_DIRS": ""})
    assert cfg.skills_dirs == []  # set but empty means "no skills", not "defaults"
    assert Config.from_env(tmp_path, {}).skills_dirs is None


@pytest.mark.parametrize("env, message", [
    ({"HARNESS_NUM_CTX": "100"}, "num_ctx must be at least 512"),
    ({"HARNESS_OUTPUT_CAP": "0"}, "output_cap"),
    ({"HARNESS_MAX_STEPS": "0"}, "max_steps"),
    ({"HARNESS_RETRIES": "-1"}, "retries"),
    ({"HARNESS_REQUEST_TIMEOUT": "0"}, "request_timeout"),
])
def test_impossible_settings_fail_at_startup(tmp_path, env, message):
    with pytest.raises(ValueError, match=message):
        Config.from_env(tmp_path, env)


def test_context_limit_cannot_exceed_ollamas_num_ctx(tmp_path):
    with pytest.raises(ValueError, match="above num_ctx"):
        Config.from_env(tmp_path, {"HARNESS_CONTEXT_LIMIT": "100000", "HARNESS_NUM_CTX": "8192"})
    # a hosted model has no num_ctx to respect
    assert Config.from_env(tmp_path, {"HARNESS_BACKEND": "openai", "HARNESS_CONTEXT_LIMIT": "100000"}).context_limit == 100000


def test_every_setting_is_reachable_from_the_environment(tmp_path):
    cfg = Config.from_env(tmp_path, {"HARNESS_SUBAGENT_MAX_STEPS": "5", "HARNESS_COMPACT_AT": "0.9",
                                     "HARNESS_COMPACT_TO": "0.3"})
    assert (cfg.subagent_max_steps, cfg.compact_at, cfg.compact_to) == (5, 0.9, 0.3)


@pytest.mark.parametrize("line, value", [
    ('HARNESS_MODEL="a # b"', "a # b"),
    ("HARNESS_MODEL=qwen2.5:7b  # the small one", "qwen2.5:7b"),
    ("HARNESS_MODEL=\"'x'\"", "'x'"),  # one pair of quotes, not every quote character
    ("HARNESS_MODEL='q' # note", "q"),
    ("HARNESS_MODEL=a#b", "a#b"),  # no space before #, not a comment
])
def test_dotenv_values(tmp_path, line, value):
    (tmp_path / ".env").write_text(line + "\n")
    env = {}
    load_dotenv(tmp_path / ".env", env)
    assert env["HARNESS_MODEL"] == value


def test_project_env_is_an_allowlist(tmp_path):
    (tmp_path / ".env").write_text("HARNESS_NUM_GPU=99\nHARNESS_MODEL=big\nHARNESS_TEMPERATURE=0\n"
                                   "HARNESS_MAX_STEPS=900\nOTHER=1\n")
    env = {}
    ignored = load_dotenv(tmp_path / ".env", env, only=PROJECT_SETTINGS)
    assert env == {"HARNESS_TEMPERATURE": "0"}
    # step counts could multiply a paid session's cost; OTHER is not ours to mention
    assert ignored == ["HARNESS_NUM_GPU", "HARNESS_MODEL", "HARNESS_MAX_STEPS"]


def test_notepad_bom_does_not_hide_the_first_setting(tmp_path):
    (tmp_path / ".env").write_bytes(b"\xef\xbb\xbfHARNESS_MODEL=x\n")
    env = {}
    load_dotenv(tmp_path / ".env", env)
    assert env == {"HARNESS_MODEL": "x"}


@pytest.mark.parametrize("value", ["nan", "inf", "-1", "3"])
def test_temperature_must_be_a_real_number_in_range(tmp_path, value):
    # nan passed every comparison and was sent as NaN, which is not valid JSON
    with pytest.raises(ValueError, match="temperature"):
        Config.from_env(tmp_path, {"HARNESS_TEMPERATURE": value})


def test_the_reply_cap_must_leave_room_for_the_prompt(tmp_path):
    with pytest.raises(ValueError, match="max_output_tokens"):
        Config.from_env(tmp_path, {"HARNESS_MAX_OUTPUT_TOKENS": "6000"})  # num_ctx 8192
    assert Config.from_env(tmp_path, {}).max_output_tokens == 2048


def test_compaction_thresholds_must_be_ordered(tmp_path):
    with pytest.raises(ValueError, match="compact_to"):
        Config(workspace=tmp_path, compact_at=0.3, compact_to=0.5)


def test_dotenv_fills_only_unset_variables(tmp_path):
    (tmp_path / ".env").write_text('# comment\nHARNESS_MODEL="qwen2.5:14b-instruct"\nexport HARNESS_NUM_GPU=3\nNOEQUALS\n')
    env = {"HARNESS_NUM_GPU": "0"}
    load_dotenv(tmp_path / ".env", env)
    assert env == {"HARNESS_NUM_GPU": "0", "HARNESS_MODEL": "qwen2.5:14b-instruct"}


def test_missing_dotenv_is_fine(tmp_path):
    env = {}
    load_dotenv(tmp_path / "nope.env", env)
    assert env == {}
