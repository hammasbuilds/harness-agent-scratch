from pathlib import Path

import pytest

from harness.config import Config, load_dotenv


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
