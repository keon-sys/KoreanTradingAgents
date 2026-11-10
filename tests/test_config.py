from pathlib import Path

import pytest

from krta.config import Settings


def test_settings_defaults(tmp_path):
    s = Settings.from_env({"KRTA_DATA_DIR": str(tmp_path), "KRTA_QUICK_MODEL": "q", "KRTA_DEEP_MODEL": "d"})
    assert (s.debate_rounds, s.risk_rounds, s.llm_max_retries) == (1, 1, 6)
    assert s.data_dir == tmp_path
    assert (s.quick_model, s.deep_model) == ("q", "d")


def test_settings_default_data_dir():
    s = Settings.from_env({"KRTA_QUICK_MODEL": "q", "KRTA_DEEP_MODEL": "d"})
    assert s.data_dir == Path("~/.krta").expanduser()


def test_settings_overrides(tmp_path):
    s = Settings.from_env({
        "KRTA_DATA_DIR": str(tmp_path),
        "KRTA_QUICK_MODEL": "q",
        "KRTA_DEEP_MODEL": "d",
        "KRTA_DEBATE_ROUNDS": "2",
        "KRTA_RISK_ROUNDS": "3",
        "KRTA_LLM_MAX_RETRIES": "0",
    })
    assert (s.debate_rounds, s.risk_rounds, s.llm_max_retries) == (2, 3, 0)


@pytest.mark.parametrize("missing", ["KRTA_QUICK_MODEL", "KRTA_DEEP_MODEL"])
def test_settings_model_required(missing):
    env = {"KRTA_QUICK_MODEL": "q", "KRTA_DEEP_MODEL": "d"}
    env[missing] = ""
    with pytest.raises(ValueError, match=missing):
        Settings.from_env(env)
    del env[missing]
    with pytest.raises(ValueError, match=missing):
        Settings.from_env(env)


@pytest.mark.parametrize("value", ["abc", "0", "-1"])
def test_settings_invalid_rounds(value):
    env = {"KRTA_QUICK_MODEL": "q", "KRTA_DEEP_MODEL": "d", "KRTA_DEBATE_ROUNDS": value}
    with pytest.raises(ValueError, match="KRTA_DEBATE_ROUNDS"):
        Settings.from_env(env)


def test_settings_negative_retries_rejected():
    env = {"KRTA_QUICK_MODEL": "q", "KRTA_DEEP_MODEL": "d", "KRTA_LLM_MAX_RETRIES": "-1"}
    with pytest.raises(ValueError, match="KRTA_LLM_MAX_RETRIES"):
        Settings.from_env(env)


def test_settings_frozen(tmp_path):
    s = Settings.from_env({"KRTA_DATA_DIR": str(tmp_path), "KRTA_QUICK_MODEL": "q", "KRTA_DEEP_MODEL": "d"})
    with pytest.raises(Exception):
        s.debate_rounds = 5  # type: ignore[misc]
