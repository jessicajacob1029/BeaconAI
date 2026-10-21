"""Tests for allow-listed environment overrides in beaconai.config."""

import pydantic
import pytest

from beaconai.config import DEFAULT_CONFIG_PATH, apply_env_overrides, load_config

BASE = {
    "seed": 42,
    "data": {
        "dev_sample": 50000,
        "paths": {"raw": "data/raw", "interim": "data/interim", "processed": "data/processed"},
    },
}


def test_no_env_returns_same_values():
    assert apply_env_overrides(BASE, {}) == BASE


def test_seed_override():
    out = apply_env_overrides(BASE, {"BEACONAI_SEED": "7"})
    assert out["seed"] == 7


def test_dev_sample_override():
    out = apply_env_overrides(BASE, {"BEACONAI_DEV_SAMPLE": "1000"})
    assert out["data"]["dev_sample"] == 1000


def test_dev_sample_null():
    out = apply_env_overrides(BASE, {"BEACONAI_DEV_SAMPLE": "null"})
    assert out["data"]["dev_sample"] is None


def test_unknown_env_var_ignored():
    assert apply_env_overrides(BASE, {"BEACONAI_FOO": "x"}) == BASE


def test_input_not_mutated():
    apply_env_overrides(BASE, {"BEACONAI_DEV_SAMPLE": "1"})
    assert BASE["data"]["dev_sample"] == 50000


def test_load_config_applies_override():
    cfg = load_config(DEFAULT_CONFIG_PATH, env={"BEACONAI_SEED": "123"})
    assert cfg.seed == 123


def test_bad_env_value_rejected():
    with pytest.raises(pydantic.ValidationError):
        load_config(DEFAULT_CONFIG_PATH, env={"BEACONAI_SEED": "abc"})


def test_negative_dev_sample_rejected():
    with pytest.raises(pydantic.ValidationError):
        load_config(DEFAULT_CONFIG_PATH, env={"BEACONAI_DEV_SAMPLE": "-5"})
