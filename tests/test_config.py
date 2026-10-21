from pathlib import Path

import pytest
from pydantic import ValidationError

from beaconai.config import DEFAULT_CONFIG_PATH, Config, load_config


def _write(tmp_path: Path, text: str) -> Path:
    p = tmp_path / "cfg.yaml"
    p.write_text(text, encoding="utf-8")
    return p


VALID_YAML = """
seed: 7
data:
  dev_sample: 100
  paths: {raw: r, interim: i, processed: p}
"""


def test_base_config_loads() -> None:
    cfg = load_config(DEFAULT_CONFIG_PATH)
    assert isinstance(cfg, Config)
    assert cfg.seed == 42
    assert cfg.data.dev_sample == 50000
    assert cfg.data.paths.raw == Path("data/raw")


def test_custom_config_loads(tmp_path: Path) -> None:
    cfg = load_config(_write(tmp_path, VALID_YAML))
    assert cfg.seed == 7
    assert cfg.data.dev_sample == 100


def test_dev_sample_can_be_null(tmp_path: Path) -> None:
    cfg = load_config(_write(tmp_path, VALID_YAML.replace("dev_sample: 100", "dev_sample: null")))
    assert cfg.data.dev_sample is None


def test_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_config(tmp_path / "nope.yaml")


@pytest.mark.parametrize(
    "bad_yaml",
    [
        VALID_YAML.replace("dev_sample: 100", "dev_sample: -5"),  # must be positive
        VALID_YAML.replace("seed: 7", "seed: not-a-number"),  # wrong type
        VALID_YAML + "typo_key: 1\n",  # unknown key
        "seed: 1\n",  # missing data section
    ],
)
def test_invalid_config_rejected(tmp_path: Path, bad_yaml: str) -> None:
    with pytest.raises(ValidationError):
        load_config(_write(tmp_path, bad_yaml))
