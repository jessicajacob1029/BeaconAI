"""Load and validate BeaconAI configuration from YAML.

Every tunable value lives in config/*.yaml. This module turns the YAML into typed,
validated Pydantic objects so that a typo or wrong type fails at startup.
"""

from __future__ import annotations

import copy
import os
from collections.abc import Mapping
from itertools import pairwise
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "base.yaml"

# Only these env vars may change config. Each maps to a path inside the YAML.
ENV_OVERRIDES: dict[str, tuple[str, ...]] = {
    "BEACONAI_SEED": ("seed",),
    "BEACONAI_DEV_SAMPLE": ("data", "dev_sample"),
}


class _StrictModel(BaseModel):
    # Unknown keys are errors, so a misspelled key in YAML is caught immediately.
    model_config = ConfigDict(extra="forbid", frozen=True)


class DataPaths(_StrictModel):
    raw: Path
    interim: Path
    processed: Path


class DataSource(_StrictModel):
    figshare_article_id: int = Field(gt=0)
    doi: str
    files: list[str] = Field(min_length=1)
    timeout_seconds: float = Field(default=60, gt=0)


class EncodingConfig(_StrictModel):
    age_bins: list[int] = Field(min_length=1)
    age_scale: float = Field(default=110, gt=0)
    scale_max: float = Field(default=10, gt=0)

    @field_validator("age_bins")
    @classmethod
    def _bins_start_at_zero_and_increase(cls, bins: list[int]) -> list[int]:
        if bins[0] != 0:
            raise ValueError("age_bins must start at 0")
        if any(b <= a for a, b in pairwise(bins)):
            raise ValueError("age_bins must be strictly increasing")
        return bins


class DataConfig(_StrictModel):
    dev_sample: int | None = Field(default=None, gt=0)
    paths: DataPaths
    source: DataSource
    encoding: EncodingConfig


class EdaConfig(_StrictModel):
    split: str = "train"
    output_dir: Path = Path("reports")


class Config(_StrictModel):
    seed: int = Field(default=42, ge=0)
    data: DataConfig
    eda: EdaConfig = EdaConfig()


def _parse_env_value(value: str) -> Any:
    """Turn an env string into a Python value. Validation checks it later."""
    if value.strip().lower() in {"null", "none", ""}:
        return None
    try:
        return int(value)
    except ValueError:
        return value  # left as a string so validation rejects it


def apply_env_overrides(raw: dict[str, Any], env: Mapping[str, str]) -> dict[str, Any]:
    """Return a copy of the raw YAML dict with allow-listed env overrides applied.

    Called before validation, so any override still goes through the Pydantic checks.
    `env` is passed in (instead of reading os.environ here) to keep this pure and testable.
    """
    result = copy.deepcopy(raw)
    for var_name, key_path in ENV_OVERRIDES.items():
        if var_name not in env:
            continue
        target = result
        for key in key_path[:-1]:
            target = target.setdefault(key, {})
        target[key_path[-1]] = _parse_env_value(env[var_name])
    return result


def load_config(
    path: str | Path = DEFAULT_CONFIG_PATH,
    env: Mapping[str, str] | None = None,
) -> Config:
    """Read a YAML config file and return a validated Config.

    Raises FileNotFoundError if the file is missing and pydantic.ValidationError
    if the contents do not match the schema.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open(encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    raw = apply_env_overrides(raw, os.environ if env is None else env)
    return Config.model_validate(raw)
