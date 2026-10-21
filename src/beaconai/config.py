"""Load and validate BeaconAI configuration from YAML.

Every tunable value lives in config/*.yaml. This module turns the YAML into typed,
validated Pydantic objects so that a typo or wrong type fails at startup.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "base.yaml"


class _StrictModel(BaseModel):
    # Unknown keys are errors, so a misspelled key in YAML is caught immediately.
    model_config = ConfigDict(extra="forbid", frozen=True)


class DataPaths(_StrictModel):
    raw: Path
    interim: Path
    processed: Path


class DataConfig(_StrictModel):
    dev_sample: int | None = Field(default=None, gt=0)
    paths: DataPaths


class Config(_StrictModel):
    seed: int = Field(default=42, ge=0)
    data: DataConfig


def apply_env_overrides(raw: dict[str, Any], env: Mapping[str, str]) -> dict[str, Any]:
    """Return a copy of the raw YAML dict with environment overrides applied.

    Called before validation, so any override still goes through the Pydantic checks.
    `env` is passed in (instead of reading os.environ here) to keep this pure and testable.
    """
    # TODO(you): decide whether env vars may override YAML values, and implement it.
    return raw


def load_config(path: str | Path = DEFAULT_CONFIG_PATH) -> Config:
    """Read a YAML config file and return a validated Config.

    Raises FileNotFoundError if the file is missing and pydantic.ValidationError
    if the contents do not match the schema.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open(encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    raw = apply_env_overrides(raw, os.environ)
    return Config.model_validate(raw)
