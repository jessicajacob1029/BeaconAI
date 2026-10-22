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
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

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


ModelName = Literal["logreg", "xgboost", "mlp"]


class ModelSpec(_StrictModel):
    fixed: dict[str, Any] = Field(default_factory=dict)
    grid: dict[str, list[Any]] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _check_grid(self) -> ModelSpec:
        empty = [k for k, v in self.grid.items() if not v]
        if empty:
            raise ValueError(f"grid values must be non-empty lists: {empty}")
        both = sorted(set(self.fixed) & set(self.grid))
        if both:
            raise ValueError(f"keys are both fixed and tuned: {both}")
        return self


class ModelsConfig(_StrictModel):
    logreg: ModelSpec
    xgboost: ModelSpec
    mlp: ModelSpec


class TrainConfig(_StrictModel):
    models: list[ModelName] = Field(min_length=1)
    selection_metric: str = "top3_accuracy"
    # Used in order only when the selection metric ties exactly.
    tie_breakers: list[str] = Field(default_factory=lambda: ["top1_accuracy", "macro_f1"])
    latency_calls: int = Field(default=100, gt=0)


class MlflowConfig(_StrictModel):
    tracking_db: Path
    artifact_root: Path
    experiment: str
    registered_model: str
    # Calibrated models (Phase 3): full-data runs move champion_alias, dev runs dev_alias.
    champion_alias: str = "champion"
    dev_alias: str = "dev"
    # Raw Phase 2 winners get their own aliases, so retraining never moves `champion`.
    raw_champion_alias: str = "raw-champion"
    raw_dev_alias: str = "raw-dev"


class EvalConfig(_StrictModel):
    ks: list[int] = Field(min_length=1)
    ddx_ks: list[int] = Field(default_factory=list)
    output_dir: Path = Path("reports")
    top_confusions: int = Field(default=15, gt=0)
    lowest_recall: int = Field(default=10, gt=0)

    @field_validator("ks", "ddx_ks")
    @classmethod
    def _positive(cls, ks: list[int]) -> list[int]:
        if any(k < 1 for k in ks):
            raise ValueError("k values must be >= 1")
        return ks


CalibrationMethod = Literal["none", "temperature", "isotonic"]


class PartialEvidenceConfig(_StrictModel):
    extra_evidences: int = Field(default=3, ge=0)


class Top3Floor(_StrictModel):
    """Minimum held-out top-3 accuracy, on full and on partial evidence."""

    full: float = Field(ge=0, le=1)
    partial: float = Field(ge=0, le=1)


CalibrationVariant = Literal["baseline", "dropout"]
CalibrationRows = Literal["full", "partial"]


class DropoutConfig(_StrictModel):
    models: list[ModelName] = Field(min_length=1)
    rate_range: tuple[float, float]
    copies: int = Field(default=1, ge=0)
    keep_original: bool = True
    experiment: str = "beaconai-phase3b"

    @field_validator("rate_range")
    @classmethod
    def _range_inside_unit_interval(cls, bounds: tuple[float, float]) -> tuple[float, float]:
        low, high = bounds
        if not 0 < low <= high < 1:
            raise ValueError("rate_range must satisfy 0 < low <= high < 1")
        return bounds

    @model_validator(mode="after")
    def _has_rows(self) -> DropoutConfig:
        if self.copies == 0 and not self.keep_original:
            raise ValueError("copies=0 with keep_original=false leaves no training rows")
        return self


class FinalEvalConfig(_StrictModel):
    # A safe file stem: the lock is reports/test_locks/<name>.json.
    name: str = Field(default="classifier_final", pattern=r"^[A-Za-z0-9_-]+$")
    experiment: str = "beaconai-final"


class CalibrationConfig(_StrictModel):
    models: list[ModelName] = Field(min_length=1)
    methods: list[CalibrationMethod] = Field(min_length=1)
    fit_fraction: float = Field(default=0.5, gt=0, lt=1)
    ece_bins: int = Field(default=15, ge=2)
    temperature_bounds: tuple[float, float] = (0.05, 20.0)
    top3_floor: Top3Floor
    experiment: str | None = None
    partial_evidence: PartialEvidenceConfig = PartialEvidenceConfig()
    variants: list[CalibrationVariant] = Field(default_factory=lambda: ["baseline"], min_length=1)
    fit_on: list[CalibrationRows] = Field(default_factory=lambda: ["full"], min_length=1)
    evidence_curve: list[int] = Field(default_factory=list)

    @field_validator("evidence_curve")
    @classmethod
    def _non_negative(cls, values: list[int]) -> list[int]:
        if any(v < 0 for v in values):
            raise ValueError("evidence_curve values must be >= 0")
        return values

    @field_validator("temperature_bounds")
    @classmethod
    def _bounds_increasing(cls, bounds: tuple[float, float]) -> tuple[float, float]:
        low, high = bounds
        if not 0 < low < high:
            raise ValueError("temperature_bounds must satisfy 0 < low < high")
        return bounds


class AbstentionConfig(_StrictModel):
    threshold: float = Field(ge=0, le=1)
    target_accuracy: float = Field(default=0.95, gt=0, le=1)
    differential_size: int = Field(default=3, ge=1)
    curve_points: int = Field(default=101, ge=2)
    # Evidence level the threshold suggestion is computed on: patients who keep the initial
    # evidence plus this many others. None means the partial-evidence set (initial plus
    # calibration.partial_evidence.extra_evidences).
    suggest_on_extra: int | None = Field(default=None, ge=0)


class ExplainConfig(_StrictModel):
    top_positive: int = Field(default=5, ge=1)
    top_negative: int = Field(default=3, ge=0)
    global_top: int = Field(default=10, ge=1)
    global_conditions: list[str]
    min_prevalence: float = Field(default=0.01, ge=0, le=1)
    # Conditions below this probability are treated as not shown when measuring ranking changes.
    ranking_min_prob: float = Field(default=0.01, ge=0, le=1)
    latency_budget_ms: float = Field(default=50, gt=0)
    latency_calls: int = Field(default=300, ge=1)
    experiment: str = "beaconai-phase4"

    @field_validator("global_conditions")
    @classmethod
    def _six_to_eight_unique(cls, names: list[str]) -> list[str]:
        if not 6 <= len(names) <= 8:
            raise ValueError("global_conditions must list 6 to 8 conditions")
        if len(set(names)) != len(names):
            raise ValueError("global_conditions must be unique")
        return names


class Config(_StrictModel):
    seed: int = Field(default=42, ge=0)
    data: DataConfig
    eda: EdaConfig = EdaConfig()
    # Training sections are optional so data-only configs stay valid.
    mlflow: MlflowConfig | None = None
    train: TrainConfig | None = None
    models: ModelsConfig | None = None
    eval: EvalConfig | None = None
    calibration: CalibrationConfig | None = None
    abstention: AbstentionConfig | None = None
    dropout: DropoutConfig | None = None
    final_eval: FinalEvalConfig = FinalEvalConfig()
    explain: ExplainConfig | None = None

    @model_validator(mode="after")
    def _selection_metric_is_computed(self) -> Config:
        if self.train is None or self.eval is None:
            return self
        known = {f"top{k}_accuracy" for k in self.eval.ks} | {"macro_f1"}
        for k in self.eval.ddx_ks:
            known |= {f"ddx_precision_at{k}", f"ddx_recall_at{k}", f"ddx_recall_norm_at{k}"}
        if self.train.selection_metric not in known:
            raise ValueError(
                f"train.selection_metric {self.train.selection_metric!r} is not computed; "
                f"choose one of {sorted(known)}"
            )
        unknown = [m for m in self.train.tie_breakers if m not in known]
        if unknown:
            raise ValueError(f"train.tie_breakers {unknown} are not computed")
        return self


def project_path(path: Path) -> Path:
    """Resolve a config path: relative paths are relative to the project root."""
    return path if path.is_absolute() else PROJECT_ROOT / path


def require_training(
    cfg: Config,
) -> tuple[MlflowConfig, TrainConfig, ModelsConfig, EvalConfig]:
    """Return the training sections, failing clearly if the config lacks any."""
    missing = [n for n in ("mlflow", "train", "models", "eval") if getattr(cfg, n) is None]
    if missing:
        raise ValueError(f"config is missing sections needed for training: {missing}")
    assert cfg.mlflow and cfg.train and cfg.models and cfg.eval
    return cfg.mlflow, cfg.train, cfg.models, cfg.eval


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


def require_calibration(cfg: Config) -> tuple[CalibrationConfig, AbstentionConfig]:
    """Return the calibration sections, failing clearly if the config lacks any."""
    missing = [n for n in ("calibration", "abstention") if getattr(cfg, n) is None]
    if missing:
        raise ValueError(f"config is missing sections needed for calibration: {missing}")
    assert cfg.calibration and cfg.abstention
    return cfg.calibration, cfg.abstention


def require_dropout(cfg: Config) -> DropoutConfig:
    if cfg.dropout is None:
        raise ValueError("config is missing the dropout section")
    return cfg.dropout


def require_explain(cfg: Config) -> ExplainConfig:
    if cfg.explain is None:
        raise ValueError("config is missing the explain section")
    return cfg.explain
