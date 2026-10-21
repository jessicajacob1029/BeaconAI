"""Model name -> wrapper class."""

from __future__ import annotations

from typing import Any

from beaconai.models.base import ModelWrapper
from beaconai.models.logreg import LogRegModel
from beaconai.models.xgb import XGBoostModel

MODEL_REGISTRY: dict[str, type[ModelWrapper]] = {
    LogRegModel.name: LogRegModel,
    XGBoostModel.name: XGBoostModel,
}


def build_model(name: str, params: dict[str, Any], n_classes: int, seed: int) -> ModelWrapper:
    if name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model {name!r}; known: {sorted(MODEL_REGISTRY)}")
    return MODEL_REGISTRY[name](params, n_classes, seed)
