"""One interface for every classifier: fit, predict_proba, save, load."""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, ClassVar, Self

import numpy as np
from scipy import sparse

Matrix = sparse.csr_matrix | np.ndarray
META_FILE = "meta.json"


def _as_labels(y: np.ndarray, n_classes: int) -> np.ndarray:
    y = np.asarray(y).astype(np.int64).ravel()
    if len(y) and (y.min() < 0 or y.max() >= n_classes):
        raise ValueError(f"labels must be in [0, {n_classes})")
    return y


class ModelWrapper(ABC):
    name: ClassVar[str]

    def __init__(self, params: dict[str, Any], n_classes: int, seed: int) -> None:
        self.params = dict(params)
        self.n_classes = n_classes
        self.seed = seed
        self.n_features: int | None = None

    def fit(
        self,
        X: Matrix,
        y: np.ndarray,
        X_val: Matrix | None = None,
        y_val: np.ndarray | None = None,
    ) -> Self:
        """Train. X_val/y_val are used only for early stopping, never for the fit itself."""
        y = _as_labels(y, self.n_classes)
        if X.shape[0] != len(y):
            raise ValueError(f"{X.shape[0]} rows but {len(y)} labels")
        if (X_val is None) != (y_val is None):
            raise ValueError("X_val and y_val must be given together")
        yv = None if y_val is None else _as_labels(y_val, self.n_classes)
        self.n_features = int(X.shape[1])
        self._fit(X, y, X_val, yv)
        return self

    def predict_proba(self, X: Matrix) -> np.ndarray:
        """Probabilities of shape (n_patients, n_classes), columns in labels.json order."""
        if self.n_features is None:
            raise RuntimeError(f"{self.name} model is not fitted")
        if X.shape[1] != self.n_features:
            raise ValueError(f"expected {self.n_features} features, got {X.shape[1]}")
        return np.asarray(self._predict_proba(X), dtype=np.float64)

    def predict(self, X: Matrix) -> np.ndarray:
        return self.predict_proba(X).argmax(axis=1)

    def save(self, path: Path) -> None:
        if self.n_features is None:
            raise RuntimeError("cannot save an unfitted model")
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        meta = {
            "name": self.name,
            "params": self.params,
            "n_classes": self.n_classes,
            "seed": self.seed,
            "n_features": self.n_features,
        }
        (path / META_FILE).write_text(json.dumps(meta, indent=2), encoding="utf-8")
        self._save(path)

    @staticmethod
    def load(path: Path) -> ModelWrapper:
        """Load a model saved by `save`. Files may be pickles: only load trusted directories."""
        from beaconai.models.registry import MODEL_REGISTRY

        path = Path(path)
        meta = json.loads((path / META_FILE).read_text(encoding="utf-8"))
        if meta.get("name") not in MODEL_REGISTRY:
            raise ValueError(f"Unknown model name in {path / META_FILE}: {meta.get('name')!r}")
        model = MODEL_REGISTRY[meta["name"]](meta["params"], meta["n_classes"], meta["seed"])
        model.n_features = meta["n_features"]
        model._load(path)
        return model

    @abstractmethod
    def _fit(
        self, X: Matrix, y: np.ndarray, X_val: Matrix | None, y_val: np.ndarray | None
    ) -> None: ...

    @abstractmethod
    def _predict_proba(self, X: Matrix) -> np.ndarray: ...

    @abstractmethod
    def _save(self, path: Path) -> None: ...

    @abstractmethod
    def _load(self, path: Path) -> None: ...
