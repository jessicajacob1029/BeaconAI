"""Post-hoc probability calibrators. State is plain JSON, so nothing is unpickled at load time."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Any, ClassVar, Self

import numpy as np
from scipy.optimize import minimize_scalar
from sklearn.isotonic import IsotonicRegression

EPS = 1e-12


def _softmax(z: np.ndarray) -> np.ndarray:
    e = np.exp(z - z.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


def _log(proba: np.ndarray) -> np.ndarray:
    return np.log(np.clip(np.asarray(proba, dtype=np.float64), EPS, 1.0))


def _nll(y: np.ndarray, proba: np.ndarray) -> float:
    return float(-np.mean(np.log(np.clip(proba[np.arange(len(y)), y], EPS, 1.0))))


class Calibrator(ABC):
    method: ClassVar[str]

    @abstractmethod
    def fit(self, proba: np.ndarray, y: Sequence[int] | np.ndarray) -> Self: ...

    @abstractmethod
    def transform(self, proba: np.ndarray) -> np.ndarray: ...

    @abstractmethod
    def to_dict(self) -> dict[str, Any]: ...


class IdentityCalibrator(Calibrator):
    """Raw probabilities, kept as a candidate so calibration has to beat doing nothing."""

    method = "none"

    def fit(self, proba: np.ndarray, y: Sequence[int] | np.ndarray) -> Self:
        return self

    def transform(self, proba: np.ndarray) -> np.ndarray:
        return np.asarray(proba, dtype=np.float64).copy()

    def to_dict(self) -> dict[str, Any]:
        return {"method": self.method}


class TemperatureCalibrator(Calibrator):
    """Divide log-probabilities by one temperature T fit by NLL. Never changes the ranking."""

    method = "temperature"

    def __init__(self, bounds: tuple[float, float], temperature: float | None = None) -> None:
        self.bounds = bounds
        self.temperature = temperature

    def fit(self, proba: np.ndarray, y: Sequence[int] | np.ndarray) -> Self:
        logits, labels = _log(proba), np.asarray(y, dtype=np.int64)
        result = minimize_scalar(
            lambda t: _nll(labels, _softmax(logits / t)), bounds=self.bounds, method="bounded"
        )
        self.temperature = float(result.x)
        return self

    def transform(self, proba: np.ndarray) -> np.ndarray:
        if self.temperature is None:
            raise RuntimeError("temperature calibrator is not fitted")
        return _softmax(_log(proba) / self.temperature)

    def to_dict(self) -> dict[str, Any]:
        return {"method": self.method, "temperature": self.temperature, "bounds": list(self.bounds)}


class IsotonicCalibrator(Calibrator):
    """One increasing step function per class (one-vs-rest), then rows renormalised.

    Can change the ranking, which is why champion selection applies top-3 accuracy floors.
    """

    method = "isotonic"

    def __init__(
        self, x: list[list[float]] | None = None, y: list[list[float]] | None = None
    ) -> None:
        self.x, self.y = x, y

    def fit(self, proba: np.ndarray, y: Sequence[int] | np.ndarray) -> Self:
        p, labels = np.asarray(proba, dtype=np.float64), np.asarray(y, dtype=np.int64)
        self.x, self.y = [], []
        for c in range(p.shape[1]):
            iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
            iso.fit(p[:, c], (labels == c).astype(np.float64))
            self.x.append(iso.X_thresholds_.tolist())
            self.y.append(iso.y_thresholds_.tolist())
        return self

    def transform(self, proba: np.ndarray) -> np.ndarray:
        if self.x is None or self.y is None:
            raise RuntimeError("isotonic calibrator is not fitted")
        p = np.asarray(proba, dtype=np.float64)
        q = np.column_stack([np.interp(p[:, c], self.x[c], self.y[c]) for c in range(p.shape[1])])
        totals = q.sum(axis=1, keepdims=True)
        # A row where every class maps to 0 keeps its uncalibrated probabilities.
        return np.where(totals > 0, q / np.where(totals > 0, totals, 1.0), p)

    def to_dict(self) -> dict[str, Any]:
        return {"method": self.method, "x": self.x, "y": self.y}


def make_calibrator(method: str, bounds: tuple[float, float]) -> Calibrator:
    if method == "none":
        return IdentityCalibrator()
    if method == "temperature":
        return TemperatureCalibrator(bounds)
    if method == "isotonic":
        return IsotonicCalibrator()
    raise ValueError(f"Unknown calibration method {method!r}")


def calibrator_from_dict(state: dict[str, Any]) -> Calibrator:
    method = state.get("method")
    if method == "none":
        return IdentityCalibrator()
    if method == "temperature":
        low, high = state["bounds"]
        return TemperatureCalibrator((float(low), float(high)), state["temperature"])
    if method == "isotonic":
        return IsotonicCalibrator(state["x"], state["y"])
    raise ValueError(f"Unknown calibration method {method!r}")
