"""Multinomial logistic regression baseline."""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression

from beaconai.models.base import Matrix, ModelWrapper

MODEL_FILE = "model.joblib"


class LogRegModel(ModelWrapper):
    name = "logreg"

    def _fit(
        self, X: Matrix, y: np.ndarray, X_val: Matrix | None, y_val: np.ndarray | None
    ) -> None:
        # No early stopping, so the validation split is unused here; the grid search uses it.
        self._clf = LogisticRegression(random_state=self.seed, **self.params).fit(X, y)

    def _predict_proba(self, X: Matrix) -> np.ndarray:
        # Classes missing from the training rows get probability 0 in their fixed column.
        out = np.zeros((X.shape[0], self.n_classes))
        out[:, self._clf.classes_] = self._clf.predict_proba(X)
        return out

    def _save(self, path: Path) -> None:
        joblib.dump(self._clf, path / MODEL_FILE)

    def _load(self, path: Path) -> None:
        # joblib is pickle: only files this project wrote (local MLflow store) are loaded.
        self._clf = joblib.load(path / MODEL_FILE)

    def linear_parameters(self) -> tuple[np.ndarray, np.ndarray]:
        """(W, b) with logits = W @ x + b and probabilities = softmax(logits), in label order."""
        classes = np.asarray(self._clf.classes_)
        if len(classes) != self.n_classes:
            missing = sorted(set(range(self.n_classes)) - set(classes.tolist()))
            raise ValueError(f"classes {missing} were missing in training; no logits for them")
        order = np.argsort(classes)
        W = self._clf.coef_[order].astype(np.float64)
        b = self._clf.intercept_[order].astype(np.float64)
        return W, b
