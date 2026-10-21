"""XGBoost multi-class model (native API, so a class missing from training is fine)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import xgboost as xgb

from beaconai.models.base import Matrix, ModelWrapper

MODEL_FILE = "model.ubj"
EXTRA_FILE = "xgb.json"


class XGBoostModel(ModelWrapper):
    name = "xgboost"

    def _booster_params(self) -> tuple[dict[str, Any], int, int | None]:
        p = dict(self.params)
        n_rounds = int(p.pop("n_estimators"))
        early = p.pop("early_stopping_rounds", None)
        booster = {
            "objective": "multi:softprob",
            "num_class": self.n_classes,
            "eval_metric": "mlogloss",
            "seed": self.seed,
            **p,
        }
        return booster, n_rounds, early

    def _fit(
        self, X: Matrix, y: np.ndarray, X_val: Matrix | None, y_val: np.ndarray | None
    ) -> None:
        params, n_rounds, early = self._booster_params()
        evals = [] if X_val is None else [(xgb.DMatrix(X_val, label=y_val), "validate")]
        use_early = bool(evals) and early is not None
        self._booster = xgb.train(
            params,
            xgb.DMatrix(X, label=y),
            num_boost_round=n_rounds,
            evals=evals,
            early_stopping_rounds=early if use_early else None,
            verbose_eval=False,
        )
        self.best_iteration = int(self._booster.best_iteration) if use_early else n_rounds - 1

    def _predict_proba(self, X: Matrix) -> np.ndarray:
        return self._booster.predict(xgb.DMatrix(X), iteration_range=(0, self.best_iteration + 1))

    def _save(self, path: Path) -> None:
        self._booster.save_model(path / MODEL_FILE)
        extra = {"best_iteration": self.best_iteration}
        (path / EXTRA_FILE).write_text(json.dumps(extra), encoding="utf-8")

    def _load(self, path: Path) -> None:
        self._booster = xgb.Booster()
        self._booster.load_model(path / MODEL_FILE)
        extra = json.loads((path / EXTRA_FILE).read_text(encoding="utf-8"))
        self.best_iteration = int(extra["best_iteration"])
