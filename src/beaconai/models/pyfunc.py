"""Expose any ModelWrapper as an MLflow pyfunc model, so the registry can serve it later."""

from __future__ import annotations

from typing import Any

import mlflow.pyfunc
import numpy as np
import pandas as pd
from mlflow.types.type_hints import TypeFromExample
from scipy import sparse

from beaconai.models.base import ModelWrapper

ARTIFACT_KEY = "model_dir"


class BeaconPyfunc(mlflow.pyfunc.PythonModel):
    def load_context(self, context: mlflow.pyfunc.PythonModelContext) -> None:
        self.model = ModelWrapper.load(context.artifacts[ARTIFACT_KEY])

    # TypeFromExample: MLflow takes the input schema from the logged input example.
    def predict(self, context: Any, model_input: TypeFromExample, params: Any = None) -> np.ndarray:
        """Feature rows in, (n_patients, n_classes) probabilities out."""
        if isinstance(model_input, pd.DataFrame):
            model_input = model_input.to_numpy(dtype=np.float32)
        if not sparse.issparse(model_input):
            model_input = sparse.csr_matrix(np.asarray(model_input, dtype=np.float32))
        return self.model.predict_proba(model_input)
