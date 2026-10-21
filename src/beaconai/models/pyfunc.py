"""MLflow pyfunc adapters, so the registry can serve raw and calibrated models."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import mlflow.pyfunc
import numpy as np
import pandas as pd
from mlflow.models import infer_signature
from mlflow.types.type_hints import TypeFromExample
from scipy import sparse

from beaconai.models.base import ModelWrapper
from beaconai.models.calibrated import CalibratedClassifier

ARTIFACT_KEY = "model_dir"
PIP_REQUIREMENTS = ["numpy", "scipy", "pandas", "scikit-learn", "xgboost", "torch", "pydantic"]


def to_csr(model_input: Any) -> sparse.csr_matrix:
    """DataFrame, array, or sparse matrix in; float32 CSR out."""
    if isinstance(model_input, pd.DataFrame):
        model_input = model_input.to_numpy(dtype=np.float32)
    if sparse.issparse(model_input):
        return sparse.csr_matrix(model_input)
    return sparse.csr_matrix(np.asarray(model_input, dtype=np.float32))


class BeaconPyfunc(mlflow.pyfunc.PythonModel):
    """A raw Phase 2 model: features in, uncalibrated probabilities out."""

    def load_context(self, context: mlflow.pyfunc.PythonModelContext) -> None:
        self.model = ModelWrapper.load(context.artifacts[ARTIFACT_KEY])

    # TypeFromExample: MLflow takes the input schema from the logged input example.
    def predict(self, context: Any, model_input: TypeFromExample, params: Any = None) -> np.ndarray:
        return self.model.predict_proba(to_csr(model_input))


class CalibratedPyfunc(mlflow.pyfunc.PythonModel):
    """A CalibratedClassifier: features in, calibrated probabilities out."""

    def load_context(self, context: mlflow.pyfunc.PythonModelContext) -> None:
        self.model = CalibratedClassifier.load(Path(context.artifacts[ARTIFACT_KEY]))

    def predict(self, context: Any, model_input: TypeFromExample, params: Any = None) -> np.ndarray:
        return self.model.predict_proba(to_csr(model_input))


def log_pyfunc(
    python_model: mlflow.pyfunc.PythonModel,
    model_dir: Path,
    example: pd.DataFrame,
    output: np.ndarray,
) -> str:
    """Log a saved model directory as an MLflow pyfunc in the active run; return its URI."""
    info = mlflow.pyfunc.log_model(
        name="model",
        python_model=python_model,
        artifacts={ARTIFACT_KEY: str(model_dir)},
        pip_requirements=PIP_REQUIREMENTS,
        signature=infer_signature(example, output),
        input_example=example,
    )
    return info.model_uri
