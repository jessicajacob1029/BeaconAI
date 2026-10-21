.PHONY: help setup data eda train train-final eval eval-val test lint format serve ui mlflow

# macOS: torch and XGBoost must share torch's OpenMP runtime (see src/beaconai/models/openmp.py).
ifeq ($(shell uname -s),Darwin)
export DYLD_LIBRARY_PATH := $(CURDIR)/.venv/lib/python3.11/site-packages/torch/lib
endif

help:  ## List available commands
	@grep -E '^[a-z]+:.*##' $(MAKEFILE_LIST) | awk -F':.*## ' '{printf "  %-8s %s\n", $$1, $$2}'

setup:  ## Install dependencies and git hooks
	uv sync
	uv run pre-commit install

lint:  ## Ruff lint + format check
	uv run ruff check .
	uv run ruff format --check .

format:  ## Auto-fix lint issues and format code
	uv run ruff check --fix .
	uv run ruff format .

test:  ## Run the test suite
	uv run pytest

data:  ## Download + preprocess DDXPlus (dvc repro)
	uv run dvc repro

eda:  ## Save EDA plots for processed data into reports/
	uv run python -m beaconai.data.eda

train:  ## Grid-search all models on validation, log to MLflow, register the best
	uv run python -m beaconai.models.train

train-final:  ## Retrain only the winning params from the last grid search (full data)
	uv run python -m beaconai.models.train --winners-only

eval:  ## Score test ONCE on full data and write reports/model_comparison.md
	uv run python -m beaconai.eval.evaluate --split test

eval-val:  ## Validation-only report, safe to run while iterating
	uv run python -m beaconai.eval.evaluate --split validate

serve:  ## FastAPI on :8000 (Phase 5)
	@echo "make serve: not implemented until Phase 5" && exit 1

ui:  ## Streamlit on :8501 (Phase 10)
	@echo "make ui: not implemented until Phase 10" && exit 1

mlflow:  ## MLflow UI on :5000 (backend path matches mlflow.tracking_db in config)
	uv run mlflow ui --backend-store-uri sqlite:///mlruns/mlflow.db --port 5000
