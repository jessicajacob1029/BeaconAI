.PHONY: help setup data train eval test lint format serve ui mlflow

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

data:  ## Download + preprocess DDXPlus (Phase 1)
	@echo "make data: not implemented until Phase 1" && exit 1

train:  ## Train models, log to MLflow (Phase 2)
	@echo "make train: not implemented until Phase 2" && exit 1

eval:  ## Metrics + plots into reports/ (Phase 2)
	@echo "make eval: not implemented until Phase 2" && exit 1

serve:  ## FastAPI on :8000 (Phase 5)
	@echo "make serve: not implemented until Phase 5" && exit 1

ui:  ## Streamlit on :8501 (Phase 10)
	@echo "make ui: not implemented until Phase 10" && exit 1

mlflow:  ## MLflow UI on :5000 (Phase 2)
	@echo "make mlflow: not implemented until Phase 2" && exit 1
