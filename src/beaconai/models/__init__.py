"""Model training and the shared model wrapper."""

# torch and XGBoost each bundle an OpenMP runtime. On macOS, loading torch first makes
# XGBoost segfault, so XGBoost is loaded as soon as this package is imported.
import xgboost  # noqa: F401
