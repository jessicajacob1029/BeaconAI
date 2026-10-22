import pytest
import yaml
from pydantic import ValidationError

from beaconai.config import DEFAULT_CONFIG_PATH, Config, ExplainConfig, load_config, require_explain


def _raw() -> dict:
    return yaml.safe_load(DEFAULT_CONFIG_PATH.read_text(encoding="utf-8"))


def test_base_config_explain_section() -> None:
    ex = require_explain(load_config())
    assert (ex.top_positive, ex.top_negative, ex.global_top) == (5, 3, 10)
    assert 6 <= len(ex.global_conditions) <= 8
    assert ex.latency_budget_ms == 50


@pytest.mark.parametrize("n", [5, 9])
def test_global_condition_count(n: int) -> None:
    raw = _raw()["explain"]
    raw["global_conditions"] = [f"C{i}" for i in range(n)]
    with pytest.raises(ValidationError, match="6 to 8"):
        ExplainConfig.model_validate(raw)


def test_global_conditions_unique() -> None:
    raw = _raw()["explain"]
    raw["global_conditions"] = ["A", "A", "B", "C", "D", "E"]
    with pytest.raises(ValidationError, match="unique"):
        ExplainConfig.model_validate(raw)


def test_require_explain_missing() -> None:
    raw = _raw()
    del raw["explain"]
    with pytest.raises(ValueError, match="explain"):
        require_explain(Config.model_validate(raw))
