"""On macOS, torch and XGBoost each ship an OpenMP runtime; loading torch first crashes XGBoost."""

import subprocess
import sys

SCRIPT = """
import beaconai.models  # must fix the load order before torch is imported
import numpy as np
import torch
import xgboost

d = xgboost.DMatrix(np.random.rand(50, 3), label=np.arange(50) % 2)
xgboost.train({"objective": "binary:logistic"}, d, 2)
print(torch.ones(1).item())
"""


def test_torch_then_xgboost_does_not_crash() -> None:
    result = subprocess.run([sys.executable, "-c", SCRIPT], capture_output=True, timeout=120)
    assert result.returncode == 0, result.stderr.decode()[-2000:]
