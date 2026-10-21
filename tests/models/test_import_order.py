"""macOS: torch and XGBoost must share one OpenMP runtime, or threaded training crashes."""

import os
import subprocess
import sys
from pathlib import Path

import pytest

from beaconai.models.openmp import ENV_VAR, shared_openmp_ok, torch_lib_dir

TRAIN_BOTH = """
import beaconai.models
import numpy as np
import torch
import xgboost

d = xgboost.DMatrix(np.random.rand(3000, 100), label=np.arange(3000) % 3)
params = {"objective": "multi:softprob", "num_class": 3}
xgboost.train(params, d, 10)
X, y = torch.rand(3000, 600), torch.arange(3000) % 3
net = torch.nn.Sequential(torch.nn.Linear(600, 256), torch.nn.ReLU(), torch.nn.Linear(256, 3))
opt = torch.optim.Adam(net.parameters())
for _ in range(5):
    opt.zero_grad()
    torch.nn.functional.cross_entropy(net(X), y).backward()
    opt.step()
xgboost.train(params, d, 10)
print("ok")
"""


def test_non_mac_needs_nothing() -> None:
    assert shared_openmp_ok("linux", {}, Path("/x/torch/lib"))


def test_mac_requires_torch_lib_on_dyld_path() -> None:
    lib = Path("/x/torch/lib")
    assert not shared_openmp_ok("darwin", {}, lib)
    assert not shared_openmp_ok("darwin", {ENV_VAR: "/other"}, lib)
    assert shared_openmp_ok("darwin", {ENV_VAR: f"/other{os.pathsep}{lib}"}, lib)


def test_torch_lib_dir_contains_torch_libraries() -> None:
    assert torch_lib_dir().name == "lib"
    assert torch_lib_dir().parent.name == "torch"


@pytest.mark.skipif(sys.platform != "darwin", reason="macOS-only OpenMP clash")
def test_import_fails_fast_without_shared_runtime() -> None:
    env = {k: v for k, v in os.environ.items() if k != ENV_VAR}
    result = subprocess.run(
        [sys.executable, "-c", "import beaconai.models"], env=env, capture_output=True, timeout=60
    )
    assert result.returncode != 0
    assert ENV_VAR in result.stderr.decode()


def test_torch_and_xgboost_train_in_one_process() -> None:
    env = {**os.environ}
    if sys.platform == "darwin":
        env[ENV_VAR] = str(torch_lib_dir())
    result = subprocess.run(
        [sys.executable, "-c", TRAIN_BOTH], env=env, capture_output=True, timeout=180
    )
    assert result.returncode == 0, result.stderr.decode()[-2000:]
    assert result.stdout.decode().strip().endswith("ok")
