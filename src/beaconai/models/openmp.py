"""macOS only: make torch and XGBoost share one OpenMP runtime.

torch ships its own libomp, while the XGBoost wheel loads Homebrew's. With two runtimes in one
process, threaded training crashes or hangs. Putting torch's lib folder on DYLD_LIBRARY_PATH
makes XGBoost (and scikit-learn) load torch's copy instead. The variable must be set before
Python starts, so the Makefile sets it and this module fails fast with the fix when it is not.
Linux (CI, Docker) is not affected.
"""

from __future__ import annotations

import importlib.util
import os
import sys
from collections.abc import Mapping
from pathlib import Path

ENV_VAR = "DYLD_LIBRARY_PATH"


def torch_lib_dir() -> Path:
    """torch's bundled library folder, found without importing torch."""
    spec = importlib.util.find_spec("torch")
    if spec is None or spec.origin is None:
        raise ModuleNotFoundError("torch is not installed; run `make setup`")
    return Path(spec.origin).parent / "lib"


def shared_openmp_ok(platform: str, env: Mapping[str, str], lib_dir: Path) -> bool:
    if platform != "darwin":
        return True
    return str(lib_dir) in env.get(ENV_VAR, "").split(os.pathsep)


def check_shared_openmp() -> None:
    lib_dir = torch_lib_dir()
    if not shared_openmp_ok(sys.platform, os.environ, lib_dir):
        raise RuntimeError(
            "On macOS, torch and XGBoost must share one OpenMP runtime or training crashes. "
            f"Run through make (make test, make train), or first run: export {ENV_VAR}={lib_dir}"
        )
