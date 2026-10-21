import random

import numpy as np
import torch

from beaconai.models.seed import set_global_seed


def _draw() -> tuple[float, float, float]:
    return random.random(), float(np.random.rand()), float(torch.rand(1))


def test_set_global_seed_makes_all_generators_repeat() -> None:
    set_global_seed(7)
    first = _draw()
    set_global_seed(7)
    assert _draw() == first
