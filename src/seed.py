"""Deterministic seeding across the libraries this project uses."""
from __future__ import annotations

import os
import random

import numpy as np


def set_global_seed(seed: int) -> None:
    """Seed Python's random module, NumPy, and (if installed) TensorFlow. Does not train anything."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import tensorflow as tf  # optional; Phase 2.5 never trains a model, but Phase 3 will reuse this
    except ImportError:
        return
    tf.random.set_seed(seed)
