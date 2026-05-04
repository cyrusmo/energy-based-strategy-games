"""Robustness metrics over sampled opponent responses."""

from __future__ import annotations

import numpy as np


def robustness_score(values: list[float] | np.ndarray) -> float:
    """Return a conservative score combining mean, worst case, and variance."""

    array = np.asarray(values, dtype=np.float32)
    if array.size == 0:
        raise ValueError("values must be non-empty")
    return float(array.min() - 0.25 * array.std())
