"""Robustness metrics over sampled opponent responses."""

from __future__ import annotations

import numpy as np


def robustness_score(values: list[float] | np.ndarray) -> float:
    """Return a conservative score combining mean, worst case, and variance."""

    array = np.asarray(values, dtype=np.float32)
    if array.size == 0:
        raise ValueError("values must be non-empty")
    return float(array.min() - 0.25 * array.std())


def robustness_score_g(worst_case_value: float, average_case_value: float, alpha: float = 0.5) -> float:
    """Proposal robustness score ``G = alpha V(sigma, BR) + (1-alpha) E[V]``."""

    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be in [0, 1]")
    return float(alpha) * float(worst_case_value) + (1.0 - float(alpha)) * float(average_case_value)
