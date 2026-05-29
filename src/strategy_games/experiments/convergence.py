"""Convergence and confidence-interval utilities for experiment summaries."""

from __future__ import annotations

import math
import random
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class ConvergenceTargets:
    """Simple scalar goals for deciding whether a run has met its objective."""

    goal_rate_target: float = 1.0
    return_target: float = 0.9
    ebm_energy_gap_min: float = 0.0


def detect_convergence(
    history: Sequence[Mapping[str, Any]],
    metric: str = "goal_rate",
    target: float = 1.0,
    window: int = 3,
    patience: int = 1,
    min_iter: int = 1,
    tolerance: float = 1e-6,
) -> dict[str, float | int | bool | str]:
    """Detect when a metric reaches a target and stops trending materially.

    The metric is resolved from each iteration's top-level keys first, then its
    ``rollout`` mapping, then ``updates``, then ``selection_metrics``. This makes
    the helper usable with both current training-loop histories and dashboard
    fixtures.
    """

    if window < 1:
        raise ValueError("window must be positive")
    if patience < 1:
        raise ValueError("patience must be positive")
    values = [_extract_metric(item, metric) for item in history]
    numeric = [(idx, value) for idx, value in enumerate(values) if value is not None]
    reached_count = 0
    reached_at: int | None = None
    trend_slope = 0.0
    for idx, value in numeric:
        if idx + 1 < min_iter:
            reached_count = 0
            continue
        if value >= target - tolerance:
            reached_count += 1
            if reached_at is None:
                reached_at = idx
        else:
            reached_count = 0
            reached_at = None
        if reached_count >= patience:
            window_values = [item[1] for item in numeric[max(0, idx - window + 1) : idx + 1]]
            trend_slope = _slope(window_values)
            return {
                "metric": metric,
                "target": float(target),
                "converged": abs(trend_slope) <= tolerance or value >= target - tolerance,
                "iteration_reached": int(reached_at if reached_at is not None else idx),
                "final_value": float(value),
                "trend_slope": float(trend_slope),
            }
    final_value = numeric[-1][1] if numeric else 0.0
    if numeric:
        trend_slope = _slope([item[1] for item in numeric[-window:]])
    return {
        "metric": metric,
        "target": float(target),
        "converged": False,
        "iteration_reached": -1,
        "final_value": float(final_value),
        "trend_slope": float(trend_slope),
    }


def summarize_convergence(
    history: Sequence[Mapping[str, Any]],
    targets: ConvergenceTargets | None = None,
) -> dict[str, dict[str, float | int | bool | str]]:
    """Return convergence checks for the default public goal metrics."""

    targets = targets or ConvergenceTargets()
    return {
        "goal_rate": detect_convergence(history, "goal_rate", targets.goal_rate_target),
        "episode_return": detect_convergence(history, "episode_return", targets.return_target),
        "ebm_energy_gap": detect_convergence(history, "ebm_energy_gap", targets.ebm_energy_gap_min),
    }


def multiseed_confidence(
    values: Sequence[float],
    confidence: float = 0.95,
    bootstrap_samples: int = 0,
    seed: int = 0,
) -> dict[str, float | int]:
    """Return mean/std plus a confidence interval for small seed sweeps."""

    numeric = [float(value) for value in values]
    n = len(numeric)
    if n == 0:
        return {"n": 0, "mean": 0.0, "std": 0.0, "ci_low": 0.0, "ci_high": 0.0}
    mean = float(np.mean(numeric))
    std = float(np.std(numeric))
    if n == 1:
        return {"n": 1, "mean": mean, "std": 0.0, "ci_low": mean, "ci_high": mean}
    if bootstrap_samples > 1:
        return _bootstrap_interval(numeric, confidence=confidence, samples=bootstrap_samples, seed=seed)
    critical = _normal_or_t_critical(n, confidence)
    margin = critical * float(np.std(numeric, ddof=1)) / math.sqrt(n)
    return {"n": n, "mean": mean, "std": std, "ci_low": mean - margin, "ci_high": mean + margin}


def energy_gap(update_metrics: Mapping[str, Any]) -> float | None:
    """Return negative-minus-positive EBM energy gap when both energies exist."""

    if "positive_energy" not in update_metrics or "negative_energy" not in update_metrics:
        return None
    return float(update_metrics["negative_energy"]) - float(update_metrics["positive_energy"])


def _extract_metric(item: Mapping[str, Any], metric: str) -> float | None:
    if metric == "ebm_energy_gap":
        updates = item.get("updates", {})
        if isinstance(updates, Mapping):
            return energy_gap(updates)
    for container in (item, item.get("rollout", {}), item.get("updates", {}), item.get("selection_metrics", {})):
        if isinstance(container, Mapping) and metric in container:
            return float(container[metric])
    return None


def _slope(values: Sequence[float]) -> float:
    if len(values) < 2:
        return 0.0
    x = np.arange(len(values), dtype=float)
    y = np.asarray(values, dtype=float)
    return float(np.polyfit(x, y, deg=1)[0])


def _normal_or_t_critical(n: int, confidence: float) -> float:
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be in (0, 1)")
    # Common two-sided 95% t critical values for the small seed counts used here.
    if abs(confidence - 0.95) < 1e-12:
        return {
            2: 12.706,
            3: 4.303,
            4: 3.182,
            5: 2.776,
            6: 2.571,
            7: 2.447,
            8: 2.365,
            9: 2.306,
            10: 2.262,
        }.get(n, 1.96)
    return 1.96


def _bootstrap_interval(values: Sequence[float], confidence: float, samples: int, seed: int) -> dict[str, float | int]:
    rng = random.Random(seed)
    n = len(values)
    means = []
    for _ in range(samples):
        draw = [values[rng.randrange(n)] for _ in range(n)]
        means.append(float(np.mean(draw)))
    alpha = (1.0 - confidence) / 2.0
    return {
        "n": n,
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "ci_low": float(np.quantile(means, alpha)),
        "ci_high": float(np.quantile(means, 1.0 - alpha)),
    }
