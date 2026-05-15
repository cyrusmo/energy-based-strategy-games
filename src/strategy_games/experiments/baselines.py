"""Baseline comparison helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from strategy_games.envs.gridworld import GridworldConfig
from strategy_games.experiments.runner import run_from_config
from strategy_games.training.ppo_baseline import run_direct_goal_baseline, run_random_policy_baseline, train_ppo_from_config

BASELINE_FIELDS = ("baseline", "episode_return", "win_rate", "goal_rate", "catch_rate", "timeout_rate")


def compare_baselines(
    config_path: str | Path = "configs/gridworld_day2.yaml",
    episodes: int = 5,
    seed: int | None = 0,
    env_config: GridworldConfig | None = None,
    include_ppo: bool = True,
    ppo_config_path: str | Path = "configs/gridworld_ppo_baseline.yaml",
) -> list[dict[str, float | str]]:
    """Return a shared metric table for public baseline comparison."""

    random_metrics = run_random_policy_baseline(episodes=episodes, seed=seed, env_config=env_config)
    direct_metrics = run_direct_goal_baseline(episodes=episodes, env_config=env_config)
    strategy_result = run_from_config(config_path)
    strategy_metrics = summarize_strategy_loop_baseline(strategy_result)

    rows = [
        _row("random_policy", random_metrics),
        _row("direct_goal_heuristic", direct_metrics),
        _row("day2_strategy_loop", strategy_metrics),
    ]
    if include_ppo:
        ppo_metrics = train_ppo_from_config(ppo_config_path)
        rows.append(_row("ppo_baseline", ppo_metrics))
    return rows


def summarize_strategy_loop_baseline(result: dict[str, Any]) -> dict[str, float]:
    """Summarize Day 2 strategy-loop rollout metrics into the baseline schema."""

    history = result.get("history", [])
    if not isinstance(history, list) or not history:
        raise ValueError("strategy-loop result must contain non-empty history")
    rollouts = [item.get("rollout", {}) for item in history if isinstance(item, dict)]
    return {
        "episode_return": _mean(rollouts, "episode_return"),
        "win_rate": _mean(rollouts, "win_rate"),
        "goal_rate": _mean(rollouts, "goal_rate"),
        "catch_rate": _mean(rollouts, "catch_rate"),
        "timeout_rate": _mean(rollouts, "timeout_rate"),
    }


def format_baseline_table(rows: list[dict[str, float | str]]) -> str:
    """Format baseline rows as a compact aligned table."""

    widths = {
        field: max(len(field), *(len(_format_value(row[field])) for row in rows))
        for field in BASELINE_FIELDS
    }
    header = "  ".join(field.ljust(widths[field]) for field in BASELINE_FIELDS)
    separator = "  ".join("-" * widths[field] for field in BASELINE_FIELDS)
    body = [
        "  ".join(_format_value(row[field]).ljust(widths[field]) for field in BASELINE_FIELDS)
        for row in rows
    ]
    return "\n".join([header, separator, *body]) + "\n"


def save_baseline_metrics(rows: list[dict[str, float | str]], path: str | Path) -> Path:
    """Save baseline metric rows as JSON."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(rows, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return output_path


def _row(name: str, metrics: dict[str, float]) -> dict[str, float | str]:
    row: dict[str, float | str] = {"baseline": name}
    for field in BASELINE_FIELDS[1:]:
        row[field] = float(metrics.get(field, 0.0))
    return row


def _mean(items: list[Any], key: str) -> float:
    values = [float(item[key]) for item in items if isinstance(item, dict) and key in item]
    if not values:
        return 0.0
    return float(np.mean(values))


def _format_value(value: float | str) -> str:
    if isinstance(value, float):
        return f"{value:.3f}"
    return value
