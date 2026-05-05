"""Lightweight file logging for public experiment artifacts."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml


class ExperimentLogger:
    """Write public experiment artifacts to a run directory."""

    def __init__(self, output_dir: str | Path, run_name: str = "debug_run") -> None:
        if not run_name:
            raise ValueError("run_name must be non-empty")
        self.output_dir = Path(output_dir)
        self.run_name = run_name
        self.run_dir = self.output_dir / run_name
        self.run_dir.mkdir(parents=True, exist_ok=True)

    @property
    def iterations_path(self) -> Path:
        """Path to JSONL per-iteration metrics."""

        return self.run_dir / "iterations.jsonl"

    @property
    def metrics_path(self) -> Path:
        """Path to aggregate metrics JSON."""

        return self.run_dir / "metrics.json"

    @property
    def config_path(self) -> Path:
        """Path to the saved YAML config snapshot."""

        return self.run_dir / "config.yaml"

    def save_config(self, config: Mapping[str, Any]) -> Path:
        """Write the raw experiment config as YAML."""

        with self.config_path.open("w", encoding="utf-8") as handle:
            yaml.safe_dump(to_jsonable(config), handle, sort_keys=False)
        return self.config_path

    def write_iterations(self, history: list[dict[str, Any]]) -> Path:
        """Write one JSON object per iteration."""

        with self.iterations_path.open("w", encoding="utf-8") as handle:
            for item in history:
                handle.write(json.dumps(to_jsonable(item), sort_keys=True) + "\n")
        return self.iterations_path

    def write_metrics(self, result: Mapping[str, Any]) -> Path:
        """Write aggregate metrics for a completed run."""

        metrics = summarize_training_result(result)
        with self.metrics_path.open("w", encoding="utf-8") as handle:
            json.dump(to_jsonable(metrics), handle, indent=2, sort_keys=True)
            handle.write("\n")
        return self.metrics_path

    def log_run(self, config: Mapping[str, Any], result: Mapping[str, Any]) -> dict[str, str]:
        """Write config, iteration JSONL, and final metrics."""

        history = result.get("history", [])
        if not isinstance(history, list):
            raise ValueError("result['history'] must be a list")
        self.save_config(config)
        self.write_iterations(history)
        self.write_metrics(result)
        return {
            "run_dir": str(self.run_dir),
            "iterations_jsonl": str(self.iterations_path),
            "metrics_json": str(self.metrics_path),
            "config_yaml": str(self.config_path),
        }


def summarize_training_result(result: Mapping[str, Any]) -> dict[str, Any]:
    """Compute aggregate metrics from a training-loop result."""

    history = result.get("history", [])
    if not isinstance(history, list):
        raise ValueError("result['history'] must be a list")

    rollouts = [item.get("rollout", {}) for item in history if isinstance(item, dict)]
    updates = [item.get("updates", {}) for item in history if isinstance(item, dict)]
    final_updates = updates[-1] if updates and isinstance(updates[-1], dict) else {}

    return {
        "iterations": len(history),
        "final_selected_label": result.get("final_selected_label"),
        "buffer_size": result.get("buffer_size", 0),
        "buffer_diversity": result.get("buffer_diversity", 0.0),
        "mean_episode_return": _mean_metric(rollouts, "episode_return"),
        "mean_win_rate": _mean_metric(rollouts, "win_rate"),
        "mean_goal_rate": _mean_metric(rollouts, "goal_rate"),
        "mean_catch_rate": _mean_metric(rollouts, "catch_rate"),
        "mean_timeout_rate": _mean_metric(rollouts, "timeout_rate"),
        "final_update_losses": final_updates,
    }


def to_jsonable(value: Any) -> Any:
    """Recursively convert common scientific Python values to JSON-safe values."""

    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [to_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, torch.Tensor):
        if value.ndim == 0:
            return value.item()
        return value.detach().cpu().tolist()
    return value


def _mean_metric(items: list[Any], key: str) -> float:
    values = [float(item[key]) for item in items if isinstance(item, dict) and key in item]
    return float(np.mean(values)) if values else 0.0
