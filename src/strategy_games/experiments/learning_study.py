"""Learning-study protocol for gridworld strategy-loop and PPO diagnostics."""

from __future__ import annotations

import copy
import csv
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from strategy_games.experiments.convergence import multiseed_confidence
from strategy_games.experiments.logging import summarize_training_result, to_jsonable
from strategy_games.experiments.runner import run_from_config
from strategy_games.training.ppo_baseline import train_ppo_from_config
from strategy_games.utils.config import load_config

LEARNING_RUN_FIELDS = (
    "method",
    "seed",
    "status",
    "error_type",
    "error_message",
    "episode_return",
    "win_rate",
    "goal_rate",
    "catch_rate",
    "timeout_rate",
    "episode_length",
    "iterations",
    "updates",
    "total_steps",
    "final_selected_label",
    "buffer_diversity",
    "policy_loss",
    "value_loss",
    "entropy",
    "strategy_diversity",
    "robustness_score",
    "exploitability_proxy",
)
TRIAGE_FIELDS = (
    "method",
    "mean_return",
    "success_rate",
    "variance",
    "main_failure_mode",
    "diagnostic_confidence",
    "next_action",
    "completed_seed_count",
)
LEARNING_CURVE_FIELDS = (
    "method",
    "seed",
    "iteration",
    "step",
    "episode_return",
    "win_rate",
    "goal_rate",
    "catch_rate",
    "timeout_rate",
    "selected_label",
    "average_case_value",
    "robustness_score",
    "exploitability_proxy",
    "candidate_diversity",
    "policy_loss",
    "value_loss",
    "entropy",
)
SUMMARY_METRICS = ("episode_return", "win_rate", "goal_rate", "catch_rate", "timeout_rate", "episode_length")
DIAGNOSTIC_METRICS = (
    "policy_loss",
    "value_loss",
    "entropy",
    "strategy_diversity",
    "robustness_score",
    "exploitability_proxy",
)


@dataclass(frozen=True)
class LearningStudyConfig:
    """Config for a modest gridworld learning-quality study."""

    seeds: tuple[int, ...] = (0, 1, 2)
    strategy_config_path: Path = Path("configs/gridworld_day3.yaml")
    ppo_config_path: Path = Path("configs/gridworld_ppo_baseline.yaml")
    output_dir: Path = Path("outputs/public/learning_study")
    include_ppo: bool = True
    ppo_total_steps: int = 4096
    strategy_iterations: int | None = None
    task_success_metric: str = "goal_rate"
    high_variance_threshold: float = 0.25
    min_completed_seeds: int = 1


def run_learning_study(config: LearningStudyConfig | None = None) -> dict[str, Any]:
    """Run a seed sweep and write learning-study artifacts."""

    config = config or LearningStudyConfig()
    if not config.seeds:
        raise ValueError("seeds must be non-empty")
    if config.ppo_total_steps < 1:
        raise ValueError("ppo_total_steps must be positive")
    if config.high_variance_threshold < 0.0:
        raise ValueError("high_variance_threshold must be non-negative")
    if config.min_completed_seeds < 1:
        raise ValueError("min_completed_seeds must be positive")

    config.output_dir.mkdir(parents=True, exist_ok=True)
    config_dir = config.output_dir / "configs"
    config_dir.mkdir(parents=True, exist_ok=True)

    runs: list[dict[str, Any]] = []
    curves: list[dict[str, Any]] = []
    for seed in config.seeds:
        try:
            strategy_raw = configure_strategy_loop(
                load_config(config.strategy_config_path),
                seed=seed,
                output_dir=config.output_dir,
                strategy_iterations=config.strategy_iterations,
            )
            strategy_path = config_dir / f"strategy_seed{seed}.yaml"
            write_yaml(strategy_raw, strategy_path)
            strategy_result = run_from_config(strategy_path)
        except Exception as exc:  # noqa: BLE001 - failed seeds are part of the public artifact.
            runs.append(failed_run_row("strategy_loop", seed, exc))
        else:
            runs.append(strategy_run_row(seed, strategy_result))
            curves.extend(strategy_curve_rows(seed, strategy_result))

        if config.include_ppo:
            try:
                ppo_raw = configure_ppo(
                    load_config(config.ppo_config_path),
                    seed=seed,
                    total_steps=config.ppo_total_steps,
                    output_dir=config.output_dir,
                )
                ppo_path = config_dir / f"ppo_seed{seed}.yaml"
                write_yaml(ppo_raw, ppo_path)
                ppo_result = train_ppo_from_config(ppo_path)
            except Exception as exc:  # noqa: BLE001 - failed seeds are part of the public artifact.
                runs.append(failed_run_row("ppo_baseline", seed, exc))
            else:
                runs.append(ppo_run_row(seed, ppo_result))
                curves.extend(ppo_curve_rows(seed, ppo_result))

    summary = summarize_learning_runs(
        runs,
        task_success_metric=config.task_success_metric,
        high_variance_threshold=config.high_variance_threshold,
        min_completed_seeds=config.min_completed_seeds,
    )
    write_learning_study_outputs(config.output_dir, runs, summary, curves)
    return {
        "runs": runs,
        "summary": summary,
        "learning_curves": curves,
        "triage_table": summary["triage_table"],
    }


def learning_study_config_from_mapping(raw: Mapping[str, Any]) -> LearningStudyConfig:
    """Build a learning-study config from YAML-style data."""

    study_raw = raw.get("study", raw)
    if not isinstance(study_raw, Mapping):
        study_raw = {}
    seeds = study_raw.get("seeds", raw.get("seeds", [0, 1, 2]))
    return LearningStudyConfig(
        seeds=tuple(int(seed) for seed in seeds),
        strategy_config_path=Path(str(study_raw.get("strategy_config", "configs/gridworld_day3.yaml"))),
        ppo_config_path=Path(str(study_raw.get("ppo_config", "configs/gridworld_ppo_baseline.yaml"))),
        output_dir=Path(str(study_raw.get("output_dir", "outputs/public/learning_study"))),
        include_ppo=bool(study_raw.get("include_ppo", True)),
        ppo_total_steps=int(study_raw.get("ppo_total_steps", 4096)),
        strategy_iterations=(
            int(study_raw["strategy_iterations"]) if study_raw.get("strategy_iterations") is not None else None
        ),
        task_success_metric=str(study_raw.get("task_success_metric", "goal_rate")),
        high_variance_threshold=float(study_raw.get("high_variance_threshold", 0.25)),
        min_completed_seeds=int(study_raw.get("min_completed_seeds", 1)),
    )


def configure_strategy_loop(
    raw: Mapping[str, Any],
    seed: int,
    output_dir: Path,
    strategy_iterations: int | None = None,
) -> dict[str, Any]:
    """Return a per-seed strategy-loop config with public logging enabled."""

    config = copy.deepcopy(dict(raw))
    config["seed"] = int(seed)
    if strategy_iterations is not None:
        training = dict(config.get("training", {}))
        training["iterations"] = int(strategy_iterations)
        config["training"] = training
    logging = dict(config.get("logging", {}))
    logging.update(
        {
            "enabled": True,
            "output_dir": str(output_dir / "strategy_runs"),
            "run_name": f"strategy_seed{seed}",
        }
    )
    config["logging"] = logging
    return config


def configure_ppo(raw: Mapping[str, Any], seed: int, total_steps: int, output_dir: Path) -> dict[str, Any]:
    """Return a per-seed PPO config with public logging enabled."""

    config = copy.deepcopy(dict(raw))
    config["seed"] = int(seed)
    ppo = dict(config.get("ppo", {}))
    ppo["total_steps"] = int(total_steps)
    config["ppo"] = ppo
    logging = dict(config.get("logging", {}))
    logging.update({"enabled": True, "output_dir": str(output_dir / "ppo_runs" / f"seed{seed}")})
    config["logging"] = logging
    return config


def strategy_run_row(seed: int, result: Mapping[str, Any]) -> dict[str, Any]:
    """Convert one strategy-loop result into the learning-study run schema."""

    metrics = summarize_training_result(result)
    history = result.get("history", [])
    final_updates = metrics.get("final_update_losses", {})
    if not isinstance(final_updates, Mapping):
        final_updates = {}
    return {
        "method": "strategy_loop",
        "seed": int(seed),
        "status": "completed",
        "error_type": "",
        "error_message": "",
        "episode_return": float(metrics["mean_episode_return"]),
        "win_rate": float(metrics["mean_win_rate"]),
        "goal_rate": float(metrics["mean_goal_rate"]),
        "catch_rate": float(metrics["mean_catch_rate"]),
        "timeout_rate": float(metrics["mean_timeout_rate"]),
        "episode_length": _mean_history_metric(history, "rollout", "steps"),
        "iterations": int(metrics["iterations"]),
        "updates": 0,
        "total_steps": 0,
        "final_selected_label": str(metrics["final_selected_label"]),
        "buffer_diversity": float(metrics["buffer_diversity"]),
        "policy_loss": _float_from_mapping(final_updates, "policy_loss"),
        "value_loss": _float_from_mapping(final_updates, "policy_value_loss"),
        "entropy": _float_from_mapping(final_updates, "policy_entropy"),
        "strategy_diversity": float(metrics["buffer_diversity"]),
        "robustness_score": _mean_history_metric(history, "selection_metrics", "robustness_score"),
        "exploitability_proxy": _mean_history_metric(history, "selection_metrics", "exploitability_proxy"),
    }


def ppo_run_row(seed: int, result: Mapping[str, Any]) -> dict[str, Any]:
    """Convert one PPO result into the learning-study run schema."""

    return {
        "method": "ppo_baseline",
        "seed": int(seed),
        "status": "completed",
        "error_type": "",
        "error_message": "",
        "episode_return": float(result.get("episode_return", 0.0)),
        "win_rate": float(result.get("win_rate", 0.0)),
        "goal_rate": float(result.get("goal_rate", 0.0)),
        "catch_rate": float(result.get("catch_rate", 0.0)),
        "timeout_rate": float(result.get("timeout_rate", 0.0)),
        "episode_length": float(result.get("episode_length", 0.0)),
        "iterations": 0,
        "updates": int(result.get("updates", 0)),
        "total_steps": int(result.get("total_steps", 0)),
        "final_selected_label": "ppo_policy",
        "buffer_diversity": 0.0,
        "policy_loss": float(result.get("policy_loss", 0.0)),
        "value_loss": float(result.get("value_loss", 0.0)),
        "entropy": float(result.get("entropy", 0.0)),
        "strategy_diversity": 0.0,
        "robustness_score": 0.0,
        "exploitability_proxy": 0.0,
        "diagnosis_failure_mode": str(result.get("diagnosis", {}).get("failure_mode", "") if isinstance(result.get("diagnosis"), Mapping) else ""),
    }


def failed_run_row(method: str, seed: int, exc: Exception) -> dict[str, Any]:
    """Return an explicit failed-seed row for ``runs.json``."""

    return {
        "method": method,
        "seed": int(seed),
        "status": "failed",
        "error_type": type(exc).__name__,
        "error_message": str(exc),
        "episode_return": 0.0,
        "win_rate": 0.0,
        "goal_rate": 0.0,
        "catch_rate": 0.0,
        "timeout_rate": 0.0,
        "episode_length": 0.0,
        "iterations": 0,
        "updates": 0,
        "total_steps": 0,
        "final_selected_label": "",
        "buffer_diversity": 0.0,
        "policy_loss": 0.0,
        "value_loss": 0.0,
        "entropy": 0.0,
        "strategy_diversity": 0.0,
        "robustness_score": 0.0,
        "exploitability_proxy": 0.0,
    }


def strategy_curve_rows(seed: int, result: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Return one learning-curve row per strategy-loop iteration."""

    rows: list[dict[str, Any]] = []
    history = result.get("history", [])
    if not isinstance(history, Sequence):
        return rows
    for item in history:
        if not isinstance(item, Mapping):
            continue
        rollout = item.get("rollout", {})
        selection = item.get("selection_metrics", {})
        updates = item.get("updates", {})
        rows.append(
            {
                "method": "strategy_loop",
                "seed": int(seed),
                "iteration": int(item.get("iteration", len(rows))),
                "step": int(item.get("iteration", len(rows))),
                "episode_return": _float_from_mapping(rollout, "episode_return"),
                "win_rate": _float_from_mapping(rollout, "win_rate"),
                "goal_rate": _float_from_mapping(rollout, "goal_rate"),
                "catch_rate": _float_from_mapping(rollout, "catch_rate"),
                "timeout_rate": _float_from_mapping(rollout, "timeout_rate"),
                "selected_label": str(item.get("selected_label", "latent")),
                "average_case_value": _float_from_mapping(selection, "average_case_value"),
                "robustness_score": _float_from_mapping(selection, "robustness_score"),
                "exploitability_proxy": _float_from_mapping(selection, "exploitability_proxy"),
                "candidate_diversity": float(item.get("candidate_diversity", 0.0)),
                "policy_loss": _float_from_mapping(updates, "policy_loss"),
                "value_loss": _float_from_mapping(updates, "policy_value_loss"),
                "entropy": _float_from_mapping(updates, "policy_entropy"),
            }
        )
    return rows


def ppo_curve_rows(seed: int, result: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Return one learning-curve row per PPO update."""

    rows: list[dict[str, Any]] = []
    history = result.get("update_history", [])
    if not isinstance(history, Sequence):
        return rows
    for item in history:
        if not isinstance(item, Mapping):
            continue
        rows.append(
            {
                "method": "ppo_baseline",
                "seed": int(seed),
                "iteration": int(item.get("update", len(rows))),
                "step": int(item.get("steps_done", 0)),
                "episode_return": float(item.get("eval_episode_return", item.get("train_episode_return", 0.0))),
                "win_rate": item.get("eval_win_rate", ""),
                "goal_rate": item.get("eval_goal_rate", ""),
                "catch_rate": item.get("eval_catch_rate", ""),
                "timeout_rate": "",
                "selected_label": "ppo_policy",
                "average_case_value": "",
                "robustness_score": "",
                "exploitability_proxy": "",
                "candidate_diversity": "",
                "policy_loss": float(item.get("policy_loss", 0.0)),
                "value_loss": float(item.get("value_loss", 0.0)),
                "entropy": float(item.get("entropy", 0.0)),
            }
        )
    return rows


def summarize_learning_runs(
    runs: Sequence[Mapping[str, Any]],
    task_success_metric: str = "goal_rate",
    high_variance_threshold: float = 0.25,
    min_completed_seeds: int = 1,
) -> dict[str, Any]:
    """Summarize run rows by method with mean/std/CI metrics."""

    grouped: dict[str, list[Mapping[str, Any]]] = {}
    for row in runs:
        grouped.setdefault(str(row["method"]), []).append(row)
    summary_rows = []
    for method, items in grouped.items():
        completed = [item for item in items if str(item.get("status", "completed")) == "completed"]
        failed = [item for item in items if str(item.get("status", "completed")) != "completed"]
        result: dict[str, Any] = {
            "method": method,
            "seeds": [int(item["seed"]) for item in completed],
            "attempted_seeds": [int(item["seed"]) for item in items],
            "failed_seeds": [int(item["seed"]) for item in failed],
            "runs": len(completed),
            "completed_seed_count": len(completed),
            "failed_seed_count": len(failed),
        }
        for metric in SUMMARY_METRICS:
            confidence = multiseed_confidence(_numeric_values(completed, metric))
            result[f"{metric}_mean"] = float(confidence["mean"])
            result[f"{metric}_std"] = float(confidence["std"])
            result[f"{metric}_ci_low"] = float(confidence["ci_low"])
            result[f"{metric}_ci_high"] = float(confidence["ci_high"])
        for metric in DIAGNOSTIC_METRICS:
            values = _numeric_values(completed, metric)
            if values:
                result[f"{metric}_mean"] = float(sum(values) / len(values))
        result["task_success_metric"] = task_success_metric
        result["success_rate_mean"] = float(result.get(f"{task_success_metric}_mean", 0.0))
        result["success_rate_std"] = float(result.get(f"{task_success_metric}_std", 0.0))
        result["selected_strategy_distribution"] = selected_strategy_distribution(completed)
        summary_rows.append(result)
    triage_rows = [
        triage_row(
            row,
            task_success_metric=task_success_metric,
            high_variance_threshold=high_variance_threshold,
            min_completed_seeds=min_completed_seeds,
        )
        for row in summary_rows
    ]
    return {
        "schema_version": "gridworld_learning_study/v1",
        "methods": sorted(summary_rows, key=lambda row: str(row["method"])),
        "triage_table": sorted(triage_rows, key=lambda row: str(row["method"])),
        "task_success_metric": task_success_metric,
        "high_variance_threshold": float(high_variance_threshold),
        "min_completed_seeds": int(min_completed_seeds),
        "notes": [
            "Learning-study metrics are diagnostics over the custom gridworld.",
            "Direct-goal remains a sanity-check heuristic ceiling for the default layout.",
            "exploitability_proxy, when present in curves, is approximate and not Nash exploitability.",
            "Failed seed runs are recorded in runs.json and excluded from aggregate metrics.",
        ],
    }


def selected_strategy_distribution(items: Sequence[Mapping[str, Any]]) -> dict[str, float]:
    """Return final selected-label frequencies for completed runs."""

    labels = [str(item.get("final_selected_label", "")) for item in items if item.get("final_selected_label")]
    if not labels:
        return {}
    total = float(len(labels))
    return {label: labels.count(label) / total for label in sorted(set(labels))}


def triage_row(
    method_summary: Mapping[str, Any],
    task_success_metric: str,
    high_variance_threshold: float,
    min_completed_seeds: int,
) -> dict[str, Any]:
    """Return one failure-mode triage row from an aggregate summary."""

    method = str(method_summary["method"])
    completed = int(method_summary.get("completed_seed_count", 0))
    failed = int(method_summary.get("failed_seed_count", 0))
    success_rate = float(method_summary.get("success_rate_mean", 0.0))
    variance = float(method_summary.get("success_rate_std", 0.0))
    mean_return = float(method_summary.get("episode_return_mean", 0.0))
    failure_mode = "low_success_rate"
    confidence = "medium"
    next_action = "Inspect reward scale, evaluator alignment, and rollout traces before adding architecture."

    if completed < min_completed_seeds:
        failure_mode = "all_runs_failed" if completed == 0 else "too_few_completed_seeds"
        confidence = "high" if completed == 0 else "medium"
        next_action = "Fix crashed seeds or rerun with enough completed seeds before changing modeling code."
    elif failed:
        failure_mode = "incomplete_runs"
        confidence = "medium"
        next_action = "Inspect failed seed rows in runs.json and stabilize the experiment runner."
    elif variance > high_variance_threshold:
        failure_mode = "inconclusive_high_variance"
        confidence = "low"
        next_action = "Rerun with more seeds before changing architecture."
    elif success_rate >= 0.5:
        failure_mode = "learning_path_live"
        confidence = "high" if completed >= 3 else "medium"
        next_action = "Increase seed count and compare against stronger baselines before making claims."
    elif method == "ppo_baseline":
        entropy = float(method_summary.get("entropy_mean", 0.0))
        value_loss = float(method_summary.get("value_loss_mean", 0.0))
        failure_mode = "ppo_policy_collapse" if entropy < 0.05 else "ppo_learning_weak"
        confidence = "medium"
        next_action = (
            "Inspect PPO reward scale, advantage normalization, learning rate, entropy, and value loss."
            if value_loss >= 0.0
            else "Inspect PPO training metrics and rollout traces."
        )
    elif method == "strategy_loop":
        diversity = float(method_summary.get("strategy_diversity_mean", 0.0))
        failure_mode = "strategy_diversity_collapse" if diversity < 0.05 else "strategy_execution_weak"
        confidence = "medium"
        next_action = (
            "Improve EBM sampling/replay if diversity collapsed; otherwise strengthen strategy-conditioned execution."
        )

    return {
        "method": method,
        "mean_return": mean_return,
        "success_rate": success_rate,
        "success_metric": task_success_metric,
        "variance": variance,
        "main_failure_mode": failure_mode,
        "diagnostic_confidence": confidence,
        "next_action": next_action,
        "completed_seed_count": completed,
    }


def write_learning_study_outputs(
    output_dir: Path,
    runs: Sequence[Mapping[str, Any]],
    summary: Mapping[str, Any],
    curves: Sequence[Mapping[str, Any]],
) -> dict[str, str]:
    """Write canonical learning-study artifacts."""

    output_dir.mkdir(parents=True, exist_ok=True)
    runs_path = output_dir / "runs.json"
    summary_path = output_dir / "summary.json"
    curves_path = output_dir / "learning_curves.csv"
    triage_path = output_dir / "triage_table.csv"
    with runs_path.open("w", encoding="utf-8") as handle:
        json.dump(to_jsonable(list(runs)), handle, indent=2, sort_keys=True)
        handle.write("\n")
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(to_jsonable(summary), handle, indent=2, sort_keys=True)
        handle.write("\n")
    with curves_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=LEARNING_CURVE_FIELDS)
        writer.writeheader()
        writer.writerows(to_jsonable(list(curves)))
    triage_rows = summary.get("triage_table", []) if isinstance(summary, Mapping) else []
    with triage_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=TRIAGE_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(to_jsonable(list(triage_rows)) if isinstance(triage_rows, Sequence) else [])
    return {
        "runs_json": str(runs_path),
        "summary_json": str(summary_path),
        "learning_curves_csv": str(curves_path),
        "triage_table_csv": str(triage_path),
    }


def write_yaml(config: Mapping[str, Any], path: Path) -> Path:
    """Write a YAML config snapshot."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(to_jsonable(config), handle, sort_keys=False)
    return path


def _float_from_mapping(value: Any, key: str) -> float:
    if isinstance(value, Mapping) and key in value:
        return float(value[key])
    return 0.0


def _mean_history_metric(history: Any, container_key: str, metric_key: str) -> float:
    if not isinstance(history, Sequence):
        return 0.0
    values = []
    for item in history:
        if not isinstance(item, Mapping):
            continue
        container = item.get(container_key, {})
        if isinstance(container, Mapping) and metric_key in container:
            values.append(float(container[metric_key]))
    return float(sum(values) / len(values)) if values else 0.0


def _numeric_values(items: Sequence[Mapping[str, Any]], key: str) -> list[float]:
    values: list[float] = []
    for item in items:
        value = item.get(key)
        if value is None or value == "":
            continue
        values.append(float(value))
    return values
