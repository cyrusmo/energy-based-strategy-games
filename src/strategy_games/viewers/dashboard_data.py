"""Headless data preparation for the performance dashboard."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from strategy_games.experiments.convergence import detect_convergence, energy_gap


def load_dashboard_data(root: str | Path = "outputs/public") -> dict[str, Any]:
    """Load all known dashboard artifacts, tolerating missing files."""

    root_path = Path(root)
    calibration_path = root_path / "device_calibration.json"
    baselines_path = root_path / "baselines" / "metrics.json"
    multiseed_path = root_path / "multiseed" / "summary.json"
    iterations_path = _default_iterations_path(root_path)
    return {
        "resource": resource_rows(_read_json(calibration_path).get("jobs", [])),
        "convergence": convergence_rows(_read_jsonl(iterations_path)),
        "quality": quality_rows(_read_json(baselines_path), _read_json(multiseed_path)),
        "artifacts": {
            "calibration": str(calibration_path),
            "baselines": str(baselines_path),
            "multiseed": str(multiseed_path),
            "iterations": str(iterations_path) if iterations_path else "",
        },
        "missing": [
            name
            for name, path in {
                "device calibration": calibration_path,
                "baseline metrics": baselines_path,
                "multiseed summary": multiseed_path,
                "training iterations": iterations_path,
            }.items()
            if path is None or not path.exists()
        ],
    }


def resource_rows(jobs: Any) -> list[dict[str, Any]]:
    """Normalize device-calibration job rows for display."""

    if not isinstance(jobs, list):
        return []
    rows = []
    for job in jobs:
        if not isinstance(job, dict):
            continue
        cpu_ms = _optional_float(job.get("cpu_ms"))
        mps_ms = _optional_float(job.get("mps_ms"))
        rows.append(
            {
                "job": str(job.get("job", "unknown")),
                "cpu_ms": cpu_ms,
                "mps_ms": mps_ms,
                "speedup": _optional_float(job.get("speedup")) or 1.0,
                "recommended_device": str(job.get("recommended_device", "cpu")),
                "mps_supported": bool(job.get("mps_supported", False)),
                "explanation": _resource_explanation(cpu_ms, mps_ms, str(job.get("recommended_device", "cpu"))),
            }
        )
    return rows


def convergence_rows(history: list[dict[str, Any]]) -> dict[str, Any]:
    """Flatten training-loop history into curve rows plus goal badges."""

    curves: list[dict[str, Any]] = []
    for item in history:
        rollout = item.get("rollout", {}) if isinstance(item, dict) else {}
        updates = item.get("updates", {}) if isinstance(item, dict) else {}
        selection = item.get("selection_metrics", {}) if isinstance(item, dict) else {}
        if not isinstance(rollout, dict):
            rollout = {}
        if not isinstance(updates, dict):
            updates = {}
        if not isinstance(selection, dict):
            selection = {}
        gap = energy_gap(updates)
        curves.append(
            {
                "iteration": int(item.get("iteration", len(curves))) if isinstance(item, dict) else len(curves),
                "episode_return": _optional_float(rollout.get("episode_return")) or 0.0,
                "goal_rate": _optional_float(rollout.get("goal_rate")) or 0.0,
                "win_rate": _optional_float(rollout.get("win_rate")) or 0.0,
                "policy_loss": _optional_float(updates.get("policy_loss")),
                "world_model_loss": _optional_float(updates.get("world_model_loss")),
                "ebm_loss": _optional_float(updates.get("ebm_loss")),
                "ebm_energy_gap": gap,
                "robustness_score": _optional_float(selection.get("robustness_score")),
                "exploitability_proxy": _optional_float(selection.get("exploitability_proxy")),
            }
        )
    return {
        "curves": curves,
        "badges": {
            "goal_rate": detect_convergence(history, metric="goal_rate", target=1.0) if history else {},
            "episode_return": detect_convergence(history, metric="episode_return", target=0.9) if history else {},
            "ebm_energy_gap": detect_convergence(history, metric="ebm_energy_gap", target=0.0) if history else {},
        },
    }


def quality_rows(baselines_payload: dict[str, Any], multiseed_payload: dict[str, Any]) -> list[dict[str, Any]]:
    """Merge baseline metrics and multiseed summaries into display rows."""

    rows: list[dict[str, Any]] = []
    metrics = baselines_payload.get("baselines", baselines_payload)
    if isinstance(metrics, list):
        for item in metrics:
            if isinstance(item, dict):
                rows.append(_quality_from_baseline(item))
    for item in multiseed_payload.get("baselines", []):
        if not isinstance(item, dict):
            continue
        baseline = str(item.get("baseline", "unknown"))
        existing = next((row for row in rows if row["baseline"] == baseline), None)
        target: dict[str, Any] = existing if existing is not None else {"baseline": baseline}
        target.update(
            {
                "episode_return": _optional_float(item.get("episode_return_mean")) or target.get("episode_return", 0.0),
                "win_rate": _optional_float(item.get("win_rate_mean")) or target.get("win_rate", 0.0),
                "goal_rate": _optional_float(item.get("goal_rate_mean")) or target.get("goal_rate", 0.0),
                "return_ci": [
                    _optional_float(item.get("episode_return_ci_low")),
                    _optional_float(item.get("episode_return_ci_high")),
                ],
                "win_rate_ci": [_optional_float(item.get("win_rate_ci_low")), _optional_float(item.get("win_rate_ci_high"))],
                "runs": int(item.get("runs", 0)),
            }
        )
        if existing is None:
            rows.append(target)
    return sorted(rows, key=lambda row: str(row["baseline"]))


def _quality_from_baseline(item: dict[str, Any]) -> dict[str, Any]:
    return {
        "baseline": str(item.get("baseline", "unknown")),
        "episode_return": _optional_float(item.get("episode_return")) or 0.0,
        "win_rate": _optional_float(item.get("win_rate")) or 0.0,
        "goal_rate": _optional_float(item.get("goal_rate")) or 0.0,
        "catch_rate": _optional_float(item.get("catch_rate")) or 0.0,
        "timeout_rate": _optional_float(item.get("timeout_rate")) or 0.0,
    }


def _resource_explanation(cpu_ms: float | None, mps_ms: float | None, recommended: str) -> str:
    if cpu_ms is None:
        return "CPU timing is unavailable; use CPU until calibration succeeds."
    if mps_ms is None:
        return "MPS did not complete this job; CPU is the safe path."
    faster = "MPS" if recommended == "mps" else "CPU"
    return f"{faster} is recommended based on measured wall-clock time."


def _default_iterations_path(root: Path) -> Path | None:
    strategy_runs = root / "multiseed" / "strategy_runs"
    if strategy_runs.exists():
        candidates = sorted(strategy_runs.glob("*/iterations.jsonl"))
        if candidates:
            return candidates[0]
    fallback = root / "debug" / "debug_run" / "iterations.jsonl"
    return fallback


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        loaded = json.load(handle)
    return loaded if isinstance(loaded, dict) else {}


def _read_jsonl(path: Path | None) -> list[dict[str, Any]]:
    if path is None or not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                loaded = json.loads(line)
                if isinstance(loaded, dict):
                    rows.append(loaded)
    return rows


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
