"""Config-driven benchmark runner."""

from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml

from strategy_games.benchmarks.registry import make_benchmark_adapter
from strategy_games.benchmarks.types import BenchmarkDependencyError
from strategy_games.experiments.convergence import multiseed_confidence
from strategy_games.experiments.logging import to_jsonable
from strategy_games.utils.config import load_config

DEFAULT_BASELINES = ("random_policy", "direct_goal_heuristic", "strategy_loop")


def run_benchmark_from_config(path: str | Path) -> dict[str, object]:
    """Load a YAML benchmark config and run it."""

    raw = load_config(path)
    result = run_benchmark_suite(raw)
    suite = raw.get("suite", {})
    if bool(suite.get("write_artifacts", True)):
        artifact_paths = write_benchmark_artifacts(result, raw)
        result["artifacts"] = artifact_paths
    return result


def run_benchmark_suite(config: Mapping[str, Any]) -> dict[str, object]:
    """Run a benchmark suite and return rows plus aggregate summary."""

    entries = config.get("benchmarks", [])
    if not isinstance(entries, list) or not entries:
        raise ValueError("benchmark config must contain a non-empty `benchmarks` list")

    rows: list[dict[str, object]] = []
    skipped: list[dict[str, str]] = []
    for entry in entries:
        if not isinstance(entry, Mapping):
            raise ValueError("each benchmark entry must be a mapping")
        env_id = str(entry["env_id"])
        baselines = tuple(entry.get("baselines", DEFAULT_BASELINES))
        seeds = tuple(int(seed) for seed in entry.get("seeds", [0]))
        adapter = make_benchmark_adapter(env_id, entry)
        for seed in seeds:
            for baseline in baselines:
                try:
                    rows.append(adapter.rollout(str(baseline), seed).to_dict())
                except BenchmarkDependencyError as exc:
                    skipped.append({"env_id": env_id, "baseline": str(baseline), "reason": str(exc)})
                    break

    return {
        "suite_name": str(config.get("suite", {}).get("name", "benchmark_suite")),
        "results": rows,
        "summary": summarize_benchmark_results(rows),
        "skipped": skipped,
    }


def summarize_benchmark_results(rows: list[dict[str, object]]) -> dict[str, object]:
    """Aggregate benchmark rows by environment and baseline."""

    groups: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        groups[(str(row["env_id"]), str(row["baseline"]))].append(row)

    summaries: list[dict[str, object]] = []
    for (env_id, baseline), items in sorted(groups.items()):
        summary: dict[str, object] = {
            "env_id": env_id,
            "baseline": baseline,
            "num_runs": len(items),
        }
        for key in (
            "episode_return",
            "win_rate",
            "goal_rate",
            "catch_rate",
            "timeout_rate",
            "survival_or_capture_rate",
            "steps",
            "wall_clock_seconds",
            "average_case_value",
            "worst_case_value",
            "exploitability_proxy",
            "strategy_diversity",
        ):
            values = [_safe_float(item[key]) for item in items if item.get(key) is not None]
            if values:
                confidence = multiseed_confidence(values)
                summary[f"mean_{key}"] = float(confidence["mean"])
                summary[f"std_{key}"] = float(confidence["std"])
                summary[f"ci_low_{key}"] = float(confidence["ci_low"])
                summary[f"ci_high_{key}"] = float(confidence["ci_high"])
        summaries.append(summary)
    return {"by_env_and_baseline": summaries}


def write_benchmark_artifacts(result: Mapping[str, object], config: Mapping[str, Any]) -> dict[str, str]:
    """Write `results.jsonl`, `summary.json`, and a config snapshot."""

    suite_config = config.get("suite", {})
    if not isinstance(suite_config, Mapping):
        suite_config = {}
    output_dir = Path(suite_config.get("output_dir", "outputs/public/benchmarks"))
    run_name = str(suite_config.get("run_name", result.get("suite_name", "benchmark_suite")))
    run_dir = output_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    results_path = run_dir / "results.jsonl"
    with results_path.open("w", encoding="utf-8") as handle:
        rows = result.get("results", [])
        if isinstance(rows, list):
            for row in rows:
                handle.write(json.dumps(to_jsonable(row), sort_keys=True) + "\n")

    summary_path = run_dir / "summary.json"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(to_jsonable(result.get("summary", {})), handle, indent=2, sort_keys=True)
        handle.write("\n")

    config_path = run_dir / "config.yaml"
    with config_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(to_jsonable(dict(config)), handle, sort_keys=False)

    return {
        "run_dir": str(run_dir),
        "results_jsonl": str(results_path),
        "summary_json": str(summary_path),
        "config_yaml": str(config_path),
    }


def _safe_float(value: object) -> float:
    if isinstance(value, int | float | str):
        return float(value)
    return 0.0
