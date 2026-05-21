"""Empirical-game diagnostics for multi-evader pursuit policies."""

from __future__ import annotations

import csv
import json
import math
import subprocess
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from strategy_games.envs.multi_evader_pursuit import MultiEvaderPursuitConfig, MultiEvaderPursuitEnv
from strategy_games.policies.pursuit_targets import (
    LearnedPursuerPolicyAdapter,
    PolicyTarget,
    ScriptedPursuitPolicyAdapter,
)
from strategy_games.policies.scripted_pursuit import EVADER_POLICIES, PURSUER_POLICIES
from strategy_games.policies.scripted_pursuit import scripted_pursuit_actions
from strategy_games.rollouts import (
    PursuitRolloutConfig,
    multi_evader_config_from_mapping,
    run_scripted_pursuit_rollout,
)

SCHEMA_VERSION = "pursuit_policy_comparison/v1"
COMPARISON_ENV_ID = "multi_evader_pursuit_v1"
PAYOFF_ORIENTATION = "rows=pursuer_policies, columns=evader_policies"
PAYOFF_METRIC = "mean_pursuer_return"
GAME_TYPE = "rectangular_role_specific_general_sum_diagnostics"
DEFAULT_OUTPUT_JSON = Path("outputs/public/pursuit_demo/policy_comparison.json")
DEFAULT_OUTPUT_CSV = Path("outputs/public/pursuit_demo/policy_comparison.csv")
CSV_FIELDS = (
    "pursuer_policy",
    "evader_policy",
    "capture_rate",
    "survival_rate",
    "mean_pursuer_return",
    "mean_evader_return",
    "average_steps",
    "num_episodes",
)
EMPIRICAL_GAME_NOTES = (
    "Approximate empirical-game diagnostics over scripted policies and explicitly provided learned policies; not a "
    "Nash equilibrium, CFR result, PSRO result, learned best response, or exact exploitability estimate."
)


@dataclass(frozen=True)
class PursuitPolicyComparisonConfig:
    """Configuration for pursuit policy comparison diagnostics."""

    seeds: tuple[int, ...] = tuple(range(20))
    pursuer_policies: tuple[str, ...] = PURSUER_POLICIES
    evader_policies: tuple[str, ...] = EVADER_POLICIES
    eta: float = 1.0
    feint_steps: int = 3
    env: MultiEvaderPursuitConfig = field(default_factory=MultiEvaderPursuitConfig)
    output_json: Path = DEFAULT_OUTPUT_JSON
    output_csv: Path = DEFAULT_OUTPUT_CSV
    save_csv: bool = True
    created_at: str | None = None
    git_commit: str | None = None
    config_path: str | None = None
    learned_pursuer_checkpoint: Path | None = None


def compute_pursuit_policy_comparison(config: PursuitPolicyComparisonConfig | None = None) -> dict[str, Any]:
    """Compare scripted pursuer and evader policies as a rectangular empirical game."""

    config = config or PursuitPolicyComparisonConfig()
    _validate_comparison_config(config)
    created_at = config.created_at or datetime.now(UTC).replace(microsecond=0).isoformat()
    git_commit = config.git_commit if config.git_commit is not None else _git_commit()

    pursuer_adapters = _pursuer_adapters(config)
    evader_targets = _evader_targets(config)
    pursuer_policy_ids = tuple(adapter.target.policy_id for adapter in pursuer_adapters)
    evader_policy_ids = tuple(target.policy_id for target in evader_targets)
    metrics = _empty_metric_matrices(len(pursuer_adapters), len(evader_policy_ids))
    cell_sample_count = _zero_matrix(len(pursuer_adapters), len(evader_policy_ids), int)

    for row_idx, pursuer_adapter in enumerate(pursuer_adapters):
        for col_idx, evader_policy in enumerate(evader_policy_ids):
            samples = [
                _run_policy_pair(config, int(seed), pursuer_adapter, evader_policy)
                for seed in config.seeds
            ]
            cell_sample_count[row_idx][col_idx] = len(samples)
            for metric_name in metrics:
                metrics[metric_name][row_idx][col_idx] = _mean(sample[metric_name] for sample in samples)

    payoff_matrix = metrics["mean_pursuer_return"]
    empirical_game = empirical_game_diagnostics(payoff_matrix, pursuer_policy_ids, eta=config.eta)
    return {
        "schema_version": SCHEMA_VERSION,
        "env_id": COMPARISON_ENV_ID,
        "payoff_orientation": PAYOFF_ORIENTATION,
        "payoff_metric": PAYOFF_METRIC,
        "game_type": GAME_TYPE,
        "metadata": {
            "created_at": created_at,
            "git_commit": git_commit,
            "config_path": config.config_path,
        },
        "config": multi_evader_config_to_dict(config.env),
        "num_seeds": len(config.seeds),
        "num_episodes_per_cell": len(config.seeds),
        "seeds": list(config.seeds),
        "pursuer_policies": list(pursuer_policy_ids),
        "evader_policies": list(evader_policy_ids),
        "pursuer_policy_targets": [adapter.target.public_dict() for adapter in pursuer_adapters],
        "evader_policy_targets": [target.public_dict() for target in evader_targets],
        "methodology": {
            "row_player": "pursuer",
            "column_player": "evader",
            "primary_payoff": PAYOFF_METRIC,
            "column_mixture_for_regret": "uniform",
            "ranking_distribution_method": "payoff_weighted_multiplicative_weights_style",
            "ranking_distribution_params": {"eta": float(config.eta)},
            "is_zero_sum": False,
            "is_equilibrium_solver": False,
        },
        "payoff_matrix": payoff_matrix,
        "cell_sample_count": cell_sample_count,
        "metrics": metrics,
        "empirical_game": empirical_game,
    }


def empirical_game_diagnostics(
    payoff_matrix: Sequence[Sequence[float]],
    pursuer_policies: Sequence[str],
    eta: float = 1.0,
) -> dict[str, Any]:
    """Compute conservative row-player diagnostics from a payoff matrix."""

    matrix = _as_2d_array(payoff_matrix)
    if matrix.shape[0] != len(pursuer_policies):
        raise ValueError("number of pursuer policies must match payoff matrix rows")
    row_payoff = matrix.mean(axis=1)
    row_worst = matrix.min(axis=1)
    best_uniform_payoff = float(row_payoff.max())
    regret = best_uniform_payoff - row_payoff
    maximin_idx = int(np.argmax(row_worst))
    ranking = payoff_weighted_row_policy_ranking_distribution(row_payoff, pursuer_policies, eta=eta)
    return {
        "column_mixture": "uniform",
        "row_payoff_vs_uniform_column_mixture": _float_list(row_payoff),
        "row_worst_case_payoff": _float_list(row_worst),
        "empirical_regret_vs_uniform_column_mixture": _float_list(np.maximum(regret, 0.0)),
        "maximin_policy": str(pursuer_policies[maximin_idx]),
        "payoff_weighted_row_policy_ranking_distribution": ranking,
        "notes": EMPIRICAL_GAME_NOTES,
    }


def payoff_weighted_row_policy_ranking_distribution(
    scores: Sequence[float] | np.ndarray,
    pursuer_policies: Sequence[str],
    eta: float = 1.0,
) -> dict[str, Any]:
    """Return a payoff-weighted row-policy ranking distribution.

    This uses a multiplicative-weights-style soft scoring rule. It is a ranking
    distribution over rows, not an equilibrium solver.
    """

    if eta < 0 or not math.isfinite(float(eta)):
        raise ValueError("eta must be a finite non-negative value")
    score_array = np.asarray(scores, dtype=np.float64)
    if score_array.ndim != 1:
        raise ValueError("scores must be a 1D sequence")
    if score_array.size == 0:
        raise ValueError("scores must be non-empty")
    if score_array.size != len(pursuer_policies):
        raise ValueError("number of policies must match number of scores")
    if not np.isfinite(score_array).all():
        raise ValueError("scores must be finite")

    shifted = score_array - float(score_array.max())
    weights = np.exp(float(eta) * shifted)
    weight_sum = float(weights.sum())
    if weight_sum <= 0.0 or not math.isfinite(weight_sum):
        probabilities = np.full_like(weights, 1.0 / weights.size, dtype=np.float64)
    else:
        probabilities = weights / weight_sum

    return {
        "method": "multiplicative_weights_style",
        "eta": float(eta),
        "probabilities": {
            str(policy): float(probability)
            for policy, probability in zip(pursuer_policies, probabilities, strict=True)
        },
    }


def pursuit_policy_comparison_config_from_mapping(
    raw: Mapping[str, Any],
    config_path: str | None = None,
) -> PursuitPolicyComparisonConfig:
    """Build a pursuit policy comparison config from a YAML-style mapping."""

    policies = raw.get("policies", {})
    if not isinstance(policies, Mapping):
        policies = {}
    empirical_game = raw.get("empirical_game", {})
    if not isinstance(empirical_game, Mapping):
        empirical_game = {}
    output = raw.get("output", {})
    if not isinstance(output, Mapping):
        output = {}
    metadata = raw.get("metadata", {})
    if not isinstance(metadata, Mapping):
        metadata = {}

    return PursuitPolicyComparisonConfig(
        seeds=_seeds_from_mapping(raw),
        pursuer_policies=_string_tuple(policies.get("pursuers", PURSUER_POLICIES)),
        evader_policies=_string_tuple(policies.get("evaders", EVADER_POLICIES)),
        eta=float(empirical_game.get("eta", raw.get("eta", 1.0))),
        feint_steps=int(policies.get("feint_steps", raw.get("feint_steps", 3))),
        env=multi_evader_config_from_mapping(raw.get("env", {})),
        output_json=_output_path(output, "json_path", "json_filename", DEFAULT_OUTPUT_JSON.name),
        output_csv=_output_path(output, "csv_path", "csv_filename", DEFAULT_OUTPUT_CSV.name),
        save_csv=bool(output.get("save_csv", True)),
        created_at=str(metadata["created_at"]) if metadata.get("created_at") is not None else None,
        git_commit=str(metadata["git_commit"]) if metadata.get("git_commit") is not None else None,
        config_path=config_path,
        learned_pursuer_checkpoint=(
            Path(str(raw["learned_pursuer_checkpoint"]))
            if raw.get("learned_pursuer_checkpoint") is not None
            else None
        ),
    )


def multi_evader_config_to_dict(config: MultiEvaderPursuitConfig) -> dict[str, Any]:
    """Return the reproducibility subset of a multi-evader pursuit config."""

    return {
        "grid_size": [int(config.grid_size[0]), int(config.grid_size[1])],
        "num_evaders": int(config.num_evaders),
        "num_pursuers": int(config.num_pursuers),
        "max_steps": int(config.max_steps),
        "catch_radius": int(config.catch_radius),
    }


def save_policy_comparison_json(result: Mapping[str, Any], path: str | Path) -> Path:
    """Save a pursuit policy comparison artifact as JSON."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return output_path


def save_policy_comparison_csv(result: Mapping[str, Any], path: str | Path) -> Path:
    """Save pursuit policy comparison cell metrics as CSV."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows = policy_comparison_csv_rows(result)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    return output_path


def policy_comparison_csv_rows(result: Mapping[str, Any]) -> list[dict[str, float | int | str]]:
    """Flatten a policy comparison result into one CSV-style row per cell."""

    pursuer_policies = [str(policy) for policy in result["pursuer_policies"]]
    evader_policies = [str(policy) for policy in result["evader_policies"]]
    metrics = result["metrics"]
    sample_counts = result["cell_sample_count"]
    rows: list[dict[str, float | int | str]] = []
    for row_idx, pursuer_policy in enumerate(pursuer_policies):
        for col_idx, evader_policy in enumerate(evader_policies):
            rows.append(
                {
                    "pursuer_policy": pursuer_policy,
                    "evader_policy": evader_policy,
                    "capture_rate": float(metrics["capture_rate"][row_idx][col_idx]),
                    "survival_rate": float(metrics["survival_rate"][row_idx][col_idx]),
                    "mean_pursuer_return": float(metrics["mean_pursuer_return"][row_idx][col_idx]),
                    "mean_evader_return": float(metrics["mean_evader_return"][row_idx][col_idx]),
                    "average_steps": float(metrics["average_steps"][row_idx][col_idx]),
                    "num_episodes": int(sample_counts[row_idx][col_idx]),
                }
            )
    return rows


def format_policy_comparison_summary(result: Mapping[str, Any]) -> str:
    """Format a public summary table for row-policy diagnostics."""

    policies = [str(policy) for policy in result["pursuer_policies"]]
    empirical = result["empirical_game"]
    probabilities = empirical["payoff_weighted_row_policy_ranking_distribution"]["probabilities"]
    rows = []
    for idx, policy in enumerate(policies):
        rows.append(
            {
                "pursuer_policy": policy,
                "payoff_uniform": float(empirical["row_payoff_vs_uniform_column_mixture"][idx]),
                "worst_case": float(empirical["row_worst_case_payoff"][idx]),
                "regret_uniform": float(empirical["empirical_regret_vs_uniform_column_mixture"][idx]),
                "ranking_prob": float(probabilities[policy]),
            }
        )

    fields = ("pursuer_policy", "payoff_uniform", "worst_case", "regret_uniform", "ranking_prob")
    widths = {
        field: max(len(field), *(len(_format_table_value(row[field])) for row in rows))
        for field in fields
    }
    header = "  ".join(field.ljust(widths[field]) for field in fields)
    separator = "  ".join("-" * widths[field] for field in fields)
    body = [
        "  ".join(_format_table_value(row[field]).ljust(widths[field]) for field in fields)
        for row in rows
    ]
    maximin = str(empirical["maximin_policy"])
    return "\n".join([header, separator, *body, f"maximin_policy={maximin}"]) + "\n"


def _run_policy_pair(
    config: PursuitPolicyComparisonConfig,
    seed: int,
    pursuer_adapter: ScriptedPursuitPolicyAdapter | LearnedPursuerPolicyAdapter,
    evader_policy: str,
) -> dict[str, float]:
    if isinstance(pursuer_adapter, ScriptedPursuitPolicyAdapter):
        return _run_scripted_policy_pair(config, seed, pursuer_adapter.policy_id, evader_policy)

    env = MultiEvaderPursuitEnv(config.env)
    pursuer_adapter.validate_env(env)
    env.reset()
    rng = np.random.default_rng(seed)
    while not env.done:
        scripted = scripted_pursuit_actions(
            env=env,
            pursuer_policy="pursuer_greedy_nearest",
            evader_policy=evader_policy,
            rng=rng,
            step_index=env.steps,
            feint_steps=config.feint_steps,
        )
        actions = dict(scripted)
        actions["pursuer_0"] = pursuer_adapter.act(env, "pursuer_0", env.steps, rng)
        env.step(actions)
    return _episode_summary(env)


def _run_scripted_policy_pair(
    config: PursuitPolicyComparisonConfig,
    seed: int,
    pursuer_policy: str,
    evader_policy: str,
) -> dict[str, float]:
    trace = run_scripted_pursuit_rollout(
        PursuitRolloutConfig(
            seed=seed,
            pursuer_policy=pursuer_policy,
            evader_policy=evader_policy,
            feint_steps=config.feint_steps,
            created_at=config.created_at,
            env=config.env,
        )
    )
    return {
        "capture_rate": float(trace.summary.capture_rate),
        "survival_rate": float(trace.summary.survival_rate),
        "mean_pursuer_return": float(trace.summary.mean_pursuer_return),
        "mean_evader_return": float(trace.summary.mean_evader_return),
        "average_steps": float(trace.summary.total_steps),
    }


def _pursuer_adapters(
    config: PursuitPolicyComparisonConfig,
) -> list[ScriptedPursuitPolicyAdapter | LearnedPursuerPolicyAdapter]:
    adapters: list[ScriptedPursuitPolicyAdapter | LearnedPursuerPolicyAdapter] = [
        ScriptedPursuitPolicyAdapter(policy_id=policy, role="pursuer", feint_steps=config.feint_steps)
        for policy in config.pursuer_policies
    ]
    if config.learned_pursuer_checkpoint is not None:
        adapters.append(
            LearnedPursuerPolicyAdapter(
                config.learned_pursuer_checkpoint,
                env=MultiEvaderPursuitEnv(config.env),
            )
        )
    return adapters


def _evader_targets(config: PursuitPolicyComparisonConfig) -> list[PolicyTarget]:
    return [
        PolicyTarget(policy_id=policy, policy_type="scripted", role="evader")
        for policy in config.evader_policies
    ]


def _episode_summary(env: MultiEvaderPursuitEnv) -> dict[str, float]:
    captured = sum(1 for status in env.per_evader_status.values() if status == "captured")
    survived = sum(1 for status in env.per_evader_status.values() if status == "survived")
    pursuer_returns = [env.per_agent_returns[pursuer_id] for pursuer_id in env.pursuer_ids]
    evader_returns = [env.per_agent_returns[evader_id] for evader_id in env.evader_ids]
    return {
        "capture_rate": float(captured / env.config.num_evaders),
        "survival_rate": float(survived / env.config.num_evaders),
        "mean_pursuer_return": float(np.mean(pursuer_returns)),
        "mean_evader_return": float(np.mean(evader_returns)),
        "average_steps": float(env.steps),
    }


def _validate_comparison_config(config: PursuitPolicyComparisonConfig) -> None:
    if not config.seeds:
        raise ValueError("seeds must be non-empty")
    if not config.pursuer_policies:
        raise ValueError("pursuer_policies must be non-empty")
    if not config.evader_policies:
        raise ValueError("evader_policies must be non-empty")
    if config.eta < 0 or not math.isfinite(float(config.eta)):
        raise ValueError("eta must be a finite non-negative value")
    unsupported_pursuers = set(config.pursuer_policies) - set(PURSUER_POLICIES)
    unsupported_evaders = set(config.evader_policies) - set(EVADER_POLICIES)
    if unsupported_pursuers:
        raise ValueError(f"unsupported pursuer policies: {sorted(unsupported_pursuers)}")
    if unsupported_evaders:
        raise ValueError(f"unsupported evader policies: {sorted(unsupported_evaders)}")


def _empty_metric_matrices(rows: int, cols: int) -> dict[str, list[list[float]]]:
    return {
        "capture_rate": _zero_matrix(rows, cols, float),
        "survival_rate": _zero_matrix(rows, cols, float),
        "mean_pursuer_return": _zero_matrix(rows, cols, float),
        "mean_evader_return": _zero_matrix(rows, cols, float),
        "average_steps": _zero_matrix(rows, cols, float),
    }


def _zero_matrix(rows: int, cols: int, value_type: type[float] | type[int]) -> list[list[Any]]:
    return [[value_type(0) for _ in range(cols)] for _ in range(rows)]


def _as_2d_array(payoff_matrix: Sequence[Sequence[float]]) -> np.ndarray:
    matrix = np.asarray(payoff_matrix, dtype=np.float64)
    if matrix.ndim != 2:
        raise ValueError("payoff_matrix must be 2D")
    if matrix.shape[0] == 0 or matrix.shape[1] == 0:
        raise ValueError("payoff_matrix must be non-empty")
    if not np.isfinite(matrix).all():
        raise ValueError("payoff_matrix must contain finite values")
    return matrix


def _float_list(values: np.ndarray) -> list[float]:
    return [float(value) for value in values.tolist()]


def _mean(values: Sequence[float] | Any) -> float:
    array = np.asarray(list(values), dtype=np.float64)
    if array.size == 0:
        raise ValueError("cannot average empty values")
    return float(array.mean())


def _seeds_from_mapping(raw: Mapping[str, Any]) -> tuple[int, ...]:
    if "seeds" in raw:
        return tuple(int(seed) for seed in raw["seeds"])
    num_seeds = int(raw.get("num_seeds", 20))
    seed_start = int(raw.get("seed_start", 0))
    return tuple(range(seed_start, seed_start + num_seeds))


def _string_tuple(raw: Any) -> tuple[str, ...]:
    if isinstance(raw, str):
        return (raw,)
    if not isinstance(raw, Sequence):
        raise ValueError("expected a policy name or sequence of policy names")
    return tuple(str(item) for item in raw)


def _output_path(
    output: Mapping[str, Any],
    path_key: str,
    filename_key: str,
    default_filename: str,
) -> Path:
    if output.get(path_key) is not None:
        return Path(str(output[path_key]))
    output_dir = Path(str(output.get("dir", "outputs/public/pursuit_demo")))
    return output_dir / str(output.get(filename_key, default_filename))


def _git_commit() -> str | None:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    commit = completed.stdout.strip()
    return commit or None


def _format_table_value(value: float | int | str) -> str:
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)
