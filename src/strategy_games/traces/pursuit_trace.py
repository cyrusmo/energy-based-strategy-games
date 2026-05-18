"""Validated, versioned trace schema for pursuit/evasion rollouts."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from strategy_games.envs.pursuit_actions import ACTIONS, evader_ids, is_evader, is_pursuer, pursuer_ids

SCHEMA_VERSION = "pursuit_trace/v1"
TRACE_TYPE = "pursuit_evasion"
ROLE_PURSUER = "pursuer"
ROLE_EVADER = "evader"
TERMINATED_REASONS = ("all_evaders_captured", "timeout", "manual_stop", "unknown")
EVADER_STATUSES = ("captured", "survived")

GridSize = list[int]
JsonPosition = list[int]


@dataclass(frozen=True)
class CaptureEvent:
    """One capture event in a pursuit/evasion trace."""

    pursuer_id: str
    evader_id: str
    position: JsonPosition
    t: int


@dataclass(frozen=True)
class PursuitStep:
    """One post-transition step in a pursuit/evasion trace."""

    t: int
    agent_positions: dict[str, JsonPosition]
    agent_roles: dict[str, str]
    actions: dict[str, str]
    step_rewards: dict[str, float]
    captures: list[CaptureEvent] = field(default_factory=list)
    active_evaders: list[str] = field(default_factory=list)
    done: bool = False


@dataclass(frozen=True)
class PursuitSummary:
    """Trace-level summary for public reporting and viewing."""

    outcome: str
    terminated_reason: Literal["all_evaders_captured", "timeout", "manual_stop", "unknown"]
    capture_rate: float
    survival_rate: float
    all_evaders_captured: bool
    mean_evader_return: float
    mean_pursuer_return: float
    per_agent_returns: dict[str, float]
    initial_positions: dict[str, JsonPosition]
    final_positions: dict[str, JsonPosition]
    total_steps: int
    per_evader_status: dict[str, str]


@dataclass(frozen=True)
class PursuitTrace:
    """Versioned pursuit/evasion rollout artifact."""

    schema_version: str
    trace_type: str
    env_id: str
    episode_id: str
    seed: int
    grid_size: GridSize
    num_evaders: int
    num_pursuers: int
    metadata: dict[str, Any]
    steps: list[PursuitStep]
    summary: PursuitSummary


def pursuit_trace_to_dict(trace: PursuitTrace) -> dict[str, Any]:
    """Convert a pursuit trace to a JSON-serializable dictionary."""

    return {
        "schema_version": trace.schema_version,
        "trace_type": trace.trace_type,
        "env_id": trace.env_id,
        "episode_id": trace.episode_id,
        "seed": trace.seed,
        "grid_size": list(trace.grid_size),
        "num_evaders": trace.num_evaders,
        "num_pursuers": trace.num_pursuers,
        "metadata": dict(trace.metadata),
        "steps": [
            {
                "t": step.t,
                "agent_positions": _positions_to_json(step.agent_positions),
                "agent_roles": dict(step.agent_roles),
                "actions": dict(step.actions),
                "step_rewards": {agent_id: float(value) for agent_id, value in step.step_rewards.items()},
                "captures": [
                    {
                        "pursuer_id": capture.pursuer_id,
                        "evader_id": capture.evader_id,
                        "position": list(capture.position),
                        "t": capture.t,
                    }
                    for capture in step.captures
                ],
                "active_evaders": list(step.active_evaders),
                "done": bool(step.done),
            }
            for step in trace.steps
        ],
        "summary": pursuit_summary_to_dict(trace.summary),
    }


def pursuit_summary_to_dict(summary: PursuitSummary) -> dict[str, Any]:
    """Convert a trace summary to a JSON-serializable dictionary."""

    return {
        "outcome": summary.outcome,
        "terminated_reason": summary.terminated_reason,
        "capture_rate": float(summary.capture_rate),
        "survival_rate": float(summary.survival_rate),
        "all_evaders_captured": bool(summary.all_evaders_captured),
        "mean_evader_return": float(summary.mean_evader_return),
        "mean_pursuer_return": float(summary.mean_pursuer_return),
        "per_agent_returns": {agent_id: float(value) for agent_id, value in summary.per_agent_returns.items()},
        "initial_positions": _positions_to_json(summary.initial_positions),
        "final_positions": _positions_to_json(summary.final_positions),
        "total_steps": int(summary.total_steps),
        "per_evader_status": dict(summary.per_evader_status),
    }


def pursuit_trace_from_dict(raw: dict[str, Any]) -> PursuitTrace:
    """Build and validate a pursuit trace from a dictionary."""

    steps = [
        PursuitStep(
            t=int(item["t"]),
            agent_positions=_positions_from_json(item["agent_positions"]),
            agent_roles={str(key): str(value) for key, value in item["agent_roles"].items()},
            actions={str(key): str(value) for key, value in item["actions"].items()},
            step_rewards={str(key): float(value) for key, value in item["step_rewards"].items()},
            captures=[
                CaptureEvent(
                    pursuer_id=str(capture["pursuer_id"]),
                    evader_id=str(capture["evader_id"]),
                    position=_position_from_json(capture["position"]),
                    t=int(capture["t"]),
                )
                for capture in item.get("captures", [])
            ],
            active_evaders=[str(evader_id) for evader_id in item.get("active_evaders", [])],
            done=bool(item.get("done", False)),
        )
        for item in raw["steps"]
    ]
    summary_raw = raw["summary"]
    summary = PursuitSummary(
        outcome=str(summary_raw["outcome"]),
        terminated_reason=str(summary_raw["terminated_reason"]),  # type: ignore[arg-type]
        capture_rate=float(summary_raw["capture_rate"]),
        survival_rate=float(summary_raw["survival_rate"]),
        all_evaders_captured=bool(summary_raw["all_evaders_captured"]),
        mean_evader_return=float(summary_raw["mean_evader_return"]),
        mean_pursuer_return=float(summary_raw["mean_pursuer_return"]),
        per_agent_returns={str(key): float(value) for key, value in summary_raw["per_agent_returns"].items()},
        initial_positions=_positions_from_json(summary_raw["initial_positions"]),
        final_positions=_positions_from_json(summary_raw["final_positions"]),
        total_steps=int(summary_raw["total_steps"]),
        per_evader_status={str(key): str(value) for key, value in summary_raw["per_evader_status"].items()},
    )
    trace = PursuitTrace(
        schema_version=str(raw["schema_version"]),
        trace_type=str(raw["trace_type"]),
        env_id=str(raw["env_id"]),
        episode_id=str(raw["episode_id"]),
        seed=int(raw["seed"]),
        grid_size=_grid_size_from_json(raw["grid_size"]),
        num_evaders=int(raw["num_evaders"]),
        num_pursuers=int(raw["num_pursuers"]),
        metadata=dict(raw.get("metadata", {})),
        steps=steps,
        summary=summary,
    )
    validate_pursuit_trace(trace)
    return trace


def save_pursuit_trace(trace: PursuitTrace, path: str | Path) -> Path:
    """Save a validated pursuit trace as JSON."""

    validate_pursuit_trace(trace)
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(pursuit_trace_to_dict(trace), handle, indent=2, sort_keys=True)
        handle.write("\n")
    return output_path


def save_pursuit_summary(summary: PursuitSummary, path: str | Path) -> Path:
    """Save a pursuit summary as JSON."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(pursuit_summary_to_dict(summary), handle, indent=2, sort_keys=True)
        handle.write("\n")
    return output_path


def load_pursuit_trace(path: str | Path) -> PursuitTrace:
    """Load and validate a pursuit trace from JSON."""

    input_path = Path(path)
    with input_path.open("r", encoding="utf-8") as handle:
        raw = json.load(handle)
    if not isinstance(raw, dict):
        raise ValueError("pursuit trace JSON must contain an object")
    return pursuit_trace_from_dict(raw)


def validate_pursuit_trace(trace: PursuitTrace) -> None:
    """Raise ``ValueError`` if a pursuit trace violates the public schema."""

    if trace.schema_version != SCHEMA_VERSION:
        raise ValueError(f"invalid schema_version: {trace.schema_version}")
    if trace.trace_type != TRACE_TYPE:
        raise ValueError(f"invalid trace_type: {trace.trace_type}")
    _validate_grid_size(trace.grid_size)
    if trace.num_evaders < 1:
        raise ValueError("num_evaders must be positive")
    if trace.num_pursuers < 1:
        raise ValueError("num_pursuers must be positive")

    expected_pursuers = pursuer_ids(trace.num_pursuers)
    expected_evaders = evader_ids(trace.num_evaders)
    expected_agents = [*expected_pursuers, *expected_evaders]
    expected_roles = {
        **{agent_id: ROLE_PURSUER for agent_id in expected_pursuers},
        **{agent_id: ROLE_EVADER for agent_id in expected_evaders},
    }

    _validate_positions(trace.summary.initial_positions, expected_agents, trace.grid_size, "initial_positions")
    _validate_positions(trace.summary.final_positions, expected_agents, trace.grid_size, "final_positions")
    _validate_agent_values(trace.summary.per_agent_returns, expected_agents, "per_agent_returns")
    _validate_evader_status(trace.summary.per_evader_status, expected_evaders)
    if trace.summary.terminated_reason not in TERMINATED_REASONS:
        raise ValueError(f"invalid terminated_reason: {trace.summary.terminated_reason}")
    if trace.summary.total_steps != len(trace.steps):
        raise ValueError("summary total_steps must match number of steps")
    if trace.steps and trace.summary.final_positions != trace.steps[-1].agent_positions:
        raise ValueError("summary final_positions must match the last step positions")

    captured = [evader_id for evader_id, status in trace.summary.per_evader_status.items() if status == "captured"]
    survived = [evader_id for evader_id, status in trace.summary.per_evader_status.items() if status == "survived"]
    _assert_close(trace.summary.capture_rate, len(captured) / trace.num_evaders, "capture_rate")
    _assert_close(trace.summary.survival_rate, len(survived) / trace.num_evaders, "survival_rate")
    if trace.summary.all_evaders_captured != (len(captured) == trace.num_evaders):
        raise ValueError("all_evaders_captured must agree with per_evader_status")
    if trace.summary.terminated_reason == "all_evaders_captured" and not trace.summary.all_evaders_captured:
        raise ValueError("all_evaders_captured termination requires all evaders captured")
    if trace.summary.terminated_reason == "timeout" and trace.summary.all_evaders_captured:
        raise ValueError("timeout termination cannot have all evaders captured")
    _validate_mean_return(trace.summary.per_agent_returns, expected_evaders, trace.summary.mean_evader_return, "mean_evader_return")
    _validate_mean_return(
        trace.summary.per_agent_returns,
        expected_pursuers,
        trace.summary.mean_pursuer_return,
        "mean_pursuer_return",
    )

    for index, step in enumerate(trace.steps):
        if step.t != index:
            raise ValueError("step t values must be zero-indexed and contiguous")
        if step.agent_roles != expected_roles:
            raise ValueError("step agent_roles must match trace agent ids and roles")
        _validate_positions(step.agent_positions, expected_agents, trace.grid_size, "agent_positions")
        _validate_actions(step.actions, expected_agents)
        _validate_agent_values(step.step_rewards, expected_agents, "step_rewards")
        _validate_active_evaders(step.active_evaders, expected_evaders)
        for capture in step.captures:
            _validate_capture(capture, expected_pursuers, expected_evaders, trace.grid_size, step.t)
        if index < len(trace.steps) - 1 and step.done:
            raise ValueError("only the final step may be marked done")
        if index == len(trace.steps) - 1 and not step.done:
            raise ValueError("final step must be marked done")


def _positions_to_json(positions: dict[str, JsonPosition]) -> dict[str, JsonPosition]:
    return {str(agent_id): _position_from_json(position) for agent_id, position in positions.items()}


def _positions_from_json(raw: Any) -> dict[str, JsonPosition]:
    if not isinstance(raw, dict):
        raise ValueError("positions must be an object")
    return {str(agent_id): _position_from_json(position) for agent_id, position in raw.items()}


def _position_from_json(raw: Any) -> JsonPosition:
    if not isinstance(raw, (list, tuple)) or len(raw) != 2:
        raise ValueError("position must be [row, col]")
    return [int(raw[0]), int(raw[1])]


def _grid_size_from_json(raw: Any) -> GridSize:
    grid_size = _position_from_json(raw)
    _validate_grid_size(grid_size)
    return grid_size


def _validate_grid_size(grid_size: GridSize) -> None:
    if not isinstance(grid_size, list) or len(grid_size) != 2:
        raise ValueError("grid_size must be [height, width]")
    height, width = grid_size
    if not isinstance(height, int) or not isinstance(width, int):
        raise ValueError("grid_size values must be integers")
    if height < 2 or width < 2:
        raise ValueError("grid_size values must be at least 2")


def _validate_positions(
    positions: dict[str, JsonPosition],
    expected_agents: list[str],
    grid_size: GridSize,
    field_name: str,
) -> None:
    if set(positions) != set(expected_agents):
        raise ValueError(f"{field_name} keys must match trace agent ids")
    height, width = grid_size
    for agent_id, position in positions.items():
        if len(position) != 2:
            raise ValueError(f"{field_name}[{agent_id}] must be [row, col]")
        row, col = position
        if not (0 <= row < height and 0 <= col < width):
            raise ValueError(f"{field_name}[{agent_id}] is outside grid bounds")


def _validate_actions(actions: dict[str, str], expected_agents: list[str]) -> None:
    if set(actions) != set(expected_agents):
        raise ValueError("actions keys must match trace agent ids")
    for agent_id, action in actions.items():
        if action not in ACTIONS:
            raise ValueError(f"invalid action for {agent_id}: {action}")


def _validate_agent_values(values: dict[str, float], expected_agents: list[str], field_name: str) -> None:
    if set(values) != set(expected_agents):
        raise ValueError(f"{field_name} keys must match trace agent ids")
    for agent_id, value in values.items():
        if not isinstance(value, float | int):
            raise ValueError(f"{field_name}[{agent_id}] must be numeric")


def _validate_evader_status(statuses: dict[str, str], expected_evaders: list[str]) -> None:
    if set(statuses) != set(expected_evaders):
        raise ValueError("per_evader_status keys must match evader ids")
    for evader_id, status in statuses.items():
        if status not in EVADER_STATUSES:
            raise ValueError(f"invalid status for {evader_id}: {status}")


def _validate_active_evaders(active_evaders: list[str], expected_evaders: list[str]) -> None:
    active_set = set(active_evaders)
    if len(active_evaders) != len(active_set):
        raise ValueError("active_evaders must not contain duplicates")
    if not active_set.issubset(set(expected_evaders)):
        raise ValueError("active_evaders must be a subset of evader ids")


def _validate_capture(
    capture: CaptureEvent,
    expected_pursuers: list[str],
    expected_evaders: list[str],
    grid_size: GridSize,
    step_t: int,
) -> None:
    if capture.pursuer_id not in expected_pursuers or not is_pursuer(capture.pursuer_id):
        raise ValueError(f"invalid capture pursuer_id: {capture.pursuer_id}")
    if capture.evader_id not in expected_evaders or not is_evader(capture.evader_id):
        raise ValueError(f"invalid capture evader_id: {capture.evader_id}")
    if capture.t != step_t:
        raise ValueError("capture t must match step t")
    _validate_positions({"capture": capture.position}, ["capture"], grid_size, "capture.position")


def _validate_mean_return(values: dict[str, float], agent_ids: list[str], expected: float, field_name: str) -> None:
    mean_value = sum(float(values[agent_id]) for agent_id in agent_ids) / len(agent_ids)
    _assert_close(expected, mean_value, field_name)


def _assert_close(actual: float, expected: float, field_name: str) -> None:
    if abs(float(actual) - float(expected)) > 1e-6:
        raise ValueError(f"{field_name} is inconsistent")
