"""Render-data helpers for pursuit trace viewers."""

from __future__ import annotations

import html
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from strategy_games.envs.pursuit_actions import is_evader, is_pursuer
from strategy_games.traces.pursuit_trace import PursuitTrace, validate_pursuit_trace

COMPARISON_GENERATION_COMMAND = "python examples/compare_pursuit_policies.py"


@dataclass(frozen=True)
class ComparisonState:
    """Loaded comparison artifact state for viewer surfaces."""

    comparison_path: str | None
    comparison_artifact: dict[str, Any] | None
    comparison_error: str | None


def grid_frame(trace: PursuitTrace, frame_index: int) -> list[list[str]]:
    """Return a text grid for a frame, where frame 0 is the initial state."""

    validate_pursuit_trace(trace)
    height, width = trace.grid_size
    positions = _positions_for_frame(trace, clamp_frame_index(trace, frame_index))
    cells: list[list[list[str]]] = [[[] for _ in range(width)] for _ in range(height)]
    for agent_id, position in positions.items():
        row, col = position
        cells[row][col].append(_agent_label(agent_id))
    return [[" ".join(sorted(cell, key=_label_sort_key)) if cell else "." for cell in row] for row in cells]


def summary_metrics(trace: PursuitTrace) -> dict[str, float | int | str | bool]:
    """Return summary metrics for display surfaces."""

    validate_pursuit_trace(trace)
    summary = trace.summary
    return {
        "episode_id": trace.episode_id,
        "outcome": summary.outcome,
        "terminated_reason": summary.terminated_reason,
        "capture_rate": summary.capture_rate,
        "survival_rate": summary.survival_rate,
        "all_evaders_captured": summary.all_evaders_captured,
        "mean_evader_return": summary.mean_evader_return,
        "mean_pursuer_return": summary.mean_pursuer_return,
        "total_steps": summary.total_steps,
        "num_evaders": trace.num_evaders,
        "num_pursuers": trace.num_pursuers,
    }


def trace_metadata(trace: PursuitTrace) -> dict[str, object]:
    """Return trace-level metadata for inspection panels."""

    validate_pursuit_trace(trace)
    return {
        "env_id": trace.env_id,
        "episode_id": trace.episode_id,
        "seed": trace.seed,
        "grid_size": trace.grid_size,
        "num_evaders": trace.num_evaders,
        "num_pursuers": trace.num_pursuers,
        "max_steps": trace.metadata.get("max_steps"),
        "catch_radius": trace.metadata.get("catch_radius"),
        "outcome": trace.summary.outcome,
        "terminated_reason": trace.summary.terminated_reason,
    }


def pursuer_effectiveness_summary(trace: PursuitTrace) -> dict[str, object]:
    """Return pursuer-facing effectiveness metrics."""

    validate_pursuit_trace(trace)
    return {
        "capture_rate": trace.summary.capture_rate,
        "all_evaders_captured": trace.summary.all_evaders_captured,
        "mean_pursuer_return": trace.summary.mean_pursuer_return,
        "average_episode_steps": trace.summary.total_steps,
        "captured_evaders": sum(1 for status in trace.summary.per_evader_status.values() if status == "captured"),
        "interpretation": "Higher capture rate and mean pursuer return indicate stronger pursuer behavior in this trace.",
    }


def evader_effectiveness_summary(trace: PursuitTrace) -> dict[str, object]:
    """Return evader-facing effectiveness metrics."""

    validate_pursuit_trace(trace)
    return {
        "survival_rate": trace.summary.survival_rate,
        "mean_evader_return": trace.summary.mean_evader_return,
        "timeout_or_survival_outcome": trace.summary.terminated_reason == "timeout",
        "surviving_evaders": sum(1 for status in trace.summary.per_evader_status.values() if status == "survived"),
        "average_episode_steps": trace.summary.total_steps,
        "interpretation": "Higher survival rate and mean evader return indicate stronger evader behavior in this trace.",
    }


def transition_context(trace: PursuitTrace, frame_index: int) -> dict[str, object]:
    """Return action/reward/capture context for a selected frame."""

    validate_pursuit_trace(trace)
    frame = clamp_frame_index(trace, frame_index)
    current_positions = _positions_for_frame(trace, frame)
    previous_positions = trace.summary.initial_positions if frame <= 1 else trace.steps[frame - 2].agent_positions
    if frame == 0:
        return {
            "frame_index": 0,
            "t": "initial",
            "previous_positions": {},
            "current_positions": current_positions,
            "actions": {},
            "step_rewards": {},
            "captures": [],
            "active_evaders": sorted(agent_id for agent_id in trace.summary.initial_positions if is_evader(agent_id)),
            "done": False,
        }

    step = trace.steps[frame - 1]
    return {
        "frame_index": frame,
        "t": step.t,
        "previous_positions": previous_positions,
        "current_positions": current_positions,
        "actions": dict(step.actions),
        "step_rewards": dict(step.step_rewards),
        "captures": [f"{capture.pursuer_id}->{capture.evader_id}" for capture in step.captures],
        "active_evaders": list(step.active_evaders),
        "done": step.done,
    }


def styled_grid_frame(trace: PursuitTrace, frame_index: int) -> list[list[dict[str, object]]]:
    """Return styled cell data for a trace frame."""

    validate_pursuit_trace(trace)
    frame = clamp_frame_index(trace, frame_index)
    height, width = trace.grid_size
    positions = _positions_for_frame(trace, frame)
    active_evaders = _active_evaders_for_frame(trace, frame)
    capture_positions = _capture_positions_for_frame(trace, frame)
    cells: list[list[dict[str, object]]] = []
    for row in range(height):
        cell_row: list[dict[str, object]] = []
        for col in range(width):
            agents = [
                _agent_cell(agent_id, active_evaders)
                for agent_id, position in positions.items()
                if position == [row, col]
            ]
            roles = {str(agent["role"]) for agent in agents}
            if len(roles) > 1:
                role = "mixed"
            elif roles:
                role = roles.pop()
            else:
                role = "empty"
            capture_highlight = [row, col] in capture_positions
            cell_row.append(
                {
                    "row": row,
                    "col": col,
                    "agents": sorted(agents, key=lambda item: _label_sort_key(str(item["label"]))),
                    "label": " ".join(
                        str(agent["label"])
                        for agent in sorted(agents, key=lambda item: _label_sort_key(str(item["label"])))
                    ),
                    "role": role,
                    "capture_highlight": capture_highlight,
                }
            )
        cells.append(cell_row)
    return cells


def styled_grid_html(trace: PursuitTrace, frame_index: int) -> str:
    """Return a compact HTML grid for Streamlit markdown rendering."""

    cells = styled_grid_frame(trace, frame_index)
    rows = []
    for row in cells:
        rendered_cells = []
        for cell in row:
            classes = ["pursuit-cell", f"role-{cell['role']}"]
            if cell["capture_highlight"]:
                classes.append("capture")
            if not cell["label"]:
                classes.append("empty")
            rendered_cells.append(
                f"<td class=\"{' '.join(classes)}\"><span>{html.escape(str(cell['label']) or '.')}</span></td>"
            )
        rows.append(f"<tr>{''.join(rendered_cells)}</tr>")

    return (
        "<style>"
        ".pursuit-grid{border-collapse:collapse;width:100%;max-width:640px;table-layout:fixed;}"
        ".pursuit-cell{border:1px solid #d7dde8;aspect-ratio:1/1;text-align:center;font-weight:700;"
        "font-size:0.9rem;min-width:34px;height:42px;}"
        ".pursuit-cell span{display:inline-block;padding:2px 3px;border-radius:4px;}"
        ".role-empty{background:#f8fafc;color:#94a3b8;font-weight:400;}"
        ".role-pursuer{background:#fee2e2;color:#991b1b;}"
        ".role-evader{background:#dbeafe;color:#1e3a8a;}"
        ".role-mixed{background:#fef3c7;color:#78350f;}"
        ".capture{outline:3px solid #f97316;outline-offset:-3px;}"
        "</style>"
        f"<table class=\"pursuit-grid\"><tbody>{''.join(rows)}</tbody></table>"
    )


def clamp_frame_index(trace: PursuitTrace, frame_index: int) -> int:
    """Clamp a frame index to the valid trace range."""

    validate_pursuit_trace(trace)
    return min(max(int(frame_index), 0), len(trace.steps))


def step_frame_index(trace: PursuitTrace, frame_index: int, delta: int) -> int:
    """Move a frame index by ``delta`` while staying in range."""

    return clamp_frame_index(trace, int(frame_index) + int(delta))


def trace_table_rows(trace: PursuitTrace) -> list[dict[str, object]]:
    """Return one table row per trace step."""

    validate_pursuit_trace(trace)
    rows: list[dict[str, object]] = []
    for step in trace.steps:
        capture_labels = [f"{capture.pursuer_id}->{capture.evader_id}" for capture in step.captures]
        rows.append(
            {
                "t": step.t,
                "actions": dict(step.actions),
                "step_rewards": dict(step.step_rewards),
                "active_evaders": list(step.active_evaders),
                "captures": capture_labels,
                "done": step.done,
            }
        )
    return rows


def load_policy_comparison_state(path: str | Path) -> ComparisonState:
    """Load a policy comparison artifact for the viewer."""

    artifact_path = Path(path)
    if not artifact_path.exists():
        return ComparisonState(
            comparison_path=str(artifact_path),
            comparison_artifact=None,
            comparison_error=f"No comparison artifact found. Generate one with `{COMPARISON_GENERATION_COMMAND}`.",
        )
    try:
        with artifact_path.open("r", encoding="utf-8") as handle:
            artifact = json.load(handle)
        validate_policy_comparison_artifact(artifact)
    except (OSError, json.JSONDecodeError, TypeError, ValueError) as exc:
        return ComparisonState(
            comparison_path=str(artifact_path),
            comparison_artifact=None,
            comparison_error=f"Could not load comparison artifact: {exc}",
        )
    return ComparisonState(
        comparison_path=str(artifact_path),
        comparison_artifact=artifact,
        comparison_error=None,
    )


def validate_policy_comparison_artifact(artifact: object) -> None:
    """Validate the minimum fields needed by viewer diagnostics."""

    if not isinstance(artifact, dict):
        raise ValueError("comparison artifact must be a JSON object")
    required = {
        "schema_version",
        "payoff_orientation",
        "payoff_metric",
        "pursuer_policies",
        "evader_policies",
        "metrics",
        "empirical_game",
    }
    missing = sorted(required - set(artifact))
    if missing:
        raise ValueError(f"comparison artifact missing keys: {missing}")
    if artifact["schema_version"] != "pursuit_policy_comparison/v1":
        raise ValueError(f"unsupported comparison schema_version: {artifact['schema_version']}")
    empirical = artifact["empirical_game"]
    if not isinstance(empirical, dict):
        raise ValueError("empirical_game must be an object")
    for key in (
        "row_payoff_vs_uniform_column_mixture",
        "empirical_regret_vs_uniform_column_mixture",
        "maximin_policy",
        "payoff_weighted_row_policy_ranking_distribution",
    ):
        if key not in empirical:
            raise ValueError(f"empirical_game missing key: {key}")


def comparison_diagnostics_summary(state: ComparisonState) -> dict[str, object]:
    """Return viewer-ready head-to-head diagnostics from comparison state."""

    if state.comparison_artifact is None:
        return {
            "loaded": False,
            "source_path": state.comparison_path,
            "error": state.comparison_error,
            "generation_command": COMPARISON_GENERATION_COMMAND,
        }
    artifact = state.comparison_artifact
    empirical = artifact["empirical_game"]
    ranking = empirical["payoff_weighted_row_policy_ranking_distribution"]
    return {
        "loaded": True,
        "source_path": state.comparison_path,
        "payoff_orientation": artifact["payoff_orientation"],
        "payoff_metric": artifact["payoff_metric"],
        "row_player": artifact.get("methodology", {}).get("row_player", "pursuer"),
        "column_player": artifact.get("methodology", {}).get("column_player", "evader"),
        "maximin_policy": empirical["maximin_policy"],
        "row_payoff_vs_uniform_column_mixture": empirical["row_payoff_vs_uniform_column_mixture"],
        "empirical_regret_vs_uniform_column_mixture": empirical["empirical_regret_vs_uniform_column_mixture"],
        "ranking_distribution": ranking["probabilities"],
        "notes": empirical.get("notes", ""),
    }


def agent_status_rows(trace: PursuitTrace) -> list[dict[str, object]]:
    """Return per-agent status rows for display surfaces."""

    validate_pursuit_trace(trace)
    rows: list[dict[str, object]] = []
    for agent_id, role in trace.steps[0].agent_roles.items():
        row = {
            "agent_id": agent_id,
            "role": role,
            "initial_position": trace.summary.initial_positions[agent_id],
            "final_position": trace.summary.final_positions[agent_id],
            "return": trace.summary.per_agent_returns[agent_id],
        }
        if role == "evader":
            row["status"] = trace.summary.per_evader_status[agent_id]
        rows.append(row)
    return rows


def plot_pursuit_trace(trace: PursuitTrace, path: str | Path) -> Path:
    """Save a simple trajectory plot for a pursuit trace."""

    validate_pursuit_trace(trace)
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/strategy_games_mplconfig")
    os.environ.setdefault("XDG_CACHE_HOME", "/tmp/strategy_games_cache")
    Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
    Path(os.environ["XDG_CACHE_HOME"]).mkdir(parents=True, exist_ok=True)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    height, width = trace.grid_size
    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    ax.set_xlim(-0.5, width - 0.5)
    ax.set_ylim(height - 0.5, -0.5)
    ax.set_xticks(range(width))
    ax.set_yticks(range(height))
    ax.grid(True, linewidth=0.5, alpha=0.45)

    for agent_id in trace.summary.initial_positions:
        path_positions = [trace.summary.initial_positions[agent_id]]
        path_positions.extend(step.agent_positions[agent_id] for step in trace.steps)
        ys = [position[0] for position in path_positions]
        xs = [position[1] for position in path_positions]
        color = "tab:red" if is_pursuer(agent_id) else "tab:blue"
        marker = "s" if is_pursuer(agent_id) else "o"
        ax.plot(xs, ys, marker=marker, label=agent_id, linewidth=1.5, color=color, alpha=0.8)

    ax.set_title(f"{trace.summary.outcome}, capture_rate={trace.summary.capture_rate:.2f}")
    ax.set_xlabel("col")
    ax.set_ylabel("row")
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=140)
    plt.close(fig)
    return output_path


def _active_evaders_for_frame(trace: PursuitTrace, frame_index: int) -> set[str]:
    if frame_index <= 0:
        return {agent_id for agent_id in trace.summary.initial_positions if is_evader(agent_id)}
    return set(trace.steps[frame_index - 1].active_evaders)


def _capture_positions_for_frame(trace: PursuitTrace, frame_index: int) -> list[list[int]]:
    if frame_index <= 0:
        return []
    return [list(capture.position) for capture in trace.steps[frame_index - 1].captures]


def _agent_cell(agent_id: str, active_evaders: set[str]) -> dict[str, object]:
    if is_pursuer(agent_id):
        return {"agent_id": agent_id, "label": _agent_label(agent_id), "role": "pursuer", "status": "active"}
    if is_evader(agent_id):
        status = "active" if agent_id in active_evaders else "captured"
        return {"agent_id": agent_id, "label": _agent_label(agent_id), "role": "evader", "status": status}
    return {"agent_id": agent_id, "label": agent_id, "role": "unknown", "status": "active"}


def _positions_for_frame(trace: PursuitTrace, frame_index: int) -> dict[str, list[int]]:
    if frame_index <= 0:
        return trace.summary.initial_positions
    if frame_index > len(trace.steps):
        frame_index = len(trace.steps)
    return trace.steps[frame_index - 1].agent_positions


def _agent_label(agent_id: str) -> str:
    if is_pursuer(agent_id):
        return f"P{agent_id.rsplit('_', 1)[-1]}"
    if is_evader(agent_id):
        return f"E{agent_id.rsplit('_', 1)[-1]}"
    return agent_id


def _label_sort_key(label: str) -> tuple[int, str]:
    return (0 if label.startswith("P") else 1, label)
