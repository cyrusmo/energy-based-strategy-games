"""Render-data helpers for pursuit trace viewers."""

from __future__ import annotations

import os
from pathlib import Path

from strategy_games.envs.pursuit_actions import is_evader, is_pursuer
from strategy_games.traces.pursuit_trace import PursuitTrace, validate_pursuit_trace


def grid_frame(trace: PursuitTrace, frame_index: int) -> list[list[str]]:
    """Return a text grid for a frame, where frame 0 is the initial state."""

    validate_pursuit_trace(trace)
    height, width = trace.grid_size
    positions = _positions_for_frame(trace, frame_index)
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
