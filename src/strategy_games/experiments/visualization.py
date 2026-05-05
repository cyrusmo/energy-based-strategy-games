"""Rollout trace collection and simple gridworld visualization."""

# ruff: noqa: E402

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/strategy_games_mplconfig")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/strategy_games_cache")
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
Path(os.environ["XDG_CACHE_HOME"]).mkdir(parents=True, exist_ok=True)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from strategy_games.envs.gridworld import ACTION_NAMES, AttackerDefenderGridworld, GridworldConfig, Position
from strategy_games.evaluation.best_response import attacker_heuristic_action, defender_heuristic_action
from strategy_games.strategies.embeddings import named_strategy_embedding


@dataclass(frozen=True)
class RolloutTrace:
    """Public trace data for one deterministic heuristic rollout."""

    attacker_positions: list[Position]
    defender_positions: list[Position]
    goal_pos: Position
    attacker_actions: list[int]
    defender_actions: list[int]
    rewards: list[float]
    outcome: str
    total_return: float

    @property
    def steps(self) -> int:
        """Number of transitions in the trace."""

        return len(self.rewards)


def collect_heuristic_trace(
    strategy_label: str = "direct_goal",
    opponent_label: str = "aggressive",
    strategy_dim: int = 8,
    env_config: GridworldConfig | None = None,
) -> RolloutTrace:
    """Collect a deterministic trace for a named attacker and defender strategy."""

    env = AttackerDefenderGridworld(env_config or GridworldConfig())
    env.reset()
    strategy = named_strategy_embedding(strategy_label, strategy_dim).vector
    attacker_positions = [env.attacker_pos]
    defender_positions = [env.defender_pos]
    attacker_actions: list[int] = []
    defender_actions: list[int] = []
    rewards: list[float] = []
    outcome = "running"

    done = False
    while not done:
        attacker_action = attacker_heuristic_action(env, strategy, label=strategy_label)
        defender_action = defender_heuristic_action(env, opponent_label)
        result = env.step(attacker_action, defender_action)
        attacker_actions.append(attacker_action)
        defender_actions.append(defender_action)
        rewards.append(float(result.reward))
        attacker_positions.append(env.attacker_pos)
        defender_positions.append(env.defender_pos)
        outcome = str(result.info["outcome"])
        done = result.done

    return RolloutTrace(
        attacker_positions=attacker_positions,
        defender_positions=defender_positions,
        goal_pos=env.goal_pos,
        attacker_actions=attacker_actions,
        defender_actions=defender_actions,
        rewards=rewards,
        outcome=outcome,
        total_return=float(sum(rewards)),
    )


def format_trace_text(trace: RolloutTrace) -> str:
    """Format a rollout trace as public, line-oriented text."""

    lines = [
        "Rollout trace",
        f"initial_attacker={trace.attacker_positions[0]} initial_defender={trace.defender_positions[0]} goal={trace.goal_pos}",
    ]
    for step, reward in enumerate(trace.rewards):
        lines.append(
            "step={step:02d} attacker={attacker_before}->{attacker_after} defender={defender_before}->{defender_after} "
            "attacker_action={attacker_action} defender_action={defender_action} reward={reward:.3f}".format(
                step=step,
                attacker_before=trace.attacker_positions[step],
                attacker_after=trace.attacker_positions[step + 1],
                defender_before=trace.defender_positions[step],
                defender_after=trace.defender_positions[step + 1],
                attacker_action=ACTION_NAMES[trace.attacker_actions[step]],
                defender_action=ACTION_NAMES[trace.defender_actions[step]],
                reward=reward,
            )
        )
    lines.append(f"outcome={trace.outcome} total_return={trace.total_return:.3f} steps={trace.steps}")
    return "\n".join(lines) + "\n"


def save_trace_text(trace: RolloutTrace, path: str | Path) -> Path:
    """Write a formatted trace to disk."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(format_trace_text(trace), encoding="utf-8")
    return output_path


def plot_trace(trace: RolloutTrace, path: str | Path, grid_size: int = 10) -> Path:
    """Save a simple matplotlib path plot for a rollout trace."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    attacker = torch.tensor(trace.attacker_positions, dtype=torch.float32)
    defender = torch.tensor(trace.defender_positions, dtype=torch.float32)
    goal_y, goal_x = trace.goal_pos

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.set_xlim(-0.5, grid_size - 0.5)
    ax.set_ylim(grid_size - 0.5, -0.5)
    ax.set_xticks(range(grid_size))
    ax.set_yticks(range(grid_size))
    ax.grid(True, linewidth=0.5, alpha=0.5)
    ax.plot(attacker[:, 1], attacker[:, 0], marker="o", label="attacker", color="tab:blue")
    ax.plot(defender[:, 1], defender[:, 0], marker="s", label="defender", color="tab:red")
    ax.scatter([goal_x], [goal_y], marker="*", s=160, color="tab:green", label="goal")
    ax.set_title(f"outcome={trace.outcome}, return={trace.total_return:.2f}")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(output_path, dpi=140)
    plt.close(fig)
    return output_path
