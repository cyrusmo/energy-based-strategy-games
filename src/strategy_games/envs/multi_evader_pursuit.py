"""Simple multi-evader pursuit/evasion gridworld for public trace demos."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

from strategy_games.envs.pursuit_actions import (
    ACTIONS,
    Position,
    all_agent_ids,
    clip_position,
    evader_ids,
    manhattan,
    move_position,
    pursuer_ids,
)
from strategy_games.traces.pursuit_trace import CaptureEvent


@dataclass(frozen=True)
class MultiEvaderPursuitConfig:
    """Configuration for the custom multi-evader pursuit gridworld."""

    grid_size: tuple[int, int] = (9, 9)
    num_evaders: int = 2
    num_pursuers: int = 1
    max_steps: int = 30
    catch_radius: int = 0
    pursuer_starts: tuple[Position, ...] | None = None
    evader_starts: tuple[Position, ...] | None = None
    evader_goals: tuple[Position, ...] | None = None
    evader_survival_reward: float = 1.0
    evader_capture_reward: float = -10.0
    evader_timeout_bonus: float = 10.0
    pursuer_step_reward: float = -0.1
    pursuer_capture_reward: float = 10.0
    pursuer_all_captured_bonus: float = 5.0


@dataclass(frozen=True)
class MultiEvaderStepResult:
    """Result returned by :meth:`MultiEvaderPursuitEnv.step`."""

    agent_positions: dict[str, Position]
    rewards: dict[str, float]
    captures: list[CaptureEvent]
    active_evaders: list[str]
    done: bool
    terminated_reason: str
    steps: int


@dataclass
class MultiEvaderPursuitEnv:
    """Deterministic gridworld with multiple pursuers and evaders.

    V1 dynamics are intentionally simple: all agents choose actions from the
    pre-step state, positions update simultaneously with boundary clipping, and
    captures are evaluated after movement only. Crossing paths are not captures.
    """

    config: MultiEvaderPursuitConfig = field(default_factory=MultiEvaderPursuitConfig)

    env_id = "custom_multi_evader_pursuit_v0"

    def __post_init__(self) -> None:
        self._validate_config()
        self.pursuer_ids = pursuer_ids(self.config.num_pursuers)
        self.evader_ids = evader_ids(self.config.num_evaders)
        self.agent_ids = all_agent_ids(self.config.num_pursuers, self.config.num_evaders)
        self.evader_goals = self._default_evader_goals()
        self.pursuer_positions: dict[str, Position] = {}
        self.evader_positions: dict[str, Position] = {}
        self.active_evaders: list[str] = []
        self.per_evader_status: dict[str, str] = {}
        self.per_agent_returns: dict[str, float] = {}
        self.steps = 0
        self.done = False
        self.terminated_reason = "unknown"
        self.reset()

    @property
    def grid_size(self) -> tuple[int, int]:
        """Return ``(height, width)`` grid size."""

        return self.config.grid_size

    def reset(self) -> dict[str, Position]:
        """Reset positions and episode bookkeeping."""

        self.pursuer_positions = dict(zip(self.pursuer_ids, self._default_pursuer_starts(), strict=True))
        self.evader_positions = dict(zip(self.evader_ids, self._default_evader_starts(), strict=True))
        self.active_evaders = list(self.evader_ids)
        self.per_evader_status = {evader_id: "survived" for evader_id in self.evader_ids}
        self.per_agent_returns = {agent_id: 0.0 for agent_id in self.agent_ids}
        self.steps = 0
        self.done = False
        self.terminated_reason = "unknown"
        return self.positions()

    def positions(self) -> dict[str, Position]:
        """Return positions for all agents."""

        return {**self.pursuer_positions, **self.evader_positions}

    def agent_roles(self) -> dict[str, str]:
        """Return roles for all agents."""

        return {
            **{agent_id: "pursuer" for agent_id in self.pursuer_ids},
            **{agent_id: "evader" for agent_id in self.evader_ids},
        }

    def step(self, actions: Mapping[str, str]) -> MultiEvaderStepResult:
        """Advance the environment by one simultaneous-action step."""

        if self.done:
            raise ValueError("cannot step a terminated environment; call reset first")
        normalized_actions = self._normalize_actions(actions)
        next_pursuers = {
            agent_id: move_position(position, normalized_actions[agent_id], self.grid_size)
            for agent_id, position in self.pursuer_positions.items()
        }
        next_evaders = {
            agent_id: (
                move_position(position, normalized_actions[agent_id], self.grid_size)
                if agent_id in self.active_evaders
                else position
            )
            for agent_id, position in self.evader_positions.items()
        }

        self.pursuer_positions = next_pursuers
        self.evader_positions = next_evaders
        self.steps += 1

        rewards = {agent_id: self.config.pursuer_step_reward for agent_id in self.pursuer_ids}
        rewards.update({agent_id: 0.0 for agent_id in self.evader_ids})
        captures = self._capture_events()
        captured_evaders = {capture.evader_id for capture in captures}
        for evader_id in self.active_evaders:
            if evader_id in captured_evaders:
                rewards[evader_id] += self.config.evader_capture_reward
                self.per_evader_status[evader_id] = "captured"
            else:
                rewards[evader_id] += self.config.evader_survival_reward

        for capture in captures:
            rewards[capture.pursuer_id] += self.config.pursuer_capture_reward

        self.active_evaders = [evader_id for evader_id in self.active_evaders if evader_id not in captured_evaders]

        all_captured = len(self.active_evaders) == 0
        timed_out = self.steps >= self.config.max_steps
        if all_captured:
            for pursuer_id in self.pursuer_ids:
                rewards[pursuer_id] += self.config.pursuer_all_captured_bonus
            self.done = True
            self.terminated_reason = "all_evaders_captured"
        elif timed_out:
            for evader_id in self.active_evaders:
                rewards[evader_id] += self.config.evader_timeout_bonus
            self.done = True
            self.terminated_reason = "timeout"
        else:
            self.done = False
            self.terminated_reason = "unknown"

        for agent_id, reward in rewards.items():
            self.per_agent_returns[agent_id] += float(reward)

        return MultiEvaderStepResult(
            agent_positions=self.positions(),
            rewards={agent_id: float(reward) for agent_id, reward in rewards.items()},
            captures=captures,
            active_evaders=list(self.active_evaders),
            done=self.done,
            terminated_reason=self.terminated_reason,
            steps=self.steps,
        )

    def _capture_events(self) -> list[CaptureEvent]:
        captures: list[CaptureEvent] = []
        for evader_id in self.active_evaders:
            evader_pos = self.evader_positions[evader_id]
            for pursuer_id in self.pursuer_ids:
                pursuer_pos = self.pursuer_positions[pursuer_id]
                if manhattan(evader_pos, pursuer_pos) <= self.config.catch_radius:
                    captures.append(
                        CaptureEvent(
                            pursuer_id=pursuer_id,
                            evader_id=evader_id,
                            position=[evader_pos[0], evader_pos[1]],
                            t=self.steps - 1,
                        )
                    )
                    break
        return captures

    def _normalize_actions(self, actions: Mapping[str, str]) -> dict[str, str]:
        normalized: dict[str, str] = {}
        for agent_id in self.agent_ids:
            action = str(actions.get(agent_id, "stay"))
            if action not in ACTIONS:
                raise ValueError(f"invalid action for {agent_id}: {action}")
            normalized[agent_id] = action
        return normalized

    def _default_pursuer_starts(self) -> tuple[Position, ...]:
        if self.config.pursuer_starts is not None:
            return self.config.pursuer_starts
        height, width = self.grid_size
        center = (height // 2, width // 2)
        starts = [center]
        offsets = [(0, -1), (0, 1), (-1, 0), (1, 0)]
        for dy, dx in offsets:
            if len(starts) >= self.config.num_pursuers:
                break
            starts.append(clip_position((center[0] + dy, center[1] + dx), self.grid_size))
        while len(starts) < self.config.num_pursuers:
            starts.append(center)
        return tuple(starts[: self.config.num_pursuers])

    def _default_evader_starts(self) -> tuple[Position, ...]:
        if self.config.evader_starts is not None:
            return self.config.evader_starts
        height, width = self.grid_size
        candidates = [
            (0, width - 1),
            (height - 1, width - 1),
            (0, 0),
            (height - 1, 0),
            (0, width // 2),
            (height - 1, width // 2),
            (height // 2, width - 1),
            (height // 2, 0),
        ]
        while len(candidates) < self.config.num_evaders:
            candidates.append((len(candidates) % height, width - 1))
        return tuple(candidates[: self.config.num_evaders])

    def _default_evader_goals(self) -> dict[str, Position]:
        raw_goals = self.config.evader_goals
        if raw_goals is None:
            height, width = self.grid_size
            goals = [(height - 1, 0), (0, 0), (height - 1, width - 1), (0, width - 1)]
            while len(goals) < self.config.num_evaders:
                goals.append((height - 1, len(goals) % width))
            raw_goals = tuple(goals[: self.config.num_evaders])
        return dict(zip(evader_ids(self.config.num_evaders), raw_goals, strict=True))

    def _validate_config(self) -> None:
        height, width = self.config.grid_size
        if height < 2 or width < 2:
            raise ValueError("grid_size must be at least 2x2")
        if self.config.num_evaders < 1:
            raise ValueError("num_evaders must be positive")
        if self.config.num_pursuers < 1:
            raise ValueError("num_pursuers must be positive")
        if self.config.max_steps < 1:
            raise ValueError("max_steps must be positive")
        if self.config.catch_radius < 0:
            raise ValueError("catch_radius must be non-negative")
        self._validate_positions(self.config.pursuer_starts, self.config.num_pursuers, "pursuer_starts")
        self._validate_positions(self.config.evader_starts, self.config.num_evaders, "evader_starts")
        self._validate_positions(self.config.evader_goals, self.config.num_evaders, "evader_goals")

    def _validate_positions(self, positions: tuple[Position, ...] | None, count: int, name: str) -> None:
        if positions is None:
            return
        if len(positions) != count:
            raise ValueError(f"{name} must contain exactly {count} positions")
        for position in positions:
            if clip_position(position, self.grid_size) != position:
                raise ValueError(f"{name} contains out-of-bounds position {position}")
