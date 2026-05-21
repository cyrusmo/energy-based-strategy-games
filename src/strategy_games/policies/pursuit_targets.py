"""Policy target adapters for scripted and learned pursuit policies."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Protocol

import numpy as np
import torch
from torch import Tensor, nn
from torch.distributions import Categorical

from strategy_games.envs.multi_evader_pursuit import MultiEvaderPursuitEnv
from strategy_games.envs.pursuit_actions import ACTIONS
from strategy_games.models.pursuit_observation import (
    OBSERVATION_SCHEMA,
    PursuitObservationSpec,
    encode_pursuer_observation,
)
from strategy_games.policies.scripted_pursuit import scripted_pursuit_actions


@dataclass(frozen=True)
class PolicyTarget:
    """Serializable policy identity for comparison artifacts."""

    policy_id: str
    policy_type: Literal["scripted", "learned"]
    role: Literal["pursuer", "evader"]
    checkpoint_path: str | None = None
    training_run_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def public_dict(self) -> dict[str, Any]:
        """Return metadata suitable for public artifacts."""

        return {
            "policy_id": self.policy_id,
            "policy_type": self.policy_type,
            "role": self.role,
            "training_run_id": self.training_run_id,
            "metadata": dict(self.metadata),
        }


class PursuitPolicyAdapter(Protocol):
    """Minimal pursuit policy adapter interface."""

    target: PolicyTarget

    def act(self, env: MultiEvaderPursuitEnv, agent_id: str, step_index: int, rng: np.random.Generator) -> str:
        """Return one action label."""


class PursuitActorCritic(nn.Module):
    """Small actor-critic for discrete pursuit actions."""

    def __init__(self, obs_dim: int, action_dim: int = len(ACTIONS), hidden_dim: int = 64) -> None:
        super().__init__()
        if obs_dim < 1:
            raise ValueError("obs_dim must be positive")
        if action_dim != len(ACTIONS):
            raise ValueError("action_dim must match pursuit action space")
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.backbone = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )
        self.actor = nn.Linear(hidden_dim, action_dim)
        self.critic = nn.Linear(hidden_dim, 1)

    def forward(self, obs: Tensor) -> tuple[Tensor, Tensor]:
        """Return action logits and values."""

        if obs.ndim != 2:
            raise ValueError(f"obs must have shape [batch, obs_dim], got {tuple(obs.shape)}")
        features = self.backbone(obs.float())
        logits = self.actor(features)
        values = self.critic(features).squeeze(-1)
        return logits, values

    @torch.no_grad()
    def act(self, obs: Tensor, deterministic: bool = True) -> int:
        """Return an action index for one observation."""

        if obs.ndim == 1:
            obs = obs.unsqueeze(0)
        logits, _ = self.forward(obs)
        if deterministic:
            return int(torch.argmax(logits, dim=-1).item())
        return int(Categorical(logits=logits).sample().item())


class ScriptedPursuitPolicyAdapter:
    """Adapter around the existing scripted pursuit policy helpers."""

    def __init__(self, policy_id: str, role: Literal["pursuer", "evader"], feint_steps: int = 3) -> None:
        self.policy_id = policy_id
        self.role = role
        self.feint_steps = feint_steps
        self.target = PolicyTarget(policy_id=policy_id, policy_type="scripted", role=role)

    def act(self, env: MultiEvaderPursuitEnv, agent_id: str, step_index: int, rng: np.random.Generator) -> str:
        """Return one scripted action for ``agent_id``."""

        if self.role == "pursuer":
            actions = scripted_pursuit_actions(
                env=env,
                pursuer_policy=self.policy_id,
                evader_policy="evader_feint",
                rng=rng,
                step_index=step_index,
                feint_steps=self.feint_steps,
            )
        else:
            actions = scripted_pursuit_actions(
                env=env,
                pursuer_policy="pursuer_greedy_nearest",
                evader_policy=self.policy_id,
                rng=rng,
                step_index=step_index,
                feint_steps=self.feint_steps,
            )
        return str(actions[agent_id])


class LearnedPursuerPolicyAdapter:
    """Strict adapter for PPO pursuer checkpoints."""

    def __init__(self, checkpoint_path: str | Path, env: MultiEvaderPursuitEnv | None = None) -> None:
        self.checkpoint_path = Path(checkpoint_path)
        self.checkpoint = load_pursuit_policy_checkpoint(self.checkpoint_path)
        self.spec = observation_spec_from_checkpoint(self.checkpoint)
        self.model = PursuitActorCritic(
            obs_dim=self.spec.obs_dim,
            action_dim=len(ACTIONS),
            hidden_dim=int(self.checkpoint.get("hidden_dim", 64)),
        )
        self.model.load_state_dict(self.checkpoint["model_state_dict"])
        self.model.eval()
        self.target = PolicyTarget(
            policy_id=str(self.checkpoint["policy_id"]),
            policy_type="learned",
            role="pursuer",
            checkpoint_path=str(self.checkpoint_path),
            training_run_id=str(self.checkpoint.get("training_run_id")),
            metadata=checkpoint_public_metadata(self.checkpoint),
        )
        if env is not None:
            self.validate_env(env)

    def validate_env(self, env: MultiEvaderPursuitEnv) -> None:
        """Reject envs incompatible with the checkpoint observation contract."""

        if env.config.num_pursuers != 1:
            raise ValueError("learned pursuer checkpoint supports exactly one pursuer")
        if env.config.num_pursuers > self.spec.max_pursuers:
            raise ValueError("env has more pursuers than checkpoint supports")
        if env.config.num_evaders > self.spec.max_evaders:
            raise ValueError("env has more evaders than checkpoint supports")

    def act(self, env: MultiEvaderPursuitEnv, agent_id: str, step_index: int, rng: np.random.Generator) -> str:
        """Return deterministic action label from checkpoint policy."""

        del step_index, rng
        if agent_id != "pursuer_0":
            raise ValueError("learned pursuer adapter only controls pursuer_0")
        self.validate_env(env)
        obs = torch.as_tensor(encode_pursuer_observation(env, agent_id, self.spec), dtype=torch.float32)
        action_idx = self.model.act(obs, deterministic=True)
        return ACTIONS[action_idx]


def load_pursuit_policy_checkpoint(path: str | Path) -> dict[str, Any]:
    """Load and validate a learned pursuit policy checkpoint."""

    checkpoint = torch.load(Path(path), map_location="cpu")
    if not isinstance(checkpoint, dict):
        raise ValueError("checkpoint must contain a dictionary")
    required = {
        "model_state_dict",
        "observation_schema",
        "action_space",
        "max_pursuers",
        "max_evaders",
        "obs_dim",
        "policy_id",
        "training_run_id",
        "training_scope",
    }
    missing = sorted(required - set(checkpoint))
    if missing:
        raise ValueError(f"checkpoint missing keys: {missing}")
    if checkpoint["observation_schema"] != OBSERVATION_SCHEMA:
        raise ValueError(f"unsupported observation schema: {checkpoint['observation_schema']}")
    if list(checkpoint["action_space"]) != ACTIONS:
        raise ValueError("checkpoint action_space does not match pursuit actions")
    if int(checkpoint["max_pursuers"]) != 1:
        raise ValueError("checkpoint max_pursuers must be 1 for today's PPO path")
    if int(checkpoint["obs_dim"]) != PursuitObservationSpec(
        role="pursuer",
        max_pursuers=int(checkpoint["max_pursuers"]),
        max_evaders=int(checkpoint["max_evaders"]),
    ).obs_dim:
        raise ValueError("checkpoint obs_dim does not match observation spec")
    return checkpoint


def observation_spec_from_checkpoint(checkpoint: Mapping[str, Any]) -> PursuitObservationSpec:
    """Build an observation spec from checkpoint metadata."""

    return PursuitObservationSpec(
        role="pursuer",
        max_pursuers=int(checkpoint["max_pursuers"]),
        max_evaders=int(checkpoint["max_evaders"]),
    )


def checkpoint_public_metadata(checkpoint: Mapping[str, Any]) -> dict[str, Any]:
    """Return checkpoint metadata safe for public comparison artifacts."""

    return {
        "observation_schema": checkpoint["observation_schema"],
        "action_space": list(checkpoint["action_space"]),
        "max_pursuers": int(checkpoint["max_pursuers"]),
        "max_evaders": int(checkpoint["max_evaders"]),
        "obs_dim": int(checkpoint["obs_dim"]),
        "training_scope": dict(checkpoint["training_scope"]),
    }
