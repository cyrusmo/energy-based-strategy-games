import json
from pathlib import Path

import numpy as np
import pytest
import torch

from strategy_games.envs.multi_evader_pursuit import MultiEvaderPursuitConfig, MultiEvaderPursuitEnv
from strategy_games.envs.pursuit_actions import ACTIONS
from strategy_games.models.pursuit_observation import OBSERVATION_SCHEMA, PursuitObservationSpec
from strategy_games.policies.pursuit_targets import LearnedPursuerPolicyAdapter, PursuitActorCritic
from strategy_games.training.ppo_pursuit import (
    PursuitPPOConfig,
    generalized_advantage_estimate,
    train_ppo_pursuer,
)


def tiny_ppo_config(tmp_path: Path) -> PursuitPPOConfig:
    return PursuitPPOConfig(
        seed=7,
        training_run_id="test_ppo_pursuer",
        policy_id="test_ppo_pursuer_v1",
        num_updates=1,
        rollout_steps=8,
        update_epochs=1,
        minibatch_size=4,
        hidden_dim=32,
        eval_seeds=(0, 1),
        output_dir=tmp_path / "public",
        checkpoint_path=tmp_path / "private" / "ppo_pursuer.pt",
        env=MultiEvaderPursuitConfig(
            grid_size=(5, 5),
            num_evaders=2,
            num_pursuers=1,
            max_steps=3,
            pursuer_starts=((2, 2),),
            evader_starts=((2, 4), (0, 4)),
            evader_goals=((4, 4), (4, 0)),
        ),
    )


def test_pursuit_actor_critic_output_shapes() -> None:
    model = PursuitActorCritic(obs_dim=20, action_dim=len(ACTIONS), hidden_dim=16)
    obs = torch.zeros(3, 20)
    logits, values = model(obs)

    assert logits.shape == (3, len(ACTIONS))
    assert values.shape == (3,)
    assert torch.isfinite(logits).all()
    assert torch.isfinite(values).all()


def test_advantage_computation_returns_finite_tensors() -> None:
    returns, advantages = generalized_advantage_estimate(
        rewards=torch.tensor([1.0, 0.5, -0.1]),
        dones=torch.tensor([0.0, 0.0, 1.0]),
        values=torch.tensor([0.2, 0.1, 0.0]),
        last_value=torch.tensor(0.0),
        gamma=0.99,
        gae_lambda=0.95,
    )

    assert returns.shape == (3,)
    assert advantages.shape == (3,)
    assert torch.isfinite(returns).all()
    assert torch.isfinite(advantages).all()


def test_tiny_ppo_pursuer_training_writes_public_metrics_and_private_checkpoint(tmp_path: Path) -> None:
    config = tiny_ppo_config(tmp_path)
    artifact = train_ppo_pursuer(config)
    metrics_path = config.output_dir / "metrics.json"
    config_path = config.output_dir / "config.json"

    assert metrics_path.exists()
    assert config_path.exists()
    assert config.checkpoint_path.exists()
    assert artifact["checkpoint_written"] is True

    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    public_config = json.loads(config_path.read_text(encoding="utf-8"))
    assert metrics["training_scope"]["controlled_agent_id"] == "pursuer_0"
    assert metrics["observation_schema"]["observation_schema"] == OBSERVATION_SCHEMA
    assert metrics["observation_schema"]["obs_dim"] == 20
    assert metrics["action_space"] == ACTIONS
    assert metrics["eval_seeds"] == [0, 1]
    assert "checkpoint_path" not in metrics
    assert "checkpoint_path" not in public_config
    assert metrics["train_metrics"]["updates"] == 1
    assert len(metrics["update_history"]) == 1
    assert "policy_loss" in metrics["update_history"][0]
    assert metrics["eval_metrics"]["episodes"] == pytest.approx(2.0)
    assert all(
        torch.isfinite(torch.tensor(float(metrics["eval_metrics"][key])))
        for key in ("capture_rate", "survival_rate", "mean_pursuer_return", "mean_evader_return", "average_steps")
    )

    checkpoint = torch.load(config.checkpoint_path, map_location="cpu")
    assert checkpoint["observation_schema"] == OBSERVATION_SCHEMA
    assert checkpoint["action_space"] == ACTIONS
    assert checkpoint["max_pursuers"] == 1
    assert checkpoint["max_evaders"] == 2
    assert checkpoint["obs_dim"] == 20
    assert checkpoint["training_scope"]["self_play"] is False


def test_learned_pursuer_adapter_loads_and_acts_deterministically(tmp_path: Path) -> None:
    config = tiny_ppo_config(tmp_path)
    train_ppo_pursuer(config)
    env = MultiEvaderPursuitEnv(config.env)
    adapter = LearnedPursuerPolicyAdapter(config.checkpoint_path, env=env)
    rng = np.random.default_rng(0)

    first = adapter.act(env, "pursuer_0", env.steps, rng)
    second = adapter.act(env, "pursuer_0", env.steps, rng)
    assert first in ACTIONS
    assert first == second
    assert adapter.target.public_dict()["policy_type"] == "learned"
    assert "checkpoint_path" not in adapter.target.public_dict()


def test_learned_pursuer_adapter_rejects_incompatible_metadata(tmp_path: Path) -> None:
    spec = PursuitObservationSpec(role="pursuer", max_pursuers=1, max_evaders=2)
    model = PursuitActorCritic(obs_dim=spec.obs_dim, action_dim=len(ACTIONS), hidden_dim=16)
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "observation_schema": OBSERVATION_SCHEMA,
        "action_space": ["stay", "up"],
        "max_pursuers": 1,
        "max_evaders": 2,
        "obs_dim": spec.obs_dim,
        "hidden_dim": 16,
        "policy_id": "bad_policy",
        "training_run_id": "bad_run",
        "training_scope": {"controlled_agent_id": "pursuer_0", "trained_role": "pursuer"},
    }
    path = tmp_path / "bad.pt"
    torch.save(checkpoint, path)

    with pytest.raises(ValueError, match="action_space"):
        LearnedPursuerPolicyAdapter(path)
