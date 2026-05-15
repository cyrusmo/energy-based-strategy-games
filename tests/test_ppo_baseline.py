import json
import math

import torch
import yaml

from strategy_games.envs.gridworld import GridworldConfig
from strategy_games.training.ppo_baseline import (
    ActorCriticPolicy,
    PPOConfig,
    generalized_advantage_estimate,
    ppo_loss,
    train_ppo_baseline,
    train_ppo_from_config,
)


def test_actor_critic_output_shapes() -> None:
    policy = ActorCriticPolicy(state_dim=9, action_dim=5, hidden_dim=16)
    logits, values = policy(torch.zeros(3, 9))
    assert logits.shape == (3, 5)
    assert values.shape == (3,)


def test_advantage_and_return_computation_is_finite() -> None:
    returns, advantages = generalized_advantage_estimate(
        rewards=torch.tensor([0.1, 0.2, -1.0]),
        dones=torch.tensor([0.0, 0.0, 1.0]),
        values=torch.zeros(3),
        last_value=torch.zeros(()),
        gamma=0.99,
        gae_lambda=0.95,
    )
    assert returns.shape == (3,)
    assert advantages.shape == (3,)
    assert torch.isfinite(returns).all()
    assert torch.isfinite(advantages).all()


def test_ppo_loss_returns_finite_scalars() -> None:
    policy = ActorCriticPolicy(state_dim=9, action_dim=5, hidden_dim=16)
    loss, stats = ppo_loss(
        policy=policy,
        states=torch.zeros(4, 9),
        actions=torch.tensor([0, 1, 2, 3]),
        old_log_probs=torch.zeros(4),
        returns=torch.zeros(4),
        advantages=torch.ones(4),
        clip_range=0.2,
        value_coef=0.5,
        entropy_coef=0.01,
    )
    assert torch.isfinite(loss)
    for key in ("policy_loss", "value_loss", "entropy"):
        assert math.isfinite(float(stats[key]))


def test_tiny_ppo_training_smoke() -> None:
    result = train_ppo_baseline(_tiny_ppo_config())
    for key in (
        "episode_return",
        "win_rate",
        "goal_rate",
        "catch_rate",
        "timeout_rate",
        "policy_loss",
        "value_loss",
        "entropy",
        "episodes",
        "updates",
    ):
        assert key in result
    assert result["episodes"] == 1
    assert result["updates"] >= 1


def test_ppo_config_runner_writes_metrics(tmp_path) -> None:
    config_path = tmp_path / "ppo.yaml"
    output_dir = tmp_path / "ppo_outputs"
    config = {
        "seed": 3,
        "env": {
            "grid_size": 5,
            "max_steps": 8,
            "attacker_start": [0, 0],
            "defender_start": [4, 4],
            "goal_pos": [4, 0],
            "catch_radius": 0,
        },
        "ppo": {
            "total_steps": 16,
            "rollout_steps": 8,
            "update_epochs": 1,
            "minibatch_size": 4,
            "hidden_dim": 8,
            "eval_episodes": 1,
        },
        "logging": {"enabled": True, "output_dir": str(output_dir)},
    }
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

    result = train_ppo_from_config(config_path)
    metrics_path = output_dir / "metrics.json"
    assert result["artifacts"]["metrics_json"] == str(metrics_path)
    assert metrics_path.exists()
    saved = json.loads(metrics_path.read_text(encoding="utf-8"))
    assert saved["episodes"] == 1
    assert "policy_loss" in saved


def _tiny_ppo_config() -> PPOConfig:
    return PPOConfig(
        seed=7,
        total_steps=16,
        rollout_steps=8,
        update_epochs=1,
        minibatch_size=4,
        hidden_dim=8,
        eval_episodes=1,
        env=GridworldConfig(grid_size=5, max_steps=8, defender_start=(4, 4), goal_pos=(4, 0)),
    )
