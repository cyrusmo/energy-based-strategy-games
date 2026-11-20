import pytest
import torch

from strategy_games.models.world_model import LearnedWorldModel


def test_rollout_shapes_and_composition() -> None:
    torch.manual_seed(0)
    model = LearnedWorldModel(state_dim=4, action_dim=5, strategy_dim=3, hidden_dim=8)
    state = torch.randn(6, 4)
    strategy = torch.randn(3)

    def policy(current: torch.Tensor, _strategy: torch.Tensor) -> torch.Tensor:
        return torch.full((current.shape[0],), 2)

    states, rewards = model.rollout(state, strategy, policy, horizon=3)
    assert states.shape == (6, 4, 4)
    assert rewards.shape == (6, 3)
    assert torch.equal(states[:, 0], state)
    # The first imagined step must equal a single forward prediction.
    one_hot = torch.nn.functional.one_hot(torch.full((6,), 2), num_classes=5).float()
    delta, reward = model(state, one_hot, strategy.expand(6, -1))
    assert torch.allclose(states[:, 1], state + delta)
    assert torch.allclose(rewards[:, 0], reward)


def test_rollout_rejects_bad_inputs() -> None:
    model = LearnedWorldModel(state_dim=4, action_dim=5, strategy_dim=3)
    with pytest.raises(ValueError):
        model.rollout(torch.zeros(2, 4), torch.zeros(3), lambda s, z: torch.zeros(2), horizon=0)
    with pytest.raises(ValueError):
        model.rollout(torch.zeros(4), torch.zeros(3), lambda s, z: torch.zeros(1), horizon=2)
