import torch

from strategy_games.envs.layouts_v2 import get_layout_v2
from strategy_games.envs.gridworld_v2 import AttackerDefenderGridworldV2
from strategy_games.strategies.controllers import parametric_attacker_action, strategy_weights
from strategy_games.strategies.embeddings import named_strategy_embedding, strategy_entropy


def test_parametric_controller_returns_legal_action() -> None:
    env = AttackerDefenderGridworldV2(get_layout_v2("corridor_split"))
    env.reset()
    strategy = named_strategy_embedding("evasive", 8).vector
    action = parametric_attacker_action(env, strategy)
    assert action in range(env.action_dim)
    weights = strategy_weights(strategy)
    assert weights["evasiveness"] > weights["aggression"]


def test_strategy_entropy_is_zero_for_identical_vectors() -> None:
    stacked = torch.zeros(4, 8)
    assert strategy_entropy(stacked, k=2) == 0.0
    diverse = torch.eye(4, 8)
    assert strategy_entropy(diverse, k=4) > 0.0
