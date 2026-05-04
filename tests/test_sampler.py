import torch

from strategy_games.models.ebm import EnergyMLP
from strategy_games.strategies.sampler import langevin_sample


def test_langevin_sampler_shape_and_finiteness() -> None:
    model = EnergyMLP(strategy_dim=6, hidden_dim=12)
    samples = langevin_sample(model, num_samples=5, strategy_dim=6, steps=3, step_size=0.01)
    assert samples.shape == (5, 6)
    assert torch.isfinite(samples).all()


def test_langevin_sampler_accepts_init() -> None:
    model = EnergyMLP(strategy_dim=3, hidden_dim=8)
    init = torch.zeros(2, 3)
    samples = langevin_sample(model, num_samples=2, strategy_dim=3, init=init, steps=1)
    assert samples.shape == (2, 3)
