import torch

from strategy_games.models.ebm import EnergyMLP, contrastive_divergence_loss


def test_ebm_forward_shape() -> None:
    model = EnergyMLP(strategy_dim=8, hidden_dim=16)
    strategies = torch.randn(4, 8)
    energy = model(strategies)
    assert energy.shape == (4,)
    assert torch.isfinite(energy).all()


def test_contrastive_divergence_loss_is_scalar() -> None:
    model = EnergyMLP(strategy_dim=8, hidden_dim=16)
    positive = torch.randn(4, 8)
    negative = torch.randn(4, 8)
    loss = contrastive_divergence_loss(model, positive, negative)
    assert loss.shape == ()
    assert torch.isfinite(loss)
