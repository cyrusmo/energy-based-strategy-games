import pytest
import torch

from strategy_games.models.ebm import EnergyMLP
from strategy_games.strategies.sampler import langevin_sample
from strategy_games.utils import device as device_utils


def test_resolve_device_cpu() -> None:
    assert device_utils.resolve_device("cpu").type == "cpu"


def test_resolve_device_mps_falls_back_when_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(device_utils, "mps_is_available", lambda: False)
    assert device_utils.resolve_device("mps").type == "cpu"


def test_move_models_places_parameters_on_device() -> None:
    model = EnergyMLP(strategy_dim=4, hidden_dim=8)
    (moved,) = device_utils.move_models(model, device="cpu")
    assert next(moved.parameters()).device.type == "cpu"
    assert next(moved.parameters()).dtype == torch.float32


def test_langevin_sample_cpu_shape() -> None:
    model = EnergyMLP(strategy_dim=4, hidden_dim=8)
    samples = langevin_sample(model, num_samples=3, strategy_dim=4, steps=1, device="cpu")
    assert samples.shape == (3, 4)
    assert samples.device.type == "cpu"


@pytest.mark.skipif(not device_utils.mps_is_available(), reason="MPS is not available")
def test_langevin_sample_mps_shape() -> None:
    model = EnergyMLP(strategy_dim=4, hidden_dim=8).to("mps")
    samples = langevin_sample(model, num_samples=3, strategy_dim=4, steps=1, device="mps")
    assert samples.shape == (3, 4)
    assert samples.device.type == "mps"
