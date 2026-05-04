import torch
import pytest

from strategy_games.strategies.buffer import StrategyBuffer, StrategyRecord


def test_strategy_buffer_add_and_sample() -> None:
    buffer = StrategyBuffer(capacity=3)
    for idx in range(4):
        buffer.add(
            StrategyRecord(
                embedding=torch.ones(5) * idx,
                episode_return=float(idx),
                robustness_score=float(idx) / 2,
                exploitability_proxy=0.1,
                iteration=idx,
                label=f"s{idx}",
            )
        )
    assert len(buffer) == 3
    sample = buffer.sample_positive(batch_size=2)
    assert sample.shape == (2, 5)
    assert torch.isfinite(sample).all()
    assert buffer.diversity() > 0


def test_empty_strategy_buffer_sample_raises() -> None:
    buffer = StrategyBuffer()
    with pytest.raises(ValueError):
        buffer.sample_positive(batch_size=1)
