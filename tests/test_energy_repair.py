import dataclasses

import numpy as np
import pytest
import torch

from strategy_games.envs.layouts_v2 import get_layout_v2
from strategy_games.experiments.energy_repair import (
    STRATEGIES,
    PositionTable,
    RepairConfig,
    _auroc,
    aggregate,
    build_position_table,
    commit_return,
    decision_energy,
    init_selector,
    random_rotation,
    repair_search,
    run_seed,
    selector_probs,
    selector_shapes,
    summary_markdown,
)

FAST = dataclasses.replace(
    RepairConfig(), hidden_dim=12, selector_steps=150, energy_steps=150, fault_steps=120, search_steps=6
)


@pytest.fixture(scope="module")
def table() -> PositionTable:
    return build_position_table(("corridor_split", "patient_hold"), max_positions=60, seed=0)


def test_commit_return_is_exact_on_simple_positions() -> None:
    config = get_layout_v2("corridor_split")
    # One step from the (0, 6) goal with the defender far away.
    assert commit_return(config, (0, 5), (6, 0), "direct_goal", "aggressive") == pytest.approx(0.99)
    for strategy in STRATEGIES:
        value = commit_return(config, config.attacker_start, config.defender_start, strategy, "aggressive")
        assert -1.0 + config.max_steps * config.step_penalty <= value <= 0.99


def test_position_table_shapes(table: PositionTable) -> None:
    assert table.features.shape == (120, 11 + 2)
    assert table.returns.shape == (120, len(STRATEGIES))
    assert set(np.unique(table.layout_ids)) == {0, 1}
    assert np.isfinite(table.returns).all()


def test_rotation_is_orthogonal_and_preserves_policy() -> None:
    rotation = random_rotation(40, seed=3)
    assert torch.allclose(rotation.T @ rotation, torch.eye(40), atol=1e-5)
    shapes = selector_shapes(4, 3, 5)
    theta = init_selector(shapes, torch.Generator().manual_seed(0))
    x = torch.randn(7, 4)
    rotation = random_rotation(theta.numel(), seed=1)
    assert torch.allclose(
        selector_probs(rotation @ (rotation.T @ theta), x, shapes), selector_probs(theta, x, shapes), atol=1e-5
    )


@pytest.mark.parametrize("rotated", [False, True])
@pytest.mark.parametrize("preserve", [False, True])
def test_repair_search_lowers_flagged_energy(rotated: bool, preserve: bool) -> None:
    generator = torch.Generator().manual_seed(0)
    shapes = selector_shapes(6, 8, 5)
    theta = init_selector(shapes, generator)
    x = torch.randn(64, 6, generator=generator)
    guidance = torch.rand(64, 5, generator=generator)
    basis = random_rotation(theta.numel(), seed=2) if rotated else None
    path = repair_search(theta, x, guidance, shapes, FAST, basis=basis, preserve=preserve)
    assert len(path) == FAST.search_steps + 1
    start = decision_energy(path[0], x, guidance, shapes)
    flagged = torch.topk(start, 16).indices
    assert decision_energy(path[-1], x, guidance, shapes)[flagged].mean() < start[flagged].mean()
    # Each step edits at most coords_per_step coordinates in the search basis.
    delta = path[1] - path[0]
    if basis is not None:
        delta = basis.T @ delta
    assert int((delta.abs() > 1e-7).sum()) <= FAST.coords_per_step


def test_auroc() -> None:
    assert _auroc(np.array([0.9, 0.8, 0.2, 0.1]), np.array([True, True, False, False])) == 1.0
    assert _auroc(np.array([0.1, 0.9]), np.array([True, False])) == 0.0


def test_run_seed_and_aggregate_smoke(table: PositionTable) -> None:
    results = [
        run_seed(table, seed, FAST, fault_strategies=("aggressive",), methods=("energy_guided", "position_blind"))
        for seed in (0, 1)
    ]
    assert results[0]["fault_layout"] == "corridor_split"
    assert results[1]["fault_layout"] == "patient_hold"
    fault = results[0]["faults"]["aggressive/blanket"]
    assert set(fault["arms"]) == {
        f"{flag}/{coords}/{method}"
        for flag in ("global", "fault_layout")
        for coords in ("original", "rotated")
        for method in ("energy_guided", "position_blind")
    }
    # The planted fault must actually make the selector overuse the strategy.
    assert fault["fault_strategy_rate_on_fault_layout"] > 0.6
    for arm in fault["arms"].values():
        assert 0.0 <= arm["final_harm"] <= arm["peak_harm"] <= 1.0
    summary = aggregate(results, FAST)
    assert summary["schema_version"] == "energy_repair/v2"
    assert summary["num_seeds"] == 2
    assert "aggressive/selective" in summary_markdown(summary)


def test_eligible_mask_restricts_flagging_and_protects_the_rest() -> None:
    generator = torch.Generator().manual_seed(0)
    shapes = selector_shapes(6, 8, 5)
    theta = init_selector(shapes, generator)
    x = torch.randn(64, 6, generator=generator)
    guidance = torch.rand(64, 5, generator=generator)
    eligible = torch.zeros(64, dtype=torch.bool)
    eligible[:16] = True
    path = repair_search(theta, x, guidance, shapes, FAST, eligible=eligible, flag_frac=0.5)
    change = decision_energy(path[-1], x, guidance, shapes) - decision_energy(path[0], x, guidance, shapes)
    assert change[:16].mean() < 0
    assert change[:16].mean() < change[16:].mean()
