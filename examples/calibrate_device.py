"""Calibrate CPU vs Apple MPS for the small experiment jobs in this repo."""

from __future__ import annotations

import argparse
import json
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch

from strategy_games.envs.gridworld import AttackerDefenderGridworld, GridworldConfig
from strategy_games.models.ebm import EnergyMLP, contrastive_divergence_loss
from strategy_games.models.policy import StrategyConditionedPolicy
from strategy_games.strategies.sampler import langevin_sample
from strategy_games.training.ppo_baseline import (
    ActorCriticPolicy,
    PPOConfig,
    bootstrap_value,
    collect_ppo_rollout,
    generalized_advantage_estimate,
    update_ppo_policy,
)
from strategy_games.training.train_loop import collect_policy_rollout
from strategy_games.utils.device import DEFAULT_CALIBRATION_PATH, mps_is_available
from strategy_games.utils.seeding import set_global_seed

BenchmarkFn = Callable[[torch.device], None]


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark CPU and MPS for repo workloads.")
    parser.add_argument("--output", type=Path, default=DEFAULT_CALIBRATION_PATH)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--runs", type=int, default=3)
    args = parser.parse_args()
    result = calibrate_devices(warmup=args.warmup, runs=args.runs)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(f"calibration_json={args.output}")


def calibrate_devices(warmup: int = 1, runs: int = 3) -> dict[str, Any]:
    """Run the calibration suite and return a JSON-safe payload."""

    if warmup < 0:
        raise ValueError("warmup must be non-negative")
    if runs < 1:
        raise ValueError("runs must be positive")
    set_global_seed(0)
    devices = [torch.device("cpu")]
    if mps_is_available():
        devices.append(torch.device("mps"))
    jobs: dict[str, BenchmarkFn] = {
        "langevin_sampling": _bench_langevin_sampling,
        "ebm_update": _bench_ebm_update,
        "policy_rollout": _bench_policy_rollout,
        "ppo_update": _bench_ppo_update,
    }
    rows = []
    for job, fn in jobs.items():
        timings: dict[str, float | None] = {}
        errors: dict[str, str] = {}
        for device in devices:
            try:
                timings[device.type] = _time_job(lambda device=device: fn(device), warmup=warmup, runs=runs)
            except Exception as exc:  # pragma: no cover - hardware-specific fallback path
                timings[device.type] = None
                errors[device.type] = str(exc)
        rows.append(_summarize_job(job, timings, errors))
    return {
        "schema_version": "device_calibration/v1",
        "torch_version": torch.__version__,
        "mps_available": mps_is_available(),
        "jobs": rows,
    }


def _time_job(fn: Callable[[], None], warmup: int, runs: int) -> float:
    for _ in range(warmup):
        fn()
        _synchronize()
    elapsed: list[float] = []
    for _ in range(runs):
        start = time.perf_counter()
        fn()
        _synchronize()
        elapsed.append((time.perf_counter() - start) * 1000.0)
    return float(sum(elapsed) / len(elapsed))


def _summarize_job(job: str, timings: dict[str, float | None], errors: dict[str, str]) -> dict[str, Any]:
    cpu_ms = timings.get("cpu")
    mps_ms = timings.get("mps")
    if mps_ms is None or cpu_ms is None:
        recommended = "cpu"
        speedup = 1.0
    elif mps_ms <= 0:
        recommended = "cpu"
        speedup = 1.0
    else:
        speedup = float(cpu_ms / mps_ms)
        recommended = "mps" if speedup > 1.05 else "cpu"
    return {
        "job": job,
        "cpu_ms": cpu_ms,
        "mps_ms": mps_ms,
        "speedup": speedup,
        "recommended_device": recommended,
        "mps_supported": mps_ms is not None,
        "errors": errors,
    }


def _bench_langevin_sampling(device: torch.device) -> None:
    ebm = EnergyMLP(8, hidden_dim=64).to(device=device, dtype=torch.float32)
    samples = langevin_sample(ebm, num_samples=16, strategy_dim=8, steps=10, step_size=0.02, device=device)
    _consume(samples)


def _bench_ebm_update(device: torch.device) -> None:
    ebm = EnergyMLP(8, hidden_dim=64).to(device=device, dtype=torch.float32)
    optimizer = torch.optim.Adam(ebm.parameters(), lr=1e-3)
    positive = torch.randn(16, 8, device=device)
    negative = langevin_sample(ebm, num_samples=16, strategy_dim=8, steps=5, step_size=0.02, device=device)
    loss = contrastive_divergence_loss(ebm, positive, negative)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    _consume(loss)


def _bench_policy_rollout(device: torch.device) -> None:
    env = AttackerDefenderGridworld(GridworldConfig(max_steps=12))
    policy = StrategyConditionedPolicy(env.state_dim, 8, env.action_dim).to(device=device, dtype=torch.float32)
    strategy = torch.randn(8, device=device)
    rollout = collect_policy_rollout(env, policy, strategy)
    _consume(torch.tensor(float(rollout.total_reward), device=device))


def _bench_ppo_update(device: torch.device) -> None:
    config = PPOConfig(total_steps=16, rollout_steps=16, update_epochs=1, minibatch_size=8, device=device.type)
    env = AttackerDefenderGridworld(config.env)
    policy = ActorCriticPolicy(env.state_dim, env.action_dim, hidden_dim=config.hidden_dim).to(
        device=device, dtype=torch.float32
    )
    optimizer = torch.optim.Adam(policy.parameters(), lr=config.learning_rate)
    observation = env.reset()
    batch = collect_ppo_rollout(env, policy, observation, steps=config.rollout_steps)
    last_value = bootstrap_value(policy, batch.last_observation, batch.last_done)
    returns, advantages = generalized_advantage_estimate(
        batch.rewards, batch.dones, batch.values, last_value, config.gamma, config.gae_lambda
    )
    stats = update_ppo_policy(policy, optimizer, batch, returns, advantages, config)
    _consume(torch.tensor(float(stats["policy_loss"]), device=device))


def _consume(value: torch.Tensor) -> None:
    if value.numel() > 0:
        _ = float(value.detach().cpu().reshape(-1)[0].item())


def _synchronize() -> None:
    if mps_is_available():
        torch.mps.synchronize()
    elif torch.cuda.is_available():
        torch.cuda.synchronize()


if __name__ == "__main__":
    main()
